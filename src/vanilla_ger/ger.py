"""Adapted from Hypo2Trans H2T-LoRA (MIT); see THIRD_PARTY.md and patches/.

Keep the author's causal-LM/LoRA objective and H2T prompts, while removing
hardcoded CUDA devices, implicit validation splits, and label-dependent inference.
"""
import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
from .common import ROOT, UPSTREAM, UPSTREAM_COMMIT, read_records, resolve_path, sha256_file, write_json, write_jsonl
from .checkpoint import CheckpointResumeMixin
from .metrics import corpus_wer
from .upstream import normalize_text, prompter_class


def prompt_for(record, prompter, one_best=False):
    return prompter.generate_prompt(input=record["input1"], input2=None if one_best else record.get("input2") or None)


def tokenize_record(record, tokenizer, prompter, cutoff_len=512, train_on_inputs=True, one_best=False):
    label = record.get("output")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Training requires a nonempty reference output")
    prompt = prompt_for(record, prompter, one_best)
    encoded = tokenizer(prompt + label, add_special_tokens=True, truncation=False)
    tokenized = {key: encoded[key] for key in ["input_ids", "attention_mask"]}
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    if tokenized["input_ids"][-1] != tokenizer.eos_token_id:
        tokenized["input_ids"].append(tokenizer.eos_token_id)
        tokenized["attention_mask"].append(1)
    if len(tokenized["input_ids"]) > cutoff_len:
        raise ValueError(f'Sample {record.get("utt_id")} has {len(tokenized["input_ids"])} tokens > cutoff_len={cutoff_len}; refusing to truncate the answer')
    tokenized["labels"] = list(tokenized["input_ids"])
    if not train_on_inputs:
        prefix_length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        if prefix_length >= len(tokenized["labels"]):
            raise ValueError("No answer tokens remain for the response-only objective")
        tokenized["labels"][:prefix_length] = [-100] * prefix_length
    return tokenized


def load_base(base_model, quantization_bits=0, device=None, revision=None):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if quantization_bits not in [0, 4, 8]:
        raise ValueError("quantization_bits must be 0, 4, or 8")
    if quantization_bits and not device.startswith("cuda"):
        raise ValueError("This project's bitsandbytes path is for CUDA; use quantization_bits=0 for local CPU/MPS smoke tests")
    kwargs = {"revision": revision, "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32}
    if quantization_bits:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization_bits == 8, load_in_4bit=quantization_bits == 4,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if not quantization_bits:
        model.to(device)
    return model


def generate_text(model, tokenizer, prompter, record, one_best=False, num_beams=4, max_new_tokens=256):
    prompt = prompt_for(record, prompter, one_best)
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {key: encoded[key] for key in ["input_ids", "attention_mask"]}
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        sequence = model.generate(**inputs, do_sample=False, num_beams=num_beams,
                                  max_new_tokens=max_new_tokens, use_cache=True,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)[0]
    return tokenizer.decode(sequence[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class GerTrainer(CheckpointResumeMixin, Trainer):
    """Select checkpoints by generated dev corpus WER, not teacher-forced loss."""
    def __init__(self, *args, raw_dev, h2t_prompter, one_best, eval_beams, eval_max_new_tokens, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_dev, self.h2t_prompter, self.one_best = raw_dev, h2t_prompter, one_best
        self.eval_beams, self.eval_max_new_tokens = eval_beams, eval_max_new_tokens

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        was_training = self.model.training
        self.model.eval()
        pairs = []
        for row in self.raw_dev:
            prediction = generate_text(self.model, self.processing_class, self.h2t_prompter, row,
                                       self.one_best, self.eval_beams, self.eval_max_new_tokens)
            pairs.append((normalize_text(row["output"]), normalize_text(prediction)))
        metrics[f"{metric_key_prefix}_wer"] = corpus_wer(pairs)["wer_percent"]
        self.log({f"{metric_key_prefix}_wer": metrics[f"{metric_key_prefix}_wer"]})
        self.model.train(was_training)
        return metrics


def train(base_model: str, data_path: str, output_dir: str,
          validation_path: str = None, batch_size: int = 128, micro_batch_size: int = 1,
          num_epochs: int = 10, learning_rate: float = 1e-4, cutoff_len: int = 512,
          lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
          lora_target_modules=None, train_on_inputs: bool = True, one_best: bool = False,
          quantization_bits: int = 0, device: str = None, model_revision: str = None,
          seed: int = 42, warmup_steps: int = 100, max_steps: int = -1,
          gradient_checkpointing: bool = True, resume_from_checkpoint: str = None,
          eval_beams: int = 4, eval_max_new_tokens: int = 256):
    if batch_size < micro_batch_size or micro_batch_size < 1 or batch_size % micro_batch_size:
        raise ValueError("batch_size must be a positive multiple of micro_batch_size")
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This verified entry point supports one GPU; validate distributed training separately")
    set_seed(seed)
    rows = read_records(data_path)
    dev = read_records(validation_path) if validation_path else []
    if not rows:
        raise ValueError("Empty training data")
    if {r["utt_id"] for r in rows} & {r["utt_id"] for r in dev}:
        raise ValueError("Train/dev utterance IDs overlap")
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompter = prompter_class()()
    def encode(row):
        return tokenize_record(row, tokenizer, prompter, cutoff_len, train_on_inputs, one_best)
    train_data = Dataset.from_list([encode(r) for r in rows])
    dev_data = Dataset.from_list([encode(r) for r in dev]) if dev else None
    model = load_base(base_model, quantization_bits, device, model_revision)
    resolved_revision = getattr(model.config, "_commit_hash", None) or model_revision
    if quantization_bits:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                        target_modules=lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
                        lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    output_dir = str(resolve_path(output_dir))
    args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=micro_batch_size,
                             per_device_eval_batch_size=1,
                             gradient_accumulation_steps=batch_size // micro_batch_size,
                             learning_rate=learning_rate, num_train_epochs=num_epochs, max_steps=max_steps,
                             warmup_steps=warmup_steps, optim="adamw_torch", seed=seed, data_seed=seed,
                             fp16=next(model.parameters()).device.type == "cuda",
                             use_cpu=next(model.parameters()).device.type == "cpu",
                             eval_strategy="epoch" if dev else "no", save_strategy="epoch", save_total_limit=2,
                             load_best_model_at_end=bool(dev), metric_for_best_model="wer" if dev else None,
                             greater_is_better=False, report_to=[], logging_steps=1 if max_steps > 0 else 10,
                             dataloader_num_workers=0)
    trainer = GerTrainer(model=model, args=args, train_dataset=train_data, eval_dataset=dev_data,
                         processing_class=tokenizer, raw_dev=dev, h2t_prompter=prompter, one_best=one_best,
                         eval_beams=eval_beams, eval_max_new_tokens=eval_max_new_tokens,
                         data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()
    metadata = {"base_model": base_model, "model_revision": resolved_revision,
                "quantization_bits": quantization_bits, "one_best": one_best, "seed": seed,
                "train_on_inputs": train_on_inputs, "cutoff_len": cutoff_len,
                "batch_size": batch_size, "micro_batch_size": micro_batch_size,
                "learning_rate": learning_rate, "num_epochs": num_epochs, "max_steps": max_steps,
                "lora": config.to_dict(), "eval_beams": eval_beams, "eval_max_new_tokens": eval_max_new_tokens,
                "upstream_commit": UPSTREAM_COMMIT,
                "prompt_sha256": sha256_file(UPSTREAM / "H2T-LoRA/templates/H2T-LoRA.json"),
                "train_sha256": sha256_file(resolve_path(data_path)),
                "dev_sha256": sha256_file(resolve_path(validation_path)) if validation_path else None,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_dev_wer": trainer.state.best_metric}
    # PEFT configuration contains sets/enums; normalize to JSON primitives.
    metadata = json.loads(json.dumps(metadata, default=lambda value: sorted(value) if isinstance(value, set) else str(value)))
    write_json(Path(output_dir) / "training_metadata.json", metadata)
    return metadata


def inference_main():
    parser = argparse.ArgumentParser(description="H2T inference; labels belong in the separate scorer")
    parser.add_argument("--ckpt_path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--test_data_path", "--inputs", dest="inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantization-bits", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()
    checkpoint = resolve_path(args.checkpoint)
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    if args.base_model and args.base_model != metadata["base_model"]:
        raise ValueError("Base model differs from the recorded training base")
    if metadata["prompt_sha256"] != sha256_file(UPSTREAM / "H2T-LoRA/templates/H2T-LoRA.json"):
        raise ValueError("Prompt template changed since training")
    rows = read_records(args.inputs)
    if any(set(row) & {"output", "reference", "ref_text"} for row in rows):
        raise ValueError("Inference input contains labels; use the separate label-free export")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    bits = metadata["quantization_bits"] if args.quantization_bits is None else args.quantization_bits
    model = load_base(metadata["base_model"], bits, args.device, metadata["model_revision"])
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()
    prompter = prompter_class()()
    predictions = []
    for row in rows:
        prediction = generate_text(model, tokenizer, prompter, row, metadata["one_best"],
                                   args.num_beams if args.num_beams is not None else metadata["eval_beams"],
                                   args.max_new_tokens if args.max_new_tokens is not None else metadata["eval_max_new_tokens"])
        predictions.append({"utt_id": row["utt_id"], "prediction": prediction})
    write_jsonl(args.output, predictions)
    write_json(args.output + ".meta.json", {"checkpoint": str(checkpoint), "training_metadata": metadata,
                                           "inference_quantization_bits": bits,
                                           "num_beams": args.num_beams if args.num_beams is not None else metadata["eval_beams"],
                                           "max_new_tokens": args.max_new_tokens if args.max_new_tokens is not None else metadata["eval_max_new_tokens"]})
