import importlib.util
from pathlib import Path
import subprocess
import sys
import json
import numpy as np
import pytest
import soundfile as sf
import torch
from vanilla_ger.common import ROOT, read_records, write_json
from vanilla_ger.metrics import corpus_wer, word_errors, score_predictions
from vanilla_ger.upstream import prompter_class, normalize_text
from vanilla_ger.asr import select_candidates


def test_corpus_weighting_empty_predictions_and_id_checks():
    result = corpus_wer([("a b c d e f g h i j", "a b c d e f g h i x"), ("yes", "")])
    assert result["errors"] == 2
    assert result["reference_words"] == 11
    assert result["wer_percent"] == pytest.approx(200 / 11)
    assert word_errors("hello world", "")["deletions"] == 2
    assert word_errors("hello", "hello extra extra")["insertions"] == 2
    with pytest.raises(ValueError, match="Empty reference"):
        word_errors("", "something")
    with pytest.raises(ValueError, match="IDs differ"):
        score_predictions([{"utt_id": "a", "ref_text": "hello"}], [{"utt_id": "b", "prediction": "hello"}])


def test_prompt_paths_one_best_and_no_reference_leak(tmp_path, monkeypatch):
    from vanilla_ger.ger import prompt_for
    monkeypatch.chdir(tmp_path)
    prompter = prompter_class()()
    one = prompter.generate_prompt(input="hello world")
    assert "hello world" in one and "Other-hypothesis" not in one
    assert prompter.generate_prompt(input="hello world", input2="") == one
    record = {"input1": "hello world", "input2": "hello word", "output": "SECRET_REFERENCE"}
    assert "SECRET_REFERENCE" not in prompt_for(record, prompter)
    assert "Other-hypothesis" in prompt_for(record, prompter)
    assert prompter.get_response("a direct response") == "a direct response"


def test_candidates_sort_by_model_score_deduplicate_and_keep_numbers():
    candidates = [{"text": "There are twenty apples.", "avg_logprob": -2},
                  {"text": "There are 20 apples!", "avg_logprob": -1},
                  {"text": "There are ten apples.", "avg_logprob": -3}]
    selected = select_candidates(candidates, 5)
    assert len(selected) == 2
    assert selected[0]["avg_logprob"] == -1
    assert "20" in selected[0]["normalized"]
    assert normalize_text("There are 20 apples.") != normalize_text("There are 10 apples.")


def test_manifest_joins_by_id_and_does_not_require_clean(tmp_path, monkeypatch):
    import vanilla_ger.data as data
    monkeypatch.setattr(data, "ROOT", tmp_path)
    raw = tmp_path / "raw"
    wave_dir = raw / "noisy_trainset_28spk_wav"
    text_dir = raw / "trainset_28spk_txt"
    wave_dir.mkdir(parents=True)
    text_dir.mkdir()
    for index, uid in enumerate(["c_001", "a_001", "b_001"], 1):
        sf.write(wave_dir / f"{uid}.wav", np.linspace(0, index * .1, 800), 16000)
        (text_dir / f"{uid}.txt").write_text(f"transcript for {uid}")
    extra = raw / "noisy_testset_wav"
    extra.mkdir()
    sf.write(extra / "test_001.wav", np.ones(800), 16000)
    report = data.prepare(raw, tmp_path / "manifests", dev_speakers=["b"], expected_speakers=3)
    assert report["splits"]["train"]["utterances"] == 2
    assert report["dev_speakers"] == ["b"]
    train = read_records(tmp_path / "manifests/train.jsonl")
    assert train[0]["ref_text"] == "transcript for a_001"
    inputs = read_records(tmp_path / "data/audio_inputs/train.jsonl")
    assert not any("ref_text" in row for row in inputs)
    (text_dir / "a_001.txt").unlink()
    with pytest.raises(ValueError, match="ID mismatch"):
        data.prepare(raw, tmp_path / "other", dev_speakers=["b"], expected_speakers=3)


def test_h2t_export_matches_ids_and_separates_inference_labels():
    spec = importlib.util.spec_from_file_location("build_ger_data", ROOT / "scripts/build_ger_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hyp = [{"utt_id": "b", "nbest": ["bee", "be"]}, {"utt_id": "a", "nbest": ["ay"]}]
    refs = [{"utt_id": "a", "ref_text": "A"}, {"utt_id": "b", "ref_text": "B"}]
    labeled, unlabeled = module.build(hyp, refs)
    assert [r["output"] for r in labeled] == ["b", "a"]
    assert all("output" not in r for r in unlabeled)
    assert unlabeled[1]["input2"] == ""
    with pytest.raises(ValueError, match="do not match"):
        module.build(hyp[:1], refs)


@pytest.fixture
def tiny_base(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    vocab = {word: i for i, word in enumerate(["<pad>", "<s>", "</s>", "<unk>", "hello", "world", "good", "day", "Response", "Best", "hypothesis"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    base = tmp_path / "tiny-random-llama"
    tokenizer.save_pretrained(base)
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=1024, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    LlamaForCausalLM(config).save_pretrained(base)
    return base, tokenizer


def test_loss_mask_and_answer_truncation(tiny_base):
    from vanilla_ger.ger import tokenize_record
    _, tokenizer = tiny_base
    row = {"utt_id": "a", "input1": "hello world", "input2": "", "output": "good day"}
    prompter = prompter_class()()
    full = tokenize_record(row, tokenizer, prompter, train_on_inputs=True)
    masked = tokenize_record(row, tokenizer, prompter, train_on_inputs=False)
    assert -100 not in full["labels"]
    assert -100 in masked["labels"]
    assert masked["labels"][-1] == tokenizer.eos_token_id
    with pytest.raises(ValueError, match="refusing to truncate"):
        tokenize_record(row, tokenizer, prompter, cutoff_len=5)


def test_real_lora_update_save_reload_and_label_free_cli(tiny_base, tmp_path):
    from safetensors.torch import load_file
    from vanilla_ger.ger import train
    base, _ = tiny_base
    training = [{"utt_id": f"train-{i}", "input1": "hello world", "input2": "hello day", "output": "good day"} for i in range(4)]
    dev = [{"utt_id": "dev-1", "input1": "hello world", "input2": "hello day", "output": "good day"}]
    write_json(tmp_path / "train.json", training)
    write_json(tmp_path / "dev.json", dev)
    output = tmp_path / "adapter"
    metadata = train(str(base), str(tmp_path / "train.json"), str(output),
                     validation_path=str(tmp_path / "dev.json"), device="cpu", quantization_bits=0,
                     batch_size=2, micro_batch_size=1, max_steps=2, learning_rate=.02, warmup_steps=0,
                     lora_r=2, lora_alpha=4, gradient_checkpointing=False,
                     eval_beams=1, eval_max_new_tokens=4)
    weights = load_file(str(output / "adapter_model.safetensors"))
    assert any(torch.count_nonzero(value) for key, value in weights.items() if "lora_B" in key)
    assert metadata["best_dev_wer"] is not None
    inputs = tmp_path / "inputs.json"
    write_json(inputs, [{k: v for k, v in dev[0].items() if k != "output"}])
    predictions = tmp_path / "predictions.jsonl"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/infer_ger.py"),
                             "--checkpoint", str(output), "--inputs", str(inputs),
                             "--output", str(predictions), "--device", "cpu"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert read_records(predictions)[0]["utt_id"] == "dev-1"
    assert "prediction" in read_records(predictions)[0]
    rejected = subprocess.run([sys.executable, str(ROOT / "scripts/infer_ger.py"),
                               "--checkpoint", str(output), "--inputs", str(tmp_path / "dev.json"),
                               "--output", str(tmp_path / "should-not-exist.jsonl"), "--device", "cpu"],
                              capture_output=True, text=True, timeout=120)
    assert rejected.returncode != 0 and "contains labels" in rejected.stderr
    assert not (tmp_path / "should-not-exist.jsonl").exists()
    # Resume the real Trainer checkpoint on CPU: no fixed CUDA device is needed.
    resumed = train(str(base), str(tmp_path / "train.json"), str(output),
                    validation_path=str(tmp_path / "dev.json"), device="cpu", quantization_bits=0,
                    batch_size=2, micro_batch_size=1, max_steps=3, learning_rate=.02, warmup_steps=0,
                    lora_r=2, lora_alpha=4, gradient_checkpointing=False,
                    eval_beams=1, eval_max_new_tokens=4,
                    resume_from_checkpoint=str(output / "checkpoint-2"))
    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 3
    assert resumed["best_dev_wer"] is not None
