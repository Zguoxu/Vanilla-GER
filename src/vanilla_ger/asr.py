"""Generate scored N-best hypotheses with the author's patched Whisper fork."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from .common import ROOT, UPSTREAM_COMMIT, read_records, resolve_path, sha256_file, write_json
from .upstream import whisper_module, normalize_text, NORMALIZATION_VERSION


def select_candidates(candidates, nbest):
    if nbest < 1:
        raise ValueError("nbest must be positive")
    selected, seen = [], set()
    for candidate in sorted(candidates, key=lambda c: c["avg_logprob"], reverse=True):
        score = candidate["avg_logprob"]
        if not math.isfinite(score):
            raise ValueError("ASR returned a non-finite model score")
        text = normalize_text(candidate["text"])
        if text and text not in seen:
            selected.append({**candidate, "normalized": text})
            seen.add(text)
        if len(selected) == nbest:
            break
    if not selected:
        raise ValueError("ASR returned no nonempty candidate; inspect the audio")
    return selected


def generate(manifest, output, model_name="tiny.en", beam_size=5, nbest=5, device="cpu", limit=10, sample_len=None):
    if beam_size < nbest or nbest < 1:
        raise ValueError("beam_size must be >= nbest >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive, or omitted for the full split")
    rows = read_records(manifest)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty audio manifest")
    whisper = whisper_module()
    output = resolve_path(output)
    audio_identity = [{k: row[k] for k in ["utt_id", "noisy_path", "audio_sha256"]} for row in rows]
    config = {"model": model_name, "beam_size": beam_size, "nbest": nbest, "device": device,
              "language": "en", "task": "transcribe", "temperature": 0.0,
              "without_timestamps": True, "sample_len": sample_len, "ranking": "sum_logprob/(tokens+1)",
              "normalization_version": NORMALIZATION_VERSION,
              "checkpoint_sha256": whisper._MODELS[model_name].split("/")[-2] if model_name in whisper._MODELS else sha256_file(resolve_path(model_name)),
              "upstream_commit": UPSTREAM_COMMIT,
              "decoder_sha256": sha256_file(ROOT / "third_party/Hypo2Trans/generate_data/whisper/whisper/decoding.py"),
              "model_sha256": sha256_file(ROOT / "third_party/Hypo2Trans/generate_data/whisper/whisper/model.py"),
              "audio_inputs_sha256": hashlib.sha256(json.dumps(audio_identity, sort_keys=True).encode()).hexdigest()}
    meta_path = output.with_suffix(".meta.json")
    if meta_path.exists() and json.loads(meta_path.read_text())["config"] != config:
        raise ValueError("Existing output belongs to a different configuration; choose another output path")
    if output.exists() and not meta_path.exists():
        raise ValueError("Existing hypotheses have no provenance sidecar; inspect before resuming")
    completed_rows = read_records(output) if output.exists() else []
    completed = {row["utt_id"] for row in completed_rows}
    if not completed <= {row["utt_id"] for row in rows}:
        raise ValueError("Output contains IDs outside the selected audio manifest")
    if len(completed) == len(rows):
        print(f"Already complete: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(meta_path, {"config": config, "status": "running"})
    torch.manual_seed(42)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    load_started = time.monotonic()
    model = whisper.load_model(model_name, device=device, download_root=str(ROOT / ".cache/whisper"))
    model.eval()
    model_load_seconds = time.monotonic() - load_started
    options = whisper.DecodingOptions(language="en", task="transcribe", beam_size=beam_size,
                                      temperature=0.0, without_timestamps=True, fp16=device.startswith("cuda"),
                                      sample_len=sample_len, return_scores=True)
    started = time.monotonic()
    generated = 0
    with output.open("a", encoding="utf-8") as stream:
        for row in rows:
            if row["utt_id"] in completed:
                continue
            path = resolve_path(row["noisy_path"])
            if sha256_file(path) != row["audio_sha256"]:
                raise ValueError(f"Audio content changed: {path}")
            audio = whisper.load_audio(str(path))
            if len(audio) > whisper.audio.N_SAMPLES:
                raise ValueError(f"Audio exceeds the 30-second decoder window; refusing to trim: {path}")
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(model.device)
            candidates = whisper.decode(model, mel, options)
            selected = select_candidates(candidates, nbest)
            result = {"utt_id": row["utt_id"], "nbest": [c["normalized"] for c in selected],
                      "candidates": selected, "num_unique": len(selected)}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            generated += 1
            print(f'{row["utt_id"]}: {len(selected)} unique candidates; 1-best={selected[0]["normalized"]}', flush=True)
    device_metrics = {}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        device_metrics = {"gpu": torch.cuda.get_device_name(model.device),
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
                          "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(model.device)}
    seconds = time.monotonic() - started
    write_json(meta_path, {"config": config, "status": "complete", "utterances": len(rows),
                           "generated_this_run": generated, "seconds_this_run": seconds,
                           "seconds_per_utterance_this_run": seconds / max(1, generated),
                           "model_load_seconds": model_load_seconds, **device_metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON config; explicit CLI flags override it")
    parser.add_argument("--manifest", required=True, help="Unlabeled audio_inputs JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--nbest", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Use 10 for local smoke; omit for full split")
    parser.add_argument("--sample-len", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text()) if args.config else {}
    for key, expected in {"language": "en", "task": "transcribe", "temperature": 0.0, "without_timestamps": True}.items():
        if key in config and config[key] != expected:
            parser.error(f"This verified entry point requires {key}={expected!r}")
    def value(key, fallback):
        explicit = getattr(args, key)
        return explicit if explicit is not None else config.get(key, fallback)
    generate(args.manifest, args.output, value("model", "tiny.en"), value("beam_size", 5),
             value("nbest", 5), value("device", "cpu"), value("limit", None), value("sample_len", None))
