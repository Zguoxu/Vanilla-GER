"""Validate the three VoiceBank splits and the fixed large-v2 decode setup."""
import json
from itertools import combinations

from vanilla_ger.common import ROOT, read_records, sha256_file, write_json
from vanilla_ger.upstream import whisper_module


def main():
    config = json.loads((ROOT / "configs/asr-paper.json").read_text())
    expected = {"model": "large-v2", "device": "cuda", "beam_size": 60, "nbest": 5,
                "language": "en", "task": "transcribe", "temperature": 0.0,
                "without_timestamps": True}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Formal ASR configuration differs from the verified large-v2/beam60 setup")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("The formal ASR configuration requires an available CUDA GPU")
    splits, identities = {}, {}
    for split, count in [("train", 10830), ("dev", 742), ("test", 824)]:
        path = ROOT / f"data/audio_inputs/{split}.jsonl"
        inputs = read_records(path)
        manifests = read_records(ROOT / f"data/manifests/{split}.jsonl")
        references = read_records(ROOT / f"data/references/{split}.jsonl")
        if len(inputs) != count:
            raise ValueError(f"Expected {count} {split} inputs, found {len(inputs)}")
        source = {row["utt_id"]: row for row in manifests}
        ids = {row["utt_id"] for row in inputs}
        if ids != source.keys() or ids != {row["utt_id"] for row in references}:
            raise ValueError(f"{split} manifest/input/reference IDs differ")
        for row in inputs:
            if set(row) != {"utt_id", "noisy_path", "audio_sha256"}:
                raise ValueError(f"{split} ASR inputs must contain only audio identity, without labels")
            original = source[row["utt_id"]]
            if any(row[key] != original[key] for key in row):
                raise ValueError(f"Audio identity differs from audited manifest: {row['utt_id']}")
            expected_source = "official_test" if split == "test" else "official_train_28spk"
            if original["split"] != split or original["source_split"] != expected_source:
                raise ValueError(f"Unexpected split source for {row['utt_id']}")
            if not 0 < original["duration_sec"] <= 30:
                raise ValueError(f"Audio cannot fit the 30-second Whisper window: {row['utt_id']}")
            if original["channels"] != 1 or original["sample_rate"] != 48000:
                raise ValueError(f"Unexpected original VoiceBank audio format: {row['utt_id']}")
        identities[split] = {"ids": ids, "hashes": {r["audio_sha256"] for r in inputs},
                             "speakers": {r["speaker_id"] for r in manifests}}
        splits[split] = {"utterances": count, "input_sha256": sha256_file(path),
                         "hours": sum(r["duration_sec"] for r in manifests) / 3600,
                         "speakers": sorted(identities[split]["speakers"]), "labels_in_asr_input": False}
    for left, right in combinations(identities, 2):
        for key in ["ids", "hashes", "speakers"]:
            if identities[left][key] & identities[right][key]:
                raise ValueError(f"Cross-split {key} overlap: {left}/{right}")
    checkpoint = ROOT / ".cache/whisper/large-v2.pt"
    expected_sha256 = whisper_module()._MODELS["large-v2"].split("/")[-2]
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError("large-v2 checkpoint SHA-256 differs from its official identity")
    report = {"status": "ready", "config": config, "gpu": torch.cuda.get_device_name(0),
              "checkpoint_sha256": actual_sha256, "splits": splits,
              "total_utterances": sum(s["utterances"] for s in splits.values()),
              "cross_split_id_speaker_audio_overlap": False,
              "audio_loading": "FFmpeg decodes to mono 16 kHz; all source recordings fit 30 seconds.",
              "test_policy": "Generate test hypotheses separately; select models using train/dev only."}
    write_json(ROOT / "reports/asr-inputs-preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
