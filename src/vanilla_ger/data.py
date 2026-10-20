"""VoiceBank manifests use ID joins, not zip/file enumeration order."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import soundfile as sf
from .common import ROOT, resolve_path, write_json, write_jsonl, sha256_file


def index_files(root, suffix):
    result = {}
    for path in sorted(Path(root).rglob("*" + suffix)):
        if any(part.startswith(".") or part == "__MACOSX" for part in path.relative_to(root).parts):
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate ID {path.stem}: {path} and {result[path.stem]}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"No {suffix} files found in {root}")
    return result


def prepare(data_root, output_dir, dev_speakers=None, expected_speakers=28):
    data_root, output_dir = resolve_path(data_root), resolve_path(output_dir)
    wave_root = data_root / "noisy_trainset_28spk_wav"
    text_root = data_root / "trainset_28spk_txt"
    # Read the official training folders explicitly: later clean/test downloads
    # must not be swept into the training corpus by a recursive root scan.
    if wave_root.exists() or text_root.exists():
        if not wave_root.is_dir() or not text_root.is_dir():
            raise ValueError("Both official training audio and transcript folders are required")
    else:
        wave_root = text_root = data_root  # flat miniature fixtures
    waves, labels = index_files(wave_root, ".wav"), index_files(text_root, ".txt")
    if waves.keys() != labels.keys():
        raise ValueError(f"Audio/text ID mismatch: missing_labels={list(waves.keys()-labels.keys())[:10]}, missing_audio={list(labels.keys()-waves.keys())[:10]}")
    speakers = sorted({uid.split("_")[0] for uid in waves})
    if len(speakers) != expected_speakers:
        raise ValueError(f"Expected {expected_speakers} speakers, found {len(speakers)}")
    dev_speakers = sorted(dev_speakers or speakers[:2])
    if not set(dev_speakers) < set(speakers):
        raise ValueError("Dev speakers must be a nonempty proper subset of train speakers.")

    def describe(uid):
        wave, label = waves[uid], labels[uid]
        ref = " ".join(label.read_text(encoding="utf-8-sig").split())
        if not ref:
            raise ValueError(f"Empty label: {label}")
        info = sf.info(wave)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Invalid audio: {wave}")
        return {"utt_id": uid, "speaker_id": uid.split("_")[0],
                "split": "dev" if uid.split("_")[0] in dev_speakers else "train",
                "source_split": "official_train_28spk", "noisy_path": str(wave.relative_to(ROOT)) if wave.is_relative_to(ROOT) else str(wave),
                "ref_text": ref, "sample_rate": info.samplerate, "channels": info.channels,
                "duration_sec": info.duration, "frames": info.frames, "audio_sha256": sha256_file(wave)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(describe, sorted(waves)))
    split_rows = {split: [r for r in rows if r["split"] == split] for split in ["train", "dev"]}
    for split, items in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", items)
        write_jsonl(ROOT / f"data/references/{split}.jsonl", [{"utt_id": r["utt_id"], "ref_text": r["ref_text"]} for r in items])
        write_jsonl(ROOT / f"data/audio_inputs/{split}.jsonl", [{k: r[k] for k in ["utt_id", "noisy_path", "audio_sha256"]} for r in items])
    hashes = {split: {r["audio_sha256"] for r in items} for split, items in split_rows.items()}
    collision = hashes["train"] & hashes["dev"]
    if collision:
        raise ValueError(f"Identical audio hashes cross train/dev: {len(collision)}")
    text_sets = {split: {r["ref_text"].casefold().strip() for r in items} for split, items in split_rows.items()}
    from .upstream import normalize_text, NORMALIZATION_VERSION
    normalized_sets = {split: {normalize_text(r["ref_text"]) for r in items} for split, items in split_rows.items()}
    if any("" in texts for texts in normalized_sets.values()):
        raise ValueError("A reference becomes empty after normalization; inspect labels before training")
    report = {"source": "https://doi.org/10.7488/ds/2117", "official_train_utterances": len(rows),
              "speakers": speakers, "dev_speakers": dev_speakers,
              "sample_rates": dict(Counter(str(r["sample_rate"]) for r in rows)),
              "channels": dict(Counter(str(r["channels"]) for r in rows)),
              "audio_hash_overlap": len(collision),
              "casefolded_exact_text_overlap": len(text_sets["train"] & text_sets["dev"]),
              "normalized_text_overlap": len(normalized_sets["train"] & normalized_sets["dev"]),
              "normalization_version": NORMALIZATION_VERSION,
              "test_downloaded": (data_root / "noisy_testset_wav").is_dir(),
              "clean_audio_downloaded": (data_root / "clean_trainset_28spk_wav").is_dir(),
              "splits": {s: {"utterances": len(rs), "hours": sum(r["duration_sec"] for r in rs)/3600,
                              "speakers": sorted({r["speaker_id"] for r in rs}),
                              "min_duration_sec": min(r["duration_sec"] for r in rs),
                              "max_duration_sec": max(r["duration_sec"] for r in rs)} for s, rs in split_rows.items()}}
    write_json(ROOT / "reports/data_audit.json", report)
    write_json(ROOT / "configs/splits.json", {"rule": "lexicographically first two official-train speakers as dev", "dev_speakers": dev_speakers, "speakers": speakers})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/voicebank_demand")
    parser.add_argument("--output-dir", default="data/manifests")
    args = parser.parse_args()
    report = prepare(args.data_root, args.output_dir)
    print({s: d["utterances"] for s, d in report["splits"].items()})
