"""VoiceBank manifests use ID joins, not zip/file enumeration order."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from itertools import combinations
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


def prepare(data_root, output_dir, dev_speakers=None, expected_speakers=28,
            include_test=False, expected_test_utterances=824,
            expected_test_speakers=("p232", "p257")):
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

    jobs = [(uid, waves[uid], labels[uid],
             "dev" if uid.split("_")[0] in dev_speakers else "train", "official_train_28spk")
            for uid in sorted(waves)]
    if include_test:
        test_waves = index_files(data_root / "noisy_testset_wav", ".wav")
        test_labels = index_files(data_root / "testset_txt", ".txt")
        if test_waves.keys() != test_labels.keys():
            raise ValueError("Official test audio/text ID mismatch")
        if len(test_waves) != expected_test_utterances:
            raise ValueError(f"Expected {expected_test_utterances} test utterances, found {len(test_waves)}")
        test_speakers = sorted({uid.split("_")[0] for uid in test_waves})
        if test_speakers != sorted(expected_test_speakers):
            raise ValueError(f"Unexpected test speakers: {test_speakers}")
        if waves.keys() & test_waves.keys() or set(speakers) & set(test_speakers):
            raise ValueError("Official test IDs/speakers overlap official training")
        jobs.extend((uid, test_waves[uid], test_labels[uid], "test", "official_test")
                    for uid in sorted(test_waves))

    def describe(entry):
        uid, wave, label, split, source_split = entry
        ref = " ".join(label.read_text(encoding="utf-8-sig").split())
        if not ref:
            raise ValueError(f"Empty label: {label}")
        info = sf.info(wave)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Invalid audio: {wave}")
        return {"utt_id": uid, "speaker_id": uid.split("_")[0],
                "split": split,
                "source_split": source_split, "noisy_path": str(wave.relative_to(ROOT)) if wave.is_relative_to(ROOT) else str(wave),
                "ref_text": ref, "sample_rate": info.samplerate, "channels": info.channels,
                "duration_sec": info.duration, "frames": info.frames, "audio_sha256": sha256_file(wave)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(describe, jobs))
    splits = ["train", "dev", "test"] if include_test else ["train", "dev"]
    split_rows = {split: [r for r in rows if r["split"] == split] for split in splits}
    hashes = {split: {r["audio_sha256"] for r in items} for split, items in split_rows.items()}
    text_sets = {split: {r["ref_text"].casefold().strip() for r in items} for split, items in split_rows.items()}
    from .upstream import normalize_text, NORMALIZATION_VERSION
    normalized_sets = {split: {normalize_text(r["ref_text"]) for r in items} for split, items in split_rows.items()}
    if any("" in texts for texts in normalized_sets.values()):
        raise ValueError("A reference becomes empty after normalization; inspect labels before training")
    overlaps = {}
    for left, right in combinations(splits, 2):
        left_rows, right_rows = split_rows[left], split_rows[right]
        overlap = {
            "utterance_id_overlap": len({r["utt_id"] for r in left_rows} & {r["utt_id"] for r in right_rows}),
            "speaker_overlap": len({r["speaker_id"] for r in left_rows} & {r["speaker_id"] for r in right_rows}),
            "audio_hash_overlap": len(hashes[left] & hashes[right]),
            "casefolded_exact_text_overlap": len(text_sets[left] & text_sets[right]),
            "normalized_text_overlap": len(normalized_sets[left] & normalized_sets[right]),
        }
        if any(overlap[key] for key in ["utterance_id_overlap", "speaker_overlap", "audio_hash_overlap"]):
            raise ValueError(f"ID/speaker/audio overlap across {left}/{right}: {overlap}")
        overlaps[f"{left}_{right}"] = overlap
    # Publish inputs only after every split passes validation. ASR inputs carry no labels.
    for split, items in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", items)
        write_jsonl(ROOT / f"data/references/{split}.jsonl", [{"utt_id": r["utt_id"], "ref_text": r["ref_text"]} for r in items])
        write_jsonl(ROOT / f"data/audio_inputs/{split}.jsonl", [{k: r[k] for k in ["utt_id", "noisy_path", "audio_sha256"]} for r in items])
    report = {"source": "https://doi.org/10.7488/ds/2117", "official_train_utterances": len(waves),
              "official_test_utterances": len(split_rows.get("test", [])),
              "speakers": speakers, "dev_speakers": dev_speakers,
              "sample_rates": dict(Counter(str(r["sample_rate"]) for r in rows)),
              "channels": dict(Counter(str(r["channels"]) for r in rows)),
              "audio_hash_overlap": sum(pair["audio_hash_overlap"] for pair in overlaps.values()),
              "casefolded_exact_text_overlap": len(text_sets["train"] & text_sets["dev"]),
              "normalized_text_overlap": len(normalized_sets["train"] & normalized_sets["dev"]),
              "normalization_version": NORMALIZATION_VERSION,
              "test_downloaded": all((data_root / name).is_dir() for name in ["noisy_testset_wav", "testset_txt"]),
              "test_prepared": include_test,
              "pairwise_overlap": overlaps,
              "clean_audio_downloaded": (data_root / "clean_trainset_28spk_wav").is_dir(),
              "splits": {s: {"utterances": len(rs), "hours": sum(r["duration_sec"] for r in rs)/3600,
                              "speakers": sorted({r["speaker_id"] for r in rs}),
                              "min_duration_sec": min(r["duration_sec"] for r in rs),
                              "max_duration_sec": max(r["duration_sec"] for r in rs)} for s, rs in split_rows.items()}}
    write_json(ROOT / "reports/data_audit.json", report)
    split_config = {"rule": "lexicographically first two official-train speakers as dev", "dev_speakers": dev_speakers, "speakers": speakers}
    if include_test:
        split_config.update(test_source="official_test", test_speakers=test_speakers)
    write_json(ROOT / "configs/splits.json", split_config)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/voicebank_demand")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--include-test", action="store_true", help="Validate and prepare the separate official 824-utterance test split")
    args = parser.parse_args()
    report = prepare(args.data_root, args.output_dir, include_test=args.include_test)
    print({s: d["utterances"] for s, d in report["splits"].items()})
