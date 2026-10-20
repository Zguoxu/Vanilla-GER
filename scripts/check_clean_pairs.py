"""Audit downloaded clean recordings against the official noisy recordings by ID."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from zoneinfo import ZoneInfo

import soundfile as sf

from vanilla_ger.common import ROOT, read_records, sha256_file, write_json
from vanilla_ger.data import index_files


def main():
    data_root = ROOT / "data/raw/voicebank_demand"
    manifests = {split: read_records(ROOT / f"data/manifests/{split}.jsonl")
                 for split in ["train", "dev", "test"]}
    split_ids = {split: {r["utt_id"] for r in rows} for split, rows in manifests.items()}
    groups, problems = {}, []
    for group, suffix, expected_count in [("official_train_28spk", "trainset_28spk_wav", 11572),
                                          ("official_test", "testset_wav", 824)]:
        clean = index_files(data_root / f"clean_{suffix}", ".wav")
        noisy = index_files(data_root / f"noisy_{suffix}", ".wav")
        if clean.keys() != noisy.keys() or len(clean) != expected_count:
            raise ValueError(f"{group} clean/noisy IDs or count differ: clean={len(clean)}, noisy={len(noisy)}")
        expected_ids = split_ids["train"] | split_ids["dev"] if group == "official_train_28spk" else split_ids["test"]
        if clean.keys() != expected_ids:
            raise ValueError(f"{group} IDs differ from the prepared manifests")

        def describe(uid):
            clean_info, noisy_info = sf.info(clean[uid]), sf.info(noisy[uid])
            issues = []
            if clean_info.frames <= 0 or clean_info.frames != noisy_info.frames:
                issues.append("frames")
            if clean_info.samplerate != noisy_info.samplerate or clean_info.samplerate != 48000:
                issues.append("sample_rate")
            if clean_info.channels != noisy_info.channels or clean_info.channels != 1:
                issues.append("channels")
            return {"utt_id": uid, "sample_rate": clean_info.samplerate,
                    "channels": clean_info.channels, "frames": clean_info.frames,
                    "duration_sec": clean_info.duration, "issues": issues}

        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(describe, sorted(clean)))
        problems.extend({"source_split": group, "utt_id": r["utt_id"], "issues": r["issues"]}
                        for r in rows if r["issues"])
        groups[group] = {"clean_directory": f"data/raw/voicebank_demand/clean_{suffix}",
                         "noisy_directory": f"data/raw/voicebank_demand/noisy_{suffix}",
                         "clean_files": len(clean), "noisy_files": len(noisy), "paired_files": len(rows),
                         "ids_match": True, "speakers": sorted({uid.split("_")[0] for uid in clean}),
                         "sample_rates": dict(Counter(str(r["sample_rate"]) for r in rows)),
                         "channels": dict(Counter(str(r["channels"]) for r in rows)),
                         "hours": sum(r["duration_sec"] for r in rows) / 3600,
                         "format_and_frame_mismatches": sum(bool(r["issues"]) for r in rows)}
    downloads = json.loads((ROOT / "reports/downloads.json").read_text())["files"]
    report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "status": "passed" if not problems else "failed", "groups": groups,
              "clean_files_by_experiment_split": {split: len(ids) for split, ids in split_ids.items()},
              "total_clean_files": sum(g["clean_files"] for g in groups.values()),
              "problems": problems,
              "archives": [item for item in downloads if item["name"].startswith("clean_")],
              "existing_input_sha256": {f"data/{folder}/{split}.jsonl": sha256_file(ROOT / f"data/{folder}/{split}.jsonl")
                                        for folder in ["manifests", "audio_inputs", "references"]
                                        for split in ["train", "dev", "test"]}}
    write_json(ROOT / "reports/voicebank-clean-audit.json", report)
    if problems:
        raise ValueError(f"Found {len(problems)} clean/noisy pairing issues; inspect reports/voicebank-clean-audit.json")
    print(json.dumps({"status": report["status"], "total_clean_files": report["total_clean_files"],
                      "pairs": {name: g["paired_files"] for name, g in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
