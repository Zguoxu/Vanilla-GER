"""Download and hash-check the safetensors base model at an explicit revision."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from huggingface_hub import HfApi, snapshot_download

from vanilla_ger.common import resolve_path, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ger-qwen2.5-7b-smoke.json")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache/huggingface/hub"))
    parser.add_argument("--report", default="reports/qwen-model-download.json")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    model, revision = config["base_model"], config["model_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin model_revision to a full 40-character commit SHA")
    report = {"model": model, "revision": revision, "status": "downloading",
              "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}
    write_json(resolve_path(args.report), report)
    try:
        info = HfApi(token=False).model_info(model, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("Hub resolved a different model revision")
        weights = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
        if not weights:
            raise ValueError("No safetensors weights at this revision")
        print(f"Downloading {model}@{revision}, {sum(f.size for f in weights):,} weight bytes", flush=True)
        snapshot = Path(snapshot_download(
            repo_id=model, revision=revision, cache_dir=args.cache_dir, token=False,
            allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json",
                            "tokenizer.model", "LICENSE", "README.md"],
            max_workers=args.max_workers))
        verified = []
        for entry in weights:
            path = snapshot / entry.rfilename
            if path.stat().st_size != entry.size:
                raise ValueError(f"Weight size mismatch: {entry.rfilename}")
            actual = sha256_file(path)
            expected = entry.lfs.sha256 if entry.lfs else None
            if not expected or actual != expected:
                raise ValueError(f"Weight SHA-256 mismatch: {entry.rfilename}")
            verified.append({"name": entry.rfilename, "size_bytes": entry.size, "sha256": actual})
            print(f"Verified {entry.rfilename}", flush=True)
        report.update(status="verified", snapshot_path=str(snapshot), files=verified,
                      finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(resolve_path(args.report), report)


if __name__ == "__main__":
    main()
