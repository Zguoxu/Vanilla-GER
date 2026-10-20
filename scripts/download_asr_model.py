"""Download a pinned Whisper checkpoint with resumable transfer and SHA-256."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from vanilla_ger.common import ROOT, UPSTREAM_COMMIT, sha256_file, write_json
from vanilla_ger.upstream import whisper_module


def main():
    whisper = whisper_module()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="large-v2", choices=sorted(whisper._MODELS))
    args = parser.parse_args()
    url = whisper._MODELS[args.model]
    expected = url.split("/")[-2]
    target = ROOT / ".cache/whisper" / Path(url).name
    target.parent.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "reports" / f"whisper-{args.model}-download.json"
    report = {"model": args.model, "url": url, "expected_sha256": expected,
              "upstream_commit": UPSTREAM_COMMIT, "status": "downloading",
              "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}
    write_json(report_path, report)
    try:
        candidate = target if target.exists() else target.with_suffix(".pt.part")
        if candidate != target:
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "3",
                            "--connect-timeout", "30", "--silent", "--show-error", "--continue-at", "-",
                            "--output", str(candidate), url], check=True)
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {candidate}")
        if candidate != target:
            candidate.replace(target)
        report.update(status="verified", path=str(target), size_bytes=target.stat().st_size,
                      sha256=actual,
                      finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
        print(f"Verified {args.model}: {target}", flush=True)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(report_path, report)


if __name__ == "__main__":
    main()
