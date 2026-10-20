"""Download only the two requested official archives; verify before extracting."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def segmented_download(item, partial, connections):
    """Resume four bounded HTTP ranges, then atomically assemble the archive."""
    if partial.exists() and partial.stat().st_size == item["size_bytes"]:
        return
    cache = ROOT / ".cache/download_parts" / item["name"]
    cache.mkdir(parents=True, exist_ok=True)
    plan_path = cache / "plan.json"
    offset = partial.stat().st_size if partial.exists() else 0
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["offset"] != offset or plan["total"] != item["size_bytes"]:
            raise ValueError("Partial archive changed since the range plan; inspect the download cache")
    else:
        remaining = item["size_bytes"] - offset
        if remaining <= 0:
            raise ValueError("Partial archive is larger than expected")
        width = (remaining + connections - 1) // connections
        plan = {"offset": offset, "total": item["size_bytes"],
                "ranges": [[start, min(start + width, item["size_bytes"]) - 1]
                           for start in range(offset, item["size_bytes"], width)]}
        plan_path.write_text(json.dumps(plan))

    def fetch_range(entry):
        start, end = entry
        target = cache / f"{start}-{end}.part"
        for attempt in range(6):
            current = target.stat().st_size if target.exists() else 0
            if current == end - start + 1:
                return target
            if current > end - start + 1:
                raise ValueError(f"Oversized range: {target}")
            request_start = start + current
            request = urllib.request.Request(item["url"], headers={"Range": f"bytes={request_start}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected = f'bytes {request_start}-{end}/{item["size_bytes"]}'
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the requested bounded range")
                    with target.open("ab") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                if target.stat().st_size == end - start + 1:
                    return target
                raise OSError(f"Short range response: {target}")
            except (OSError, urllib.error.URLError):
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"Incomplete range: {target}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
        parts = list(pool.map(fetch_range, plan["ranges"]))
    assembled = partial.with_suffix(".assembled")
    with assembled.open("wb") as destination:
        if partial.exists():
            with partial.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if assembled.stat().st_size != item["size_bytes"]:
        raise ValueError("Assembled archive size differs from the official metadata")
    assembled.replace(partial)


def download(item, connections=1):
    folder = ROOT / "data/raw/archives"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / item["name"]
    partial = destination.with_suffix(destination.suffix + ".part")
    if not destination.exists():
        print(f'Downloading {item["name"]} ({item["size_bytes"]:,} bytes)', flush=True)
        if connections > 1 and item["size_bytes"] > 128 * 1024 * 1024:
            segmented_download(item, partial, connections)
        else:
            subprocess.run([
                "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "3",
                "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "90",
                "--silent", "--show-error", "--continue-at", "-", "--output", str(partial),
                item["url"],
            ], check=True)
        candidate = partial
    else:
        candidate = destination
    if candidate.stat().st_size != item["size_bytes"]:
        raise ValueError(f"Unexpected size: {candidate}")
    md5, sha256 = hashes(candidate)
    if md5 != item["md5"]:
        raise ValueError(f"Official MD5 mismatch: {candidate}; keep file for inspection")
    if candidate != destination:
        candidate.replace(destination)
    cache = ROOT / ".cache/download_parts" / item["name"]
    if cache.exists():
        for path in cache.iterdir():
            path.unlink()  # only temporary range files created by this downloader
        cache.rmdir()
    print(f'Verified {item["name"]}', flush=True)
    return {**item, "sha256": sha256}


def extract(item):
    archive = ROOT / "data/raw/archives" / item["name"]
    destination = ROOT / "data/raw/voicebank_demand"
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (base / member.filename).resolve().is_relative_to(base):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        expanded = sum(member.file_size for member in bundle.infolist())
        bundle.extractall(destination)
    print(f'Extracted {item["name"]}: {expanded:,} bytes', flush=True)
    return {**item, "uncompressed_bytes": expanded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--connections", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    files = json.loads((ROOT / "configs/downloads.json").read_text())["files"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(lambda item: download(item, args.connections), files))
    if args.extract:
        verified = [extract(item) for item in verified]
    target = ROOT / "reports/downloads.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"files": verified}, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {target}", flush=True)


if __name__ == "__main__":
    main()
