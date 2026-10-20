"""Verify all parts and every archived project file in a GitHub Release snapshot."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PartsReader:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def read(self, size):
        if size < 0:
            raise ValueError("This reader requires bounded streaming reads")
        chunks = []
        while size:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open("rb")
            chunk = self.current.read(size)
            if chunk:
                chunks.append(chunk)
                size -= len(chunk)
            else:
                self.current.close()
                self.current = None
        return b"".join(chunks)

    def close(self):
        if self.current:
            self.current.close()


def verify(folder):
    checksums = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        checksum, filename = line.split(maxsplit=1)
        filename = filename.lstrip("*")
        if Path(filename).name != filename:
            raise ValueError(f"Unexpected checksum path: {filename}")
        checksums[filename] = checksum
    parts = sorted(folder.glob("Vanilla-GER-full.tar.gz.part-*"))
    expected_parts = {name for name in checksums if name.startswith("Vanilla-GER-full.tar.gz.part-")}
    if not parts or {p.name for p in parts} != expected_parts:
        raise ValueError("Missing or unexpected archive parts")
    if "FILE_MANIFEST.jsonl" not in checksums:
        raise ValueError("The file manifest must have a checksum")
    for name, expected in checksums.items():
        if digest_file(folder / name) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        print(f"Checksum OK: {name}", flush=True)
    expected_files = {}
    with (folder / "FILE_MANIFEST.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["path"] in expected_files:
                raise ValueError(f"Duplicate manifest path: {row['path']}")
            expected_files[row["path"]] = row
    reader, found = PartsReader(parts), set()
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as bundle:
            for member in bundle:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.isdir():
                    continue
                if member.name in found or member.name not in expected_files:
                    raise ValueError(f"Duplicate or unexpected archive member: {member.name}")
                original = expected_files[member.name]
                if member.isfile():
                    digest = hashlib.sha256()
                    with bundle.extractfile(member) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if member.size != original["bytes"] or digest.hexdigest() != original["sha256"]:
                        raise ValueError(f"File content mismatch: {member.name}")
                elif member.issym():
                    if original.get("type") != "symlink" or original["target"] != member.linkname:
                        raise ValueError(f"Symlink mismatch: {member.name}")
                else:
                    raise ValueError(f"Unsupported archive entry: {member.name}")
                found.add(member.name)
                if len(found) % 5000 == 0:
                    print(f"Verified {len(found)} files", flush=True)
    finally:
        reader.close()
    if found != expected_files.keys():
        raise ValueError(f"Archive is missing {len(expected_files.keys() - found)} files")
    result = {"status": "verified", "files": len(found), "parts": len(parts),
              "archive_bytes": sum(p.stat().st_size for p in parts),
              "file_bytes": sum(r.get("bytes", 0) for r in expected_files.values())}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Directory containing the downloaded Release assets")
    parser.add_argument("--report", type=Path, help="Optional JSON verification result")
    args = parser.parse_args()
    result = verify(args.folder)
    if args.report:
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
