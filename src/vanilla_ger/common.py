import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/Hypo2Trans"
UPSTREAM_COMMIT = "27824d9da316b34855157dd9f6b80ea47119e804"


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_records(path):
    path = resolve_path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows = json.loads(text) if text.startswith("[") else [json.loads(s) for s in text.splitlines() if s.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of objects: {path}")
    ids = [row.get("utt_id") for row in rows]
    if any(not isinstance(uid, str) or not uid for uid in ids):
        raise ValueError(f"Missing utterance ID: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate utterance IDs: {path}")
    return rows


def write_json(path, value):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
