"""Restore the pinned official checkout and apply the project's tracked fixes."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "third_party/Hypo2Trans"
COMMIT = "27824d9da316b34855157dd9f6b80ea47119e804"


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(DESTINATION), *args], check=check, capture_output=True, text=True)


def main():
    if not DESTINATION.exists():
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/Hypotheses-Paradise/Hypo2Trans.git", str(DESTINATION)], check=True)
        git("checkout", "--detach", COMMIT)
    actual = git("rev-parse", "HEAD").stdout.strip()
    if actual != COMMIT:
        raise RuntimeError(f"Existing checkout is at {actual}; expected {COMMIT}. Inspect it before changing revisions.")
    patch = ROOT / "patches/hypo2trans.patch"
    if not patch.is_file():
        raise FileNotFoundError(patch)
    if git("apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        print(f"Official source {COMMIT}: project patch already applied")
        return
    git("apply", "--check", str(patch))
    git("apply", str(patch))
    print(f"Official source {COMMIT}: project patch applied")


if __name__ == "__main__":
    main()
