import importlib.metadata
import json
import platform
import shutil
import sys
import torch
from vanilla_ger.common import ROOT, write_json


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


report = {
    "python": sys.version, "executable": sys.executable,
    "platform": platform.platform(), "architecture": platform.machine(),
    "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
    "mps_available": torch.backends.mps.is_available(), "ffmpeg": shutil.which("ffmpeg"),
    "packages": {p: package_version(p) for p in ["torch", "torchaudio", "transformers", "peft", "numpy", "soundfile", "datasets", "accelerate", "bitsandbytes"]},
}
report["missing_packages"] = [name for name, version in report["packages"].items() if version is None]
if torch.cuda.is_available():
    report["gpu"] = [{"name": torch.cuda.get_device_name(i),
                      "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())]
name = "environment-macos.json" if platform.system() == "Darwin" else "environment-linux.json"
write_json(ROOT / "reports" / name, report)
print(json.dumps(report, indent=2, ensure_ascii=False))
