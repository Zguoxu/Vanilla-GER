import importlib.metadata
import json
import platform
import shutil
import sys
import torch
from vanilla_ger.common import ROOT, write_json

report = {
    "python": sys.version, "executable": sys.executable,
    "platform": platform.platform(), "architecture": platform.machine(),
    "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
    "mps_available": torch.backends.mps.is_available(), "ffmpeg": shutil.which("ffmpeg"),
    "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "numpy", "soundfile", "datasets", "accelerate"]},
}
if torch.cuda.is_available():
    report["gpu"] = [{"name": torch.cuda.get_device_name(i),
                      "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())]
name = "environment-macos.json" if platform.system() == "Darwin" else "environment-linux.json"
write_json(ROOT / "reports" / name, report)
print(json.dumps(report, indent=2, ensure_ascii=False))
