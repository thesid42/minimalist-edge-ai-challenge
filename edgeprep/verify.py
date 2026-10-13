import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import psutil
import torch


def main():
    modules = {"torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio",
               "onnx": "onnx", "onnxruntime": "onnxruntime", "thop": "thop",
               "codecarbon": "codecarbon", "psutil": "psutil", "qai-hub": "qai_hub",
               "matplotlib": "matplotlib", "torch-pruning": "torch_pruning",
               "scikit-learn": "sklearn", "onnxscript": "onnxscript"}
    checks = {}
    for package, module in modules.items():
        importlib.import_module(module)
        checks[package] = importlib.metadata.version(package)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10)).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8,16,device=device)), torch.arange(8,device=device))
    loss.backward()
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "ram_gib": psutil.virtual_memory().total/2**30, "logical_cpu_count": psutil.cpu_count(),
              "packages": checks, "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "optimizer_step": "passed", "qai_config_present": (Path.home()/".qai_hub"/"client.ini").exists()}
    target = Path("results") / ("environment_gpu.json" if device == "cuda" else "environment_cpu.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
