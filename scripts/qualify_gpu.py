from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/environment/gpu_environment_qualification.json"


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def main() -> None:
    rng = np.random.default_rng(20260905)
    features = rng.normal(size=(20_000, 24)).astype(np.float32)
    labels = (features[:, 0] + 0.7 * features[:, 1] > 0).astype(np.float32)
    matrix = xgb.DMatrix(features, label=labels)
    start = time.perf_counter()
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": 4,
            "seed": 20260905,
        },
        matrix,
        num_boost_round=12,
    )
    xgb_seconds = time.perf_counter() - start
    prediction = booster.predict(matrix)

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    torch.manual_seed(20260905)
    torch.cuda.manual_seed_all(20260905)
    left = torch.randn((2048, 2048), device="cuda")
    right = torch.randn((2048, 2048), device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    product = left @ right
    torch.cuda.synchronize()
    torch_seconds = time.perf_counter() - start
    if not torch.isfinite(product).all().item():
        raise RuntimeError("non-finite CUDA matrix multiplication")

    report = {
        "phase": "P01_ENVIRONMENT",
        "timestamp": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "xgboost_version": xgb.__version__,
        "xgboost_build_info": xgb.build_info(),
        "xgboost_cuda_training": {
            "pass": bool(np.isfinite(prediction).all()),
            "rows": len(features),
            "features": features.shape[1],
            "seconds": xgb_seconds,
            "parameters": {"tree_method": "hist", "device": "cuda"},
        },
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_synchronized_matmul": {
            "pass": True,
            "shape": [2048, 2048],
            "seconds": torch_seconds,
        },
        "pip": importlib.metadata.version("pip"),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "pass": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    freeze = subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (ROOT / "artifacts/environment/pip_freeze_py312_gpu_qualification.txt").write_text(
        freeze, encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
