"""Startup wrapper that configures IPEX before importing anything else."""
import os
os.environ["IPEX_OPTIMIZE_TRANSFORMERS"] = "0"

import torch
# Force FP32 math mode on XPU to prevent bfloat16 auto-promotion
if hasattr(torch, "xpu") and torch.xpu.is_available():
    torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.FP32)
torch.set_default_dtype(torch.float32)

import uvicorn
uvicorn.run("server:app", host="0.0.0.0", port=8079)
