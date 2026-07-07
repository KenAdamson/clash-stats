"""Startup wrapper that forces FP32 math before importing anything else.

IPEX_OPTIMIZE_TRANSFORMERS / set_fp32_math_mode are IPEX-era workarounds for
BF16 auto-promotion; on native torch XPU builds they're absent and unneeded,
so both are applied only if present.
"""
import os
os.environ["IPEX_OPTIMIZE_TRANSFORMERS"] = "0"
os.environ.setdefault("TQDM_DISABLE", "1")  # sam2's per-frame bars spam logs + burn CPU

import torch
xpu = getattr(torch, "xpu", None)
if xpu is not None and xpu.is_available() and hasattr(xpu, "set_fp32_math_mode"):
    xpu.set_fp32_math_mode(xpu.FP32MathMode.FP32)
torch.set_default_dtype(torch.float32)

import uvicorn
uvicorn.run("server:app", host="0.0.0.0", port=8079)
