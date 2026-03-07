"""SAMv2 video object tracking API for replay-guided label generation.

Accepts tracking requests with:
  - frame directory (path to extracted JPEGs)
  - initial prompts: list of (frame_number, bbox, object_id)

Returns per-frame masks and bounding boxes for each tracked object.

Designed to run on Intel Arc GPU via IPEX (XPU device).
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

# Disable IPEX auto-optimizations that cause bfloat16 promotion issues
os.environ["IPEX_OPTIMIZE_TRANSFORMERS"] = "0"

import numpy as np
import torch

# Disable IPEX's BF16 fast math path
if hasattr(torch, "xpu"):
    torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.FP32)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# Monkey-patch torch.nn.Linear to auto-cast inputs to match weight dtype
# IPEX on the Intel base image introduces bfloat16 tensors in SAMv2's memory
# attention pipeline, causing dtype mismatches with float32 weights
_original_linear_forward = torch.nn.Linear.forward

def _safe_linear_forward(self, input):
    if input.dtype != self.weight.dtype:
        input = input.to(self.weight.dtype)
    return _original_linear_forward(self, input)

torch.nn.Linear.forward = _safe_linear_forward

logger = logging.getLogger("samv2-tracker")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SAMv2 Tracker", version="0.1.0")

# Global predictor — initialized once on startup
predictor = None
device = None


class TrackPrompt(BaseModel):
    """A single object to track."""
    object_id: int
    spawn_frame: int  # frame number where the unit appears
    bbox: list[float]  # [x1, y1, x2, y2] in pixel coordinates
    card_name: str
    team: str  # "friendly" or "opponent"


class TrackRequest(BaseModel):
    """Request to track units through a video."""
    frame_dir: str  # path to frame_NNNN.jpg files
    prompts: list[TrackPrompt]
    start_frame: int = 1
    end_frame: Optional[int] = None
    confidence_threshold: float = 0.3


class TrackResult(BaseModel):
    """Tracking result for a single object at a single frame."""
    object_id: int
    frame_number: int
    bbox: list[float]  # [x1, y1, x2, y2] normalized
    confidence: float
    card_name: str
    team: str


class TrackResponse(BaseModel):
    """Response with all tracking results."""
    results: list[TrackResult]
    frames_processed: int
    objects_tracked: int
    elapsed_seconds: float


@app.on_event("startup")
async def load_model():
    """Load SAMv2 model on startup."""
    global predictor, device

    # Detect best available device
    # XPU (Intel Arc) works with the Linear dtype monkey-patch above
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        logger.info("Using Intel XPU (Arc GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    try:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path="/app/checkpoints/sam2.1_hiera_large.pt",
            device=device,
        )
        # Ensure float32 — XPU can auto-promote to bfloat16 causing dtype mismatches
        predictor = predictor.float()
        logger.info("SAMv2 model loaded successfully (float32)")
    except Exception as e:
        logger.error("Failed to load SAMv2: %s", e)
        raise


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "device": str(device),
    }


@app.post("/track", response_model=TrackResponse)
async def track_objects(req: TrackRequest):
    """Track objects through video frames using SAMv2."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    frame_dir = Path(req.frame_dir)
    if not frame_dir.exists():
        raise HTTPException(404, f"Frame directory not found: {req.frame_dir}")

    # Support both naming conventions: frame_NNNN.jpg and NNNNN.jpg
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise HTTPException(404, f"No frames found in {req.frame_dir}")

    # Get frame dimensions from first frame
    sample = Image.open(frames[0])
    img_w, img_h = sample.size

    end_frame = req.end_frame or len(frames)
    t0 = time.monotonic()

    # Initialize SAMv2 video state
    # Force FP32 — IPEX promotes to bfloat16 somewhere in the pipeline
    if hasattr(torch, "xpu"):
        torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.FP32)
    with torch.inference_mode():
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
        )
        # Ensure all cached features are float32
        for key in state.get("cached_features", {}):
            feat = state["cached_features"][key]
            if isinstance(feat, torch.Tensor) and feat.dtype == torch.bfloat16:
                state["cached_features"][key] = feat.float()
                logger.info("Converted cached feature %s from bf16 to fp32", key)

        # Register each prompt at its spawn frame
        for prompt in req.prompts:
            frame_idx = prompt.spawn_frame - 1  # 0-indexed
            if frame_idx < 0 or frame_idx >= len(frames):
                logger.warning(
                    "Prompt %d spawn frame %d out of range, skipping",
                    prompt.object_id, prompt.spawn_frame,
                )
                continue

            # Convert normalized bbox to pixel coords
            x1 = prompt.bbox[0] * img_w
            y1 = prompt.bbox[1] * img_h
            x2 = prompt.bbox[2] * img_w
            y2 = prompt.bbox[3] * img_h

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=prompt.object_id,
                box=np.array([x1, y1, x2, y2]),
            )
            logger.info(
                "Registered object %d (%s:%s) at frame %d",
                prompt.object_id, prompt.card_name, prompt.team,
                prompt.spawn_frame,
            )

        # Propagate tracking through all frames (forward only from earliest spawn)
        results = []
        prompt_lookup = {p.object_id: p for p in req.prompts}
        # Track spawn frames to filter backward propagation
        spawn_frame_idx = {p.object_id: p.spawn_frame - 1 for p in req.prompts}
        # Track initial bbox size for sanity checking (detect mask explosion)
        initial_bbox_area = {}

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            frame_num = frame_idx + 1  # back to 1-indexed

            for obj_id, mask in zip(obj_ids, masks):
                # Skip frames before this object's spawn (backward propagation)
                if frame_idx < spawn_frame_idx.get(obj_id, 0):
                    continue

                mask_np = mask.cpu().numpy().squeeze()

                # Compute bbox from mask
                ys, xs = np.where(mask_np > 0.5)
                if len(xs) == 0:
                    continue

                # Confidence from mask logits (apply sigmoid for probability)
                mask_probs = 1.0 / (1.0 + np.exp(-mask_np))
                confidence = float(mask_probs[mask_probs > 0.5].mean()) if (mask_probs > 0.5).any() else 0.0
                if confidence < req.confidence_threshold:
                    continue

                # Normalized bbox
                x1 = float(xs.min()) / img_w
                y1 = float(ys.min()) / img_h
                x2 = float(xs.max()) / img_w
                y2 = float(ys.max()) / img_h

                # Bbox sanity check: reject if bbox covers > 15% of screen
                # (a single unit should never be that large)
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area > 0.15:
                    logger.warning(
                        "Object %d bbox too large (%.1f%% of screen) at frame %d, skipping",
                        obj_id, bbox_area * 100, frame_num,
                    )
                    continue

                # Track initial bbox area and reject if it grows > 5x
                if obj_id not in initial_bbox_area:
                    initial_bbox_area[obj_id] = bbox_area
                elif bbox_area > initial_bbox_area[obj_id] * 5:
                    logger.warning(
                        "Object %d bbox exploded (%.4f -> %.4f) at frame %d, skipping",
                        obj_id, initial_bbox_area[obj_id], bbox_area, frame_num,
                    )
                    continue

                prompt = prompt_lookup.get(obj_id)
                results.append(TrackResult(
                    object_id=obj_id,
                    frame_number=frame_num,
                    bbox=[round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                    confidence=round(confidence, 3),
                    card_name=prompt.card_name if prompt else "unknown",
                    team=prompt.team if prompt else "unknown",
                ))

    elapsed = time.monotonic() - t0
    logger.info(
        "Tracked %d objects across %d frames in %.1fs",
        len(req.prompts), len(frames), elapsed,
    )

    return TrackResponse(
        results=results,
        frames_processed=len(frames),
        objects_tracked=len(req.prompts),
        elapsed_seconds=round(elapsed, 2),
    )
