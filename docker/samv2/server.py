"""SAMv2 video object tracking API for replay-guided label generation.

Accepts tracking requests with:
  - frame directory (path to extracted JPEGs)
  - initial prompts: list of (frame_number, bbox, object_id)

Returns per-frame masks and bounding boxes for each tracked object.

Designed to run on Intel Arc GPU via IPEX (XPU device).
Supports concurrent tracking via a predictor pool.
"""

import asyncio
import logging
import os
import queue
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

# Number of concurrent predictor instances (1 is optimal on XPU —
# concurrent sessions cause memory bandwidth contention and 3x slowdown)
POOL_SIZE = int(os.environ.get("SAMV2_POOL_SIZE", "1"))

app = FastAPI(title="SAMv2 Tracker", version="0.2.0")

# Predictor pool — thread-safe queue of predictor instances
predictor_pool: queue.Queue = queue.Queue()
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
    """Load SAMv2 predictor pool on startup."""
    global device

    # Detect best available device
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

        for i in range(POOL_SIZE):
            pred = build_sam2_video_predictor(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                ckpt_path="/app/checkpoints/sam2.1_hiera_large.pt",
                device=device,
            )
            pred = pred.float()
            predictor_pool.put(pred)
            logger.info("Loaded predictor %d/%d (float32)", i + 1, POOL_SIZE)

        logger.info(
            "SAMv2 predictor pool ready: %d instances on %s",
            POOL_SIZE, device,
        )
    except Exception as e:
        logger.error("Failed to load SAMv2: %s", e)
        raise


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "pool_size": POOL_SIZE,
        "pool_available": predictor_pool.qsize(),
        "device": str(device),
    }


def _do_tracking(predictor, req: TrackRequest) -> TrackResponse:
    """Run tracking inference synchronously with a specific predictor instance."""
    frame_dir = Path(req.frame_dir)

    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise HTTPException(404, f"No frames found in {req.frame_dir}")

    sample = Image.open(frames[0])
    img_w, img_h = sample.size

    t0 = time.monotonic()

    if hasattr(torch, "xpu"):
        torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.FP32)

    with torch.inference_mode():
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
        )
        for key in state.get("cached_features", {}):
            feat = state["cached_features"][key]
            if isinstance(feat, torch.Tensor) and feat.dtype == torch.bfloat16:
                state["cached_features"][key] = feat.float()

        # Register each prompt at its spawn frame
        for prompt in req.prompts:
            frame_idx = prompt.spawn_frame - 1  # 0-indexed
            if frame_idx < 0 or frame_idx >= len(frames):
                logger.warning(
                    "Prompt %d spawn frame %d out of range, skipping",
                    prompt.object_id, prompt.spawn_frame,
                )
                continue

            x1 = prompt.bbox[0] * img_w
            y1 = prompt.bbox[1] * img_h
            x2 = prompt.bbox[2] * img_w
            y2 = prompt.bbox[3] * img_h

            bbox_area_norm = (prompt.bbox[2] - prompt.bbox[0]) * (prompt.bbox[3] - prompt.bbox[1])
            if bbox_area_norm > 0.008:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=prompt.object_id,
                    box=np.array([x1, y1, x2, y2]),
                    points=np.array([[cx, cy]]),
                    labels=np.array([1]),
                )
            else:
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

        # Propagate tracking
        results = []
        prompt_lookup = {p.object_id: p for p in req.prompts}
        spawn_frame_idx = {p.object_id: p.spawn_frame - 1 for p in req.prompts}
        initial_bbox_area = {}
        for p in req.prompts:
            pw = p.bbox[2] - p.bbox[0]
            ph = p.bbox[3] - p.bbox[1]
            initial_bbox_area[p.object_id] = pw * ph

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            frame_num = frame_idx + 1

            for obj_id, mask in zip(obj_ids, masks):
                if frame_idx < spawn_frame_idx.get(obj_id, 0):
                    continue

                mask_np = mask.cpu().numpy().squeeze()

                ys, xs = np.where(mask_np > 0.5)
                if len(xs) == 0:
                    continue

                mask_probs = 1.0 / (1.0 + np.exp(-mask_np))
                confidence = float(mask_probs[mask_probs > 0.5].mean()) if (mask_probs > 0.5).any() else 0.0
                if confidence < req.confidence_threshold:
                    continue

                x1 = float(xs.min()) / img_w
                y1 = float(ys.min()) / img_h
                x2 = float(xs.max()) / img_w
                y2 = float(ys.max()) / img_h

                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area > 0.15:
                    continue

                prompt_area = initial_bbox_area.get(obj_id, bbox_area)
                if bbox_area > prompt_area * 2.5:
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


@app.post("/track", response_model=TrackResponse)
async def track_objects(req: TrackRequest):
    """Track objects through video frames using SAMv2.

    Acquires a predictor from the pool and runs inference in a thread
    so multiple requests can be processed concurrently.
    """
    if predictor_pool.qsize() == 0 and predictor_pool.empty():
        raise HTTPException(503, "Model not loaded")

    frame_dir = Path(req.frame_dir)
    if not frame_dir.exists():
        raise HTTPException(404, f"Frame directory not found: {req.frame_dir}")

    # Acquire predictor from pool (blocks if all in use)
    try:
        predictor = predictor_pool.get(timeout=300)
    except queue.Empty:
        raise HTTPException(503, "All predictors busy, try again later")

    try:
        # Run inference in thread pool to allow concurrent requests
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _do_tracking, predictor, req)
        return response
    finally:
        # Always return predictor to pool
        predictor_pool.put(predictor)
