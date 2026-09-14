"""FastAPI demonstration service for the trained RGB feed-segmentation model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from feed_inference import FeedSegmenter, decode_rgb_image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "rgb_baseline" / "best_rgb_unet.pt"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title="RGB Feed Segmentation Demo API",
    version="0.1.0",
    description=(
        "Segments an RGB feed image into background, leaf, stem, and "
        "foreign_material, then reports pixel-area composition."
    ),
)


@lru_cache(maxsize=1)
def get_segmenter() -> FeedSegmenter:
    checkpoint = Path(os.getenv("FEED_MODEL_PATH", str(DEFAULT_CHECKPOINT)))
    return FeedSegmenter(
        checkpoint,
        device=os.getenv("FEED_DEVICE") or None,
        tile_size=int(os.getenv("FEED_TILE_SIZE", "512")),
        stride=int(os.getenv("FEED_TILE_STRIDE", "384")),
        tile_batch_size=int(os.getenv("FEED_TILE_BATCH_SIZE", "4")),
    )


@app.get("/health")
def health() -> dict:
    """Load the checkpoint if necessary and report service readiness."""
    try:
        model_info = get_segmenter().model_info
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model is not ready: {exc}") from exc
    return {"status": "ok", "model": model_info}


@app.get("/model-info")
def model_info() -> dict:
    return get_segmenter().model_info


@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="One JPG or PNG RGB feed image"),
    sample_id: str | None = Form(default=None),
    include_visuals: bool = Query(
        default=True,
        description="Include mask, overlay, and composition-chart PNG data URIs.",
    ),
    min_component_pixels: int = Query(
        default=250,
        ge=1,
        le=100000,
        description="Ignore smaller connected regions when estimating part counts.",
    ),
) -> dict:
    """Return segmentation, composition, and approximate connected-part counts."""
    image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 25 MB upload limit.")

    try:
        image = decode_rgb_image(image_bytes)
        result = get_segmenter().predict(
            image,
            include_visuals=include_visuals,
            min_component_pixels=min_component_pixels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    result["sample_id"] = sample_id or Path(file.filename or "uploaded_image").stem
    result["filename"] = file.filename
    return result
