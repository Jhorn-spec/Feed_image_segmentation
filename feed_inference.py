"""Reusable RGB feed-segmentation inference and composition calculation."""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from rgb_baseline import (
    CLASS_NAMES,
    build_rgb_unet,
    load_best_checkpoint,
    predict_full_image_tiled,
)


CLASS_COLORS = np.asarray(
    [
        (0, 0, 0),        # background
        (102, 255, 102),  # leaf
        (255, 106, 77),   # stem
        (184, 61, 245),   # foreign_material
    ],
    dtype=np.uint8,
)


def decode_rgb_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an H x W x 3 uint8 RGB image."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The uploaded file is not a readable RGB image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def calculate_composition(mask: np.ndarray) -> dict[str, Any]:
    """Calculate whole-image and sample-only class percentages.

    ``image_percentage`` uses every pixel as the denominator. For plant-part
    composition, ``sample_percentage`` excludes class 0 (background), so the
    leaf/stem/foreign_material percentages sum to 100 when sample pixels exist.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D class mask, received shape {mask.shape}.")
    if mask.size == 0:
        raise ValueError("Cannot calculate composition for an empty mask.")
    if mask.min() < 0 or mask.max() >= len(CLASS_NAMES):
        raise ValueError(f"Mask contains invalid class IDs: {np.unique(mask).tolist()}")

    counts = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES)).astype(np.int64)
    total_pixels = int(mask.size)
    sample_pixels = int(total_pixels - counts[0])

    classes: dict[str, dict[str, Any]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = int(counts[class_id])
        sample_percentage = None
        if class_id == 0:
            sample_percentage = 0.0 if sample_pixels else None
        elif sample_pixels:
            sample_percentage = round(100.0 * count / sample_pixels, 4)

        classes[class_name] = {
            "class_id": class_id,
            "pixels": count,
            "image_percentage": round(100.0 * count / total_pixels, 4),
            "sample_percentage": sample_percentage,
        }

    return {
        "total_pixels": total_pixels,
        "sample_pixels": sample_pixels,
        "background_pixels": int(counts[0]),
        "classes": classes,
    }


def estimate_part_counts(mask: np.ndarray, min_component_pixels: int = 250) -> dict[str, Any]:
    """Estimate distinct foreground parts using 8-connected mask regions.

    This is an approximation on a semantic mask: touching parts become one
    region and a fragmented part can become several regions. Very small regions
    are discarded as likely prediction noise.
    """
    if min_component_pixels < 1:
        raise ValueError("min_component_pixels must be at least 1.")

    class_results: dict[str, dict[str, Any]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
        binary = (mask == class_id).astype(np.uint8)
        number, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64) if number > 1 else np.array([], dtype=np.int64)
        retained = areas[areas >= min_component_pixels]

        class_results[class_name] = {
            "estimated_part_count": int(retained.size),
            "raw_connected_regions": int(areas.size),
            "discarded_small_regions": int(areas.size - retained.size),
            "component_area_pixels": {
                "minimum": int(retained.min()) if retained.size else None,
                "median": round(float(np.median(retained)), 1) if retained.size else None,
                "maximum": int(retained.max()) if retained.size else None,
            },
        }

    return {
        "method": "8-connected components on the predicted semantic mask",
        "min_component_pixels": min_component_pixels,
        "classes": class_results,
    }


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a class-ID mask to an RGB visualization."""
    return CLASS_COLORS[mask]


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a class-colour mask over an RGB image."""
    return cv2.addWeighted(image, 1.0 - alpha, colorize_mask(mask), alpha, 0)


def encode_png_base64(rgb_image: np.ndarray) -> str:
    """Encode an RGB array as a base64 PNG string for an optional API response."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Could not encode prediction image as PNG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def png_data_uri(rgb_image: np.ndarray) -> str:
    """Return a PNG as a browser-ready data URI."""
    return "data:image/png;base64," + encode_png_base64(rgb_image)


def make_composition_chart(composition: dict[str, Any]) -> np.ndarray:
    """Create a simple RGB bar chart of sample-only foreground percentages."""
    width, height = 900, 520
    chart = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 105, 40, 65, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    foreground_names = list(CLASS_NAMES[1:])

    cv2.putText(
        chart,
        "Predicted sample composition",
        (190, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.line(chart, (left, top), (left, top + plot_height), (70, 70, 70), 2)
    cv2.line(chart, (left, top + plot_height), (left + plot_width, top + plot_height), (70, 70, 70), 2)

    for tick in range(0, 101, 20):
        y = top + plot_height - round(plot_height * tick / 100)
        cv2.line(chart, (left - 5, y), (left + plot_width, y), (225, 225, 225), 1)
        cv2.putText(chart, str(tick), (48, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)

    slot = plot_width / len(foreground_names)
    bar_width = int(slot * 0.52)
    for index, class_name in enumerate(foreground_names, start=1):
        value = composition["classes"][class_name]["sample_percentage"] or 0.0
        x_center = int(left + (index - 0.5) * slot)
        x1, x2 = x_center - bar_width // 2, x_center + bar_width // 2
        y1 = top + plot_height - round(plot_height * value / 100)
        color = tuple(int(channel) for channel in CLASS_COLORS[index])
        cv2.rectangle(chart, (x1, y1), (x2, top + plot_height), color, -1)
        cv2.rectangle(chart, (x1, y1), (x2, top + plot_height), (60, 60, 60), 1)
        cv2.putText(chart, f"{value:.2f}%", (x1, max(top + 20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 2, cv2.LINE_AA)
        label = class_name.replace("_", " ")
        cv2.putText(chart, label, (x_center - 72, top + plot_height + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)

    cv2.putText(chart, "Percentage of non-background pixels", (270, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 70, 70), 1, cv2.LINE_AA)
    return chart


class FeedSegmenter:
    """Load one model checkpoint and perform thread-safe tiled inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        encoder_name: str = "resnet34",
        tile_size: int = 512,
        stride: int = 384,
        tile_batch_size: int = 4,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tile_size = tile_size
        self.stride = stride
        self.tile_batch_size = tile_batch_size
        self.model = build_rgb_unet(encoder_name=encoder_name, num_classes=len(CLASS_NAMES))
        self.checkpoint = load_best_checkpoint(self.model, self.checkpoint_path, self.device)
        self._inference_lock = threading.Lock()

    def _predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Pad small inputs, run tiled prediction, then crop back to original size."""
        height, width = image.shape[:2]
        pad_bottom = max(0, self.tile_size - height)
        pad_right = max(0, self.tile_size - width)
        if pad_bottom or pad_right:
            image_for_model = cv2.copyMakeBorder(
                image,
                0,
                pad_bottom,
                0,
                pad_right,
                cv2.BORDER_REFLECT_101,
            )
        else:
            image_for_model = image

        with self._inference_lock:
            prediction = predict_full_image_tiled(
                self.model,
                image_for_model,
                self.device,
                tile_size=self.tile_size,
                stride=self.stride,
                tile_batch_size=self.tile_batch_size,
                num_classes=len(CLASS_NAMES),
            )
        return prediction[:height, :width]

    def predict(
        self,
        image: np.ndarray,
        *,
        include_visuals: bool = True,
        min_component_pixels: int = 250,
    ) -> dict[str, Any]:
        """Segment one RGB image and return its class composition."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an H x W x 3 RGB image, received {image.shape}.")

        started = time.perf_counter()
        mask = self._predict_mask(image)
        elapsed_ms = round(1000.0 * (time.perf_counter() - started), 1)
        composition = calculate_composition(mask)
        estimated_parts = estimate_part_counts(mask, min_component_pixels=min_component_pixels)

        warnings: list[str] = []
        background_percentage = composition["classes"]["background"]["image_percentage"]
        if background_percentage < 0.01:
            warnings.append(
                "The model detected virtually no background. The current demonstration "
                "checkpoint is known to perform poorly on the background class."
            )
        warnings.append(
            "Part counts are connected-region estimates, not instance-segmentation counts; "
            "touching parts may merge and fragmented parts may be counted more than once."
        )

        result: dict[str, Any] = {
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "inference_ms": elapsed_ms,
            "composition": composition,
            "estimated_parts": estimated_parts,
            "warnings": warnings,
        }
        if include_visuals:
            result["visuals"] = {
                "encoding": "PNG data URI",
                "segmented_mask": png_data_uri(colorize_mask(mask)),
                "overlay": png_data_uri(make_overlay(image, mask)),
                "composition_chart": png_data_uri(make_composition_chart(composition)),
            }
        return result

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint_path.name,
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "classes": list(CLASS_NAMES),
            "input_channels": 3,
            "input_type": "RGB",
            "device": str(self.device),
            "tile_size": self.tile_size,
            "stride": self.stride,
        }
