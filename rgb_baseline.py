"""Reusable training and full-image evaluation helpers for the RGB feed baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch import nn


CLASS_NAMES = ("background", "leaf", "stem", "foreign_material")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def resolve_manifest_path(path_value: str, project_root: Path | str = Path(".")) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else Path(project_root) / path


def split_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize pixel count and image presence for each class in one split."""
    rows = []
    total_pixels = sum(int(row.width) * int(row.height) for row in df.itertuples())
    for class_name in CLASS_NAMES:
        pixels = int(df[f"{class_name}_pixels"].sum())
        rows.append(
            {
                "class": class_name,
                "pixels": pixels,
                "pixel_fraction": pixels / total_pixels if total_pixels else 0.0,
                "images_with_class": int(df[f"has_{class_name}"].sum()),
                "images": len(df),
            }
        )
    return pd.DataFrame(rows)


def compute_class_weights(df: pd.DataFrame, minimum: float = 0.25, maximum: float = 3.0) -> torch.Tensor:
    """Compute stable square-root inverse-frequency weights from training rows only."""
    counts = np.asarray(
        [float(df[f"{class_name}_pixels"].sum()) for class_name in CLASS_NAMES],
        dtype=np.float64,
    )
    if np.any(counts <= 0):
        missing = [CLASS_NAMES[i] for i, count in enumerate(counts) if count <= 0]
        raise ValueError(f"Training split has no pixels for classes: {missing}")

    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights /= weights.mean()
    weights = np.clip(weights, minimum, maximum)
    return torch.tensor(weights, dtype=torch.float32)


def build_rgb_unet(encoder_name: str = "resnet34", num_classes: int = 4) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, dice_weight: float = 1.0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, targets) + self.dice_weight * self.dice(logits, targets)


def update_confusion_matrix(
    confusion: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int = 4,
) -> None:
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    valid = (targets >= 0) & (targets < num_classes)
    encoded = num_classes * targets[valid].astype(np.int64) + predictions[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    confusion = confusion.astype(np.float64)
    true_positive = np.diag(confusion)
    ground_truth = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
    dice_denominator = ground_truth + predicted
    dice = np.divide(
        2 * true_positive,
        dice_denominator,
        out=np.full_like(true_positive, np.nan),
        where=dice_denominator > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.full_like(true_positive, np.nan),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        ground_truth,
        out=np.full_like(true_positive, np.nan),
        where=ground_truth > 0,
    )

    per_class = {
        class_name: {
            "iou": float(iou[i]),
            "dice": float(dice[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support_pixels": int(ground_truth[i]),
        }
        for i, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "macro_iou": float(np.nanmean(iou)),
        "macro_dice": float(np.nanmean(dice)),
        "pixel_accuracy": float(true_positive.sum() / confusion.sum()),
        "per_class": per_class,
        "confusion_matrix": confusion.astype(np.int64).tolist(),
    }


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _normalize_tile(tile: np.ndarray) -> torch.Tensor:
    tile = tile.astype(np.float32) / 255.0
    tile = (tile - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(tile.transpose(2, 0, 1)).float()


@torch.inference_mode()
def predict_full_image_tiled(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    stride: int = 384,
    tile_batch_size: int = 4,
    num_classes: int = 4,
) -> np.ndarray:
    """Predict one full-resolution mask by averaging logits in overlapping tiles."""
    model.eval()
    height, width = image.shape[:2]
    if height < tile_size or width < tile_size:
        raise ValueError(
            f"Image size {(height, width)} is smaller than tile size {tile_size}. "
            "Pad small images before tiled inference."
        )

    y_starts = _tile_starts(height, tile_size, stride)
    x_starts = _tile_starts(width, tile_size, stride)
    coordinates = [(y, x) for y in y_starts for x in x_starts]

    logit_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    logit_count = np.zeros((height, width), dtype=np.float32)

    for offset in range(0, len(coordinates), tile_batch_size):
        batch_coordinates = coordinates[offset : offset + tile_batch_size]
        batch = torch.stack(
            [
                _normalize_tile(image[y : y + tile_size, x : x + tile_size])
                for y, x in batch_coordinates
            ]
        ).to(device)
        batch_logits = model(batch).detach().cpu().numpy()

        for logits, (y, x) in zip(batch_logits, batch_coordinates):
            logit_sum[:, y : y + tile_size, x : x + tile_size] += logits
            logit_count[y : y + tile_size, x : x + tile_size] += 1.0

    logit_sum /= np.maximum(logit_count[None, :, :], 1.0)
    return np.argmax(logit_sum, axis=0).astype(np.uint8)


def load_rgb_and_mask(row, project_root: Path | str = Path(".")) -> tuple[np.ndarray, np.ndarray]:
    image_path = resolve_manifest_path(row.image_path, project_root)
    mask_path = resolve_manifest_path(row.class_id_mask_path, project_root)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image-mask shape mismatch for {row.sample_id}: {image.shape[:2]} vs {mask.shape}")
    return image, mask


@torch.inference_mode()
def evaluate_full_images(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    project_root: Path | str = Path("."),
    tile_size: int = 512,
    stride: int = 384,
    tile_batch_size: int = 4,
) -> tuple[dict, pd.DataFrame]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    per_image_rows = []

    for row in df.itertuples(index=False):
        image, target = load_rgb_and_mask(row, project_root)
        prediction = predict_full_image_tiled(
            model,
            image,
            device,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
            num_classes=len(CLASS_NAMES),
        )
        update_confusion_matrix(confusion, target, prediction, len(CLASS_NAMES))

        image_confusion = np.zeros_like(confusion)
        update_confusion_matrix(image_confusion, target, prediction, len(CLASS_NAMES))
        image_metrics = metrics_from_confusion(image_confusion)
        per_image_rows.append(
            {
                "sample_id": row.sample_id,
                "group_id": row.group_id,
                "macro_iou": image_metrics["macro_iou"],
                "macro_dice": image_metrics["macro_dice"],
                "pixel_accuracy": image_metrics["pixel_accuracy"],
            }
        )

    return metrics_from_confusion(confusion), pd.DataFrame(per_image_rows)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def fit_rgb_baseline(
    model: nn.Module,
    train_loader: Iterable,
    val_df: pd.DataFrame,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Path | str,
    epochs: int = 80,
    patience: int = 12,
    tile_size: int = 512,
    stride: int = 384,
    tile_batch_size: int = 4,
    project_root: Path | str = Path("."),
    config: dict | None = None,
) -> pd.DataFrame:
    """Train and retain the checkpoint with the best validation macro IoU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_rgb_unet.pt"
    last_path = output_dir / "last_rgb_unet.pt"
    history_path = output_dir / "training_history.csv"
    metrics_path = output_dir / "best_validation_metrics.json"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    best_score = -np.inf
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate_full_images(
            model,
            val_df,
            device,
            project_root=project_root,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
        )
        score = val_metrics["macro_iou"]
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_iou": score,
            "val_macro_dice": val_metrics["macro_dice"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for class_name in CLASS_NAMES:
            row[f"val_iou_{class_name}"] = val_metrics["per_class"][class_name]["iou"]
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": val_metrics,
            "class_names": list(CLASS_NAMES),
            "config": config or {},
        }
        torch.save(checkpoint, last_path)

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
            metrics_path.write_text(json.dumps(val_metrics, indent=2, allow_nan=True))
        else:
            epochs_without_improvement += 1

        class_ious = ", ".join(
            f"{name}={val_metrics['per_class'][name]['iou']:.3f}"
            if np.isfinite(val_metrics["per_class"][name]["iou"])
            else f"{name}=N/A"
            for name in CLASS_NAMES
        )
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val mIoU={score:.4f} | {class_ious}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs. Best validation mIoU={best_score:.4f}")
            break

    return pd.DataFrame(history_rows)


def load_best_checkpoint(model: nn.Module, checkpoint_path: Path | str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint
