"""Create review artifacts for a trained RGB feed-segmentation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rgb_baseline import (
    CLASS_NAMES,
    build_rgb_unet,
    evaluate_full_images,
    load_best_checkpoint,
    load_rgb_and_mask,
    metrics_from_confusion,
    predict_full_image_tiled,
    update_confusion_matrix,
)


CLASS_COLORS = np.asarray(
    [
        (0, 0, 0),        # background
        (102, 255, 102),  # leaf
        (255, 106, 77),   # stem
        (184, 61, 245),   # foreign material
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    if mask.min() < 0 or mask.max() >= len(CLASS_COLORS):
        raise ValueError(f"Mask contains invalid IDs: {np.unique(mask).tolist()}")
    return CLASS_COLORS[mask]


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = colorize(mask)
    return cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)


def save_training_plots(history_path: Path, output_dir: Path) -> Path:
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["epoch"], history["train_loss"], color="#2f5597")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")

    axes[1].plot(history["epoch"], history["val_macro_iou"], label="Macro IoU")
    axes[1].plot(history["epoch"], history["val_macro_dice"], label="Macro Dice")
    best_idx = history["val_macro_iou"].idxmax()
    best_epoch = int(history.loc[best_idx, "epoch"])
    best_iou = float(history.loc[best_idx, "val_macro_iou"])
    axes[1].scatter([best_epoch], [best_iou], color="red", zorder=3)
    axes[1].annotate(
        f"best: epoch {best_epoch}\nIoU={best_iou:.3f}",
        (best_epoch, best_iou),
        xytext=(8, -35),
        textcoords="offset points",
    )
    axes[1].set(title="Validation summary", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()

    for class_name in CLASS_NAMES:
        column = f"val_iou_{class_name}"
        if column in history:
            axes[2].plot(history["epoch"], history[column], label=class_name)
    axes[2].set(title="Validation IoU by class", xlabel="Epoch", ylabel="IoU", ylim=(0, 1))
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metric_comparison(validation_metrics: dict, test_metrics: dict, output_dir: Path) -> Path:
    x = np.arange(len(CLASS_NAMES))
    validation_iou = [validation_metrics["per_class"][name]["iou"] for name in CLASS_NAMES]
    test_iou = [test_metrics["per_class"][name]["iou"] for name in CLASS_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.38
    axes[0].bar(x - width / 2, validation_iou, width, label="validation")
    axes[0].bar(x + width / 2, test_iou, width, label="annotated test")
    axes[0].set_xticks(x, CLASS_NAMES, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Per-class IoU")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    confusion = np.asarray(test_metrics["confusion_matrix"], dtype=np.float64)
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sum, out=np.zeros_like(confusion), where=row_sum > 0)
    shown = axes[1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            value = normalized[row, col]
            axes[1].text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )
    axes[1].set_xticks(x, CLASS_NAMES, rotation=20, ha="right")
    axes[1].set_yticks(x, CLASS_NAMES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground truth")
    axes[1].set_title("Annotated-test confusion (row-normalized)")
    fig.colorbar(shown, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    output_path = output_dir / "metric_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _thumbnail(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(array)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def save_review_sheet(records: list[dict], output_path: Path, annotated: bool) -> None:
    if not records:
        return
    panel_width = 280
    panel_height = 250
    columns = 4 if annotated else 3
    sheet = Image.new("RGB", (panel_width * columns, panel_height * len(records)), "white")
    draw = ImageDraw.Draw(sheet)

    for row_index, record in enumerate(records):
        panels = [("RGB", record["image"])]
        if annotated:
            panels.append(("Ground truth", colorize(record["target"])))
        panels.extend(
            [
                ("Prediction", colorize(record["prediction"])),
                ("Overlay", overlay(record["image"], record["prediction"])),
            ]
        )
        y = row_index * panel_height
        draw.text((8, y + 5), record["name"], fill="black")
        if "macro_iou" in record:
            draw.text((8, y + 22), f"image macro IoU: {record['macro_iou']:.3f}", fill="black")

        for column_index, (label, panel) in enumerate(panels):
            thumb = _thumbnail(panel, (panel_width - 16, 190))
            x = column_index * panel_width + (panel_width - thumb.width) // 2
            sheet.paste(thumb, (x, y + 42))
            draw.text((column_index * panel_width + 8, y + 232), label, fill="black")

    sheet.save(output_path, quality=92)


def evaluate_and_visualize_annotated(
    model,
    df: pd.DataFrame,
    device: torch.device,
    project_root: Path,
    output_dir: Path,
    tile_size: int,
    stride: int,
    tile_batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    mask_dir = output_dir / "annotated_test" / "masks"
    overlay_dir = output_dir / "annotated_test" / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    records = []
    per_image = []

    for number, row in enumerate(df.itertuples(index=False), start=1):
        image, target = load_rgb_and_mask(row, project_root)
        prediction = predict_full_image_tiled(
            model,
            image,
            device,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
        )
        update_confusion_matrix(confusion, target, prediction, len(CLASS_NAMES))
        image_confusion = np.zeros_like(confusion)
        update_confusion_matrix(image_confusion, target, prediction, len(CLASS_NAMES))
        image_metrics = metrics_from_confusion(image_confusion)
        per_image.append(
            {
                "sample_id": row.sample_id,
                "group_id": row.group_id,
                "macro_iou": image_metrics["macro_iou"],
                "macro_dice": image_metrics["macro_dice"],
                "pixel_accuracy": image_metrics["pixel_accuracy"],
            }
        )
        Image.fromarray(prediction).save(mask_dir / f"{row.sample_id}.png")
        Image.fromarray(overlay(image, prediction)).save(overlay_dir / f"{row.sample_id}.jpg", quality=92)
        records.append(
            {
                "name": row.sample_id,
                "image": image,
                "target": target,
                "prediction": prediction,
                "macro_iou": image_metrics["macro_iou"],
            }
        )
        print(f"Annotated test {number}/{len(df)}: {row.sample_id}", flush=True)

    per_image_df = pd.DataFrame(per_image).sort_values("macro_iou")
    per_image_df.to_csv(output_dir / "annotated_test" / "per_image_metrics.csv", index=False)
    metrics = metrics_from_confusion(confusion)
    (output_dir / "annotated_test" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True)
    )
    record_lookup = {record["name"]: record for record in records}
    ordered_records = [record_lookup[name] for name in per_image_df["sample_id"]]
    save_review_sheet(ordered_records, output_dir / "annotated_test_review.jpg", annotated=True)
    return metrics, per_image_df


def infer_unseen(
    model,
    image_paths: list[Path],
    device: torch.device,
    output_dir: Path,
    tile_size: int,
    stride: int,
    tile_batch_size: int,
) -> pd.DataFrame:
    mask_dir = output_dir / "unseen" / "masks"
    overlay_dir = output_dir / "unseen" / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    records = []

    for number, image_path in enumerate(image_paths, start=1):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skipping unreadable image: {image_path}", flush=True)
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prediction = predict_full_image_tiled(
            model,
            image,
            device,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
        )
        counts = np.bincount(prediction.ravel(), minlength=len(CLASS_NAMES))
        fractions = counts / counts.sum()
        row = {"image": str(image_path)}
        row.update({f"fraction_{name}": float(fractions[i]) for i, name in enumerate(CLASS_NAMES)})
        rows.append(row)
        safe_stem = f"{number:03d}_{image_path.stem}"
        Image.fromarray(prediction).save(mask_dir / f"{safe_stem}.png")
        Image.fromarray(overlay(image, prediction)).save(overlay_dir / f"{safe_stem}.jpg", quality=92)
        records.append({"name": image_path.name, "image": image, "prediction": prediction})
        print(f"Unseen {number}/{len(image_paths)}: {image_path.name}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "unseen" / "predicted_pixel_fractions.csv", index=False)
    save_review_sheet(records, output_dir / "unseen_review.jpg", annotated=False)
    return result


def discover_images(directory: Path) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--unseen-dir", type=Path, default=Path("../test_data"))
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts/rgb_baseline")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/rgb_baseline/review")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    manifest = pd.read_csv(args.project_root / "data/clean_dataset/manifest.csv")
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    validation_metrics = json.loads(
        (args.artifact_dir / "best_validation_metrics.json").read_text(),
        parse_constant=lambda _: float("nan"),
    )

    model = build_rgb_unet()
    checkpoint = load_best_checkpoint(model, args.artifact_dir / "best_rgb_unet.pt", device)
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}", flush=True)

    save_training_plots(args.artifact_dir / "training_history.csv", args.output_dir)
    test_metrics, _ = evaluate_and_visualize_annotated(
        model,
        test_df,
        device,
        args.project_root,
        args.output_dir,
        args.tile_size,
        args.stride,
        args.tile_batch_size,
    )
    save_metric_comparison(validation_metrics, test_metrics, args.output_dir)

    unseen_paths = discover_images(args.unseen_dir)
    if unseen_paths:
        infer_unseen(
            model,
            unseen_paths,
            device,
            args.output_dir,
            args.tile_size,
            args.stride,
            args.tile_batch_size,
        )
    else:
        print(f"No unseen images found under {args.unseen_dir}", flush=True)

    print(f"Review artifacts saved under: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
