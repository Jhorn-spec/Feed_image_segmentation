"""Call the running demo API and save its returned visual artifacts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests


def save_data_uri(data_uri: str, output_path: Path) -> None:
    _, encoded = data_uri.split(",", 1)
    output_path.write_bytes(base64.b64decode(encoded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--output-dir", type=Path, default=Path("api_output"))
    parser.add_argument("--min-component-pixels", type=int, default=250)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.image.open("rb") as image_file:
        response = requests.post(
            args.url,
            params={
                "include_visuals": "true",
                "min_component_pixels": args.min_component_pixels,
            },
            data={"sample_id": args.sample_id or args.image.stem},
            files={"file": (args.image.name, image_file, "image/jpeg")},
            timeout=300,
        )
    response.raise_for_status()
    result = response.json()

    visuals = result.pop("visuals")
    save_data_uri(visuals["segmented_mask"], args.output_dir / "segmented_mask.png")
    save_data_uri(visuals["overlay"], args.output_dir / "overlay.png")
    save_data_uri(visuals["composition_chart"], args.output_dir / "composition_chart.png")
    (args.output_dir / "prediction.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"Saved API response under: {args.output_dir.resolve()}")
    for class_name, values in result["estimated_parts"]["classes"].items():
        percentage = result["composition"]["classes"][class_name]["sample_percentage"]
        print(
            f"{class_name}: {values['estimated_part_count']} estimated parts, "
            f"{percentage:.2f}% of detected sample"
        )


if __name__ == "__main__":
    main()
