# RGB feed segmentation demo API

This service accepts one RGB image, predicts a four-class semantic mask, and
returns the pixel-area composition for `background`, `leaf`, `stem`, and
`foreign_material`.

## 1. Install the API packages

Activate the same environment used to train the model, then run:

```powershell
python -m pip install -r requirements-api.txt
```

The training dependencies (`torch`, `opencv-python`,
`segmentation-models-pytorch`, and `numpy`) must remain installed as well.

## 2. Start the local server

From the `Feed Image segmentation` directory:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger interface.
The first `/health` or `/predict` call loads the approximately 280 MB model and
therefore takes longer than subsequent calls.

## 3. Call the API

PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "sample_id=canola-001" `
  -F "file=@..\test_data\Canola (1).JPG"
```

Python:

```python
from pathlib import Path
import requests

image_path = Path(r"..\test_data\Canola (1).JPG")
with image_path.open("rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        data={"sample_id": "canola-001"},
        files={"file": (image_path.name, image_file, "image/jpeg")},
        timeout=180,
    )

response.raise_for_status()
result = response.json()
for class_name, values in result["composition"]["classes"].items():
    print(class_name, values["sample_percentage"])
```

Or use the included client, which saves the JSON, segmented mask, overlay, and
composition chart into one directory:

```powershell
python examples\api_client_example.py "..\test_data\Canola (1).JPG" `
  --sample-id canola-001 `
  --output-dir "artifacts\rgb_baseline\api_demo"
```

The response includes browser-ready PNG data URIs for the segmented mask,
overlay, and composition chart by default. Use `?include_visuals=false` when
only the numeric JSON is required.

Save the returned visuals from Python:

```python
import base64
from pathlib import Path

def save_data_uri(data_uri: str, output_path: str) -> None:
    _, encoded = data_uri.split(",", 1)
    Path(output_path).write_bytes(base64.b64decode(encoded))

save_data_uri(result["visuals"]["segmented_mask"], "segmented_mask.png")
save_data_uri(result["visuals"]["overlay"], "overlay.png")
save_data_uri(result["visuals"]["composition_chart"], "composition_chart.png")
```

## 4. Meaning of the percentages

- `image_percentage`: class pixels divided by all image pixels. These four
  values sum to 100%.
- `sample_percentage`: class pixels divided by all non-background pixels. Use
  this for plant-part composition. Leaf, stem, and foreign material sum to
  100%; background is reported as 0% for this denominator.
- These are projected two-dimensional pixel-area percentages, not physical
  mass or volume percentages.

## 5. Meaning of the part counts

`estimated_parts.classes.leaf.estimated_part_count` and the corresponding stem
and foreign-material values are calculated using connected components. Regions
smaller than 250 pixels are ignored by default. Change this with, for example,
`?min_component_pixels=100`.

This does **not** provide a guaranteed count of physical pieces. Semantic
segmentation assigns a class to every pixel: touching pieces can form one region
and one broken prediction can form several regions. True physical-object counts
eventually require instance annotations and an instance-segmentation model.

## 6. Call the predictor without an API

```python
from pathlib import Path
import cv2

from feed_inference import FeedSegmenter

root = Path.cwd()
segmenter = FeedSegmenter(root / "artifacts/rgb_baseline/best_rgb_unet.pt")

image_bgr = cv2.imread(str(root.parent / "test_data/Canola (1).JPG"))
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = segmenter.predict(image_rgb)
print(result["composition"])
```

## 7. Demonstration limitations

The current checkpoint is a proof of concept. It has zero measured background
IoU on the annotated holdout set and treats the unannotated canola images almost
entirely as `foreign_material`. Keep the warning in the response visible during
the demonstration. Retrain with representative background and domain samples
before treating the percentages as scientifically or operationally reliable.

For deployment, run one API worker per model/GPU. Multiple workers each load a
separate copy of the checkpoint into memory.

## 8. Run or deploy the container

Build and run locally from the project directory:

```powershell
docker build -t feed-segmentation-demo .
docker run --rm -p 8000:8000 feed-segmentation-demo
```

The resulting container can be deployed to a Docker-compatible platform. It
expects the platform's `PORT` environment variable and starts one worker. Choose
a service with enough memory for PyTorch, the ResNet-34 U-Net, and tiled logits;
start with at least 2 GB RAM for a CPU demonstration and measure actual peak
usage before selecting a final production size.

Do not make this demonstration endpoint public without adding authentication,
HTTPS, request logging, rate limiting, and stricter content validation.
