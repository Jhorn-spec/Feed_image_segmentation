# RGB Feed Image Segmentation

Proof-of-concept semantic segmentation service for estimating the visible-area
composition of livestock feed from RGB images.

The current model predicts four pixel classes:

- `background`
- `leaf`
- `stem`
- `foreign_material`

The API returns the colourized segmentation, an overlay, class pixel counts,
area percentages, approximate connected-region counts, and a composition chart.

> **Model status:** experimental. The present checkpoint has weak background
> detection and limited domain generalization. Its results demonstrate technical
> feasibility and must not yet be treated as validated analytical measurements.

## Repository layout

```text
.
|-- api.py                         FastAPI endpoints
|-- feed_inference.py              Inference, composition, and visualization
|-- rgb_baseline.py                U-Net training and tiled prediction helpers
|-- feed_seg.ipynb                 Training and evaluation notebook
|-- Dockerfile                     Linux AMD64 deployment image
|-- requirements-api.txt           Local API dependencies
|-- requirements-production.txt    Pinned container dependencies
|-- docs/
|   |-- API_GUIDE.md               Local usage and API walkthrough
|   `-- BACKEND_HANDOFF.md         Deployment/integration contract
|-- examples/
|   `-- api_client_example.py      Example client that saves returned images
`-- scripts/
    `-- visualize_rgb_baseline.py  Evaluation and review-artifact generator
```

The following directories are intentionally excluded from Git:

- `data/` — annotated and source images
- `artifacts/` — checkpoints, metrics, and evaluation outputs
- `outputs/` — generated client outputs
- `private/` — scratch notebooks and local-only material

## Run locally

```powershell
& "C:\Users\ajayi\venvs\colab-env\Scripts\Activate.ps1"
python -m pip install -r requirements-api.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## Build the container

The checkpoint is intentionally excluded from Git. Before building, place it at:

```text
artifacts/rgb_baseline/best_rgb_unet.pt
```

Then build a Linux AMD64 image:

```powershell
docker buildx build --platform linux/amd64 -t feed-segmentation-api:v0.1.0 --load .
```

## Documentation

- [API and local deployment guide](docs/API_GUIDE.md)
- [Backend handoff checklist](docs/BACKEND_HANDOFF.md)

## Security

The proof-of-concept API does not yet implement authentication. Keep it behind a
backend service, private network, IP allowlist, or SSH tunnel until authentication,
rate limiting, HTTPS, and production request validation are added.
