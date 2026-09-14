# Backend handoff

## Container

Provide the backend engineer with an immutable image reference such as:

```text
registry.digitalocean.com/<registry>/feed-segmentation-api:v0.1.0
```

Deployment assumptions:

- Linux AMD64 container
- one Uvicorn worker
- container port `8000`
- minimum starting allocation: 1 CPU and 2 GiB RAM
- health check: `GET /health`
- allow a 60-second startup delay for model initialization

## API contract

### `GET /health`

Loads the checkpoint when first called and returns model readiness and metadata.

### `POST /predict`

Request type: `multipart/form-data`.

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | JPG or PNG RGB image, maximum 25 MB |
| `sample_id` | no | Caller-defined sample identifier |
| `include_visuals` | no | Return PNG data URIs; defaults to `true` |
| `min_component_pixels` | no | Small-region filter for approximate part counts; defaults to 250 |

Use a client timeout of at least 120 seconds for CPU inference. The response
contains composition values, approximate connected-region counts, warnings, and
optionally base64-encoded PNGs. The formal schema is served at `/openapi.json`
and interactive documentation at `/docs`.

## Integration architecture

```text
browser/mobile client -> authenticated backend -> segmentation service
```

Do not expose the current model endpoint directly to untrusted clients. The
backend should validate the caller, apply rate limits, enforce its own upload
limit and timeout, and decide whether large base64 visuals should be forwarded
or stored in object storage.

## Known limitations

- The checkpoint is an RGB proof of concept, not a validated production model.
- Background IoU was zero on the annotated holdout evaluation.
- Unseen canola images were classified almost entirely as `foreign_material`.
- Percentages describe projected pixel area, not mass or volume.
- Part counts are connected-region estimates, not true instance counts.

Keep these limitations visible in demonstrations and downstream interfaces.

## Release checklist

- Record the image tag and SHA digest.
- Confirm `GET /health` succeeds after a clean container start.
- Run a known sample through `POST /predict`.
- Preserve the API response and expected visual outputs for regression checking.
- Share registry access through team permissions; never share personal tokens.
- Document model changes and increment the image version for every release.
