FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-production.txt ./
RUN python -m pip install --no-cache-dir -r requirements-production.txt

COPY api.py feed_inference.py rgb_baseline.py ./
COPY artifacts/rgb_baseline/best_rgb_unet.pt ./artifacts/rgb_baseline/best_rgb_unet.pt

EXPOSE 8000

# One worker prevents multiple copies of the ~280 MB checkpoint in memory.
CMD ["sh", "-c", "python -m uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
