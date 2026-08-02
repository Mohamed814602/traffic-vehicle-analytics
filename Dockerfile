FROM python:3.12-slim

# libgl1 / libglib required by opencv-python-headless at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

# Bake in the COCO baseline weights at build time so the container works
# out of the box; mount/replace with a fine-tuned UA-DETRAC checkpoint
# for production use.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
