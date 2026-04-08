FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (smaller image)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install server dependencies (pybullet compiles from source here)
COPY server/requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY robot_arm/ robot_arm/
COPY server/ server/

# Copy trained model checkpoints
# Note: checkpoints/ is gitignored, so you must build this image locally
# or upload checkpoints separately. See README for details.
COPY checkpoints/ checkpoints/

ENV PYTHONPATH=/app
ENV KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
