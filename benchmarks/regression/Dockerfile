# Dockerfile.benchmarks - Run BayesiPy regression benchmarks on GPU with Python 3.11

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata & code
COPY pyproject.toml README.md ./
COPY . .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install GPU-enabled PyTorch for CUDA 12.1 (adjust version if you like)
# See https://pytorch.org/get-started/locally for latest command.
RUN pip install "torch>=2.3.0" --index-url https://download.pytorch.org/whl/cu121

# Install BayesiPy from pyproject.toml (this pulls your runtime deps)
RUN pip install .[benchmarks]

# Use SQLite-backed MLflow inside container
ENV MLFLOW_TRACKING_URI=sqlite:////workspace/mlflow.db

# Default command: run the full benchmark sweep script
CMD ["bash", "benchmarks/regression/run_all_experiments.sh"]
