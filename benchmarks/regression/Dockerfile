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

# Upgrade pip
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN python -m pip install --upgrade pip

# Install BayesiPy from pyproject.toml (this pulls your runtime deps)
RUN pip install .[benchmarks] --find-links https://download.pytorch.org/whl/cu121

COPY benchmarks benchmarks
COPY bayesipy bayesipy

ENV PYTHONPATH="/workspace:$PYTHONPATH"

# Use SQLite-backed MLflow inside container
RUN mkdir -p /workspace/mlflow
ENV MLFLOW_TRACKING_URI=sqlite:////workspace/mlflow/mlflow.db

# Make the script executable and use its shebang directly
RUN chmod +x benchmarks/regression/run_all_experiments.sh