## ------------------------------- Builder Stage ------------------------------ ## 
FROM python:3.12-bookworm AS builder

WORKDIR /app

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and uv
RUN pip install --no-cache-dir uv \
    && pip install --upgrade pip

# Copy only the dependency files
COPY ./pyproject.toml ./

# Install dependencies manually
RUN uv pip install --no-cache-dir --system \
    pytest==8.3.4 \
    pytest-cov==6.0.0 \
    pydantic==2.10.6 \
    pydantic-settings==2.8.1 \
    gymnasium[mujoco,other]==1.1.1 \
    plotly==6.0.0 \
    sqlmodel==0.0.24 \
    comet-ml==3.49.3

# Install CPU-only PyTorch
RUN uv pip install --no-cache-dir --system \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url "https://download.pytorch.org/whl/cpu"

## ------------------------------- Test Stage ------------------------------ ##
FROM python:3.12-slim-bookworm AS production

RUN useradd --create-home appuser
USER appuser

WORKDIR /app

# Set working directory permissions to ensure GitHub Actions can write to it
RUN chmod -R 777 /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Default entrypoint for running container
ENTRYPOINT ["python", "-m"]
