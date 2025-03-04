FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry \
    && pip install --upgrade pip

# Copy only the dependency files
COPY pyproject.toml ./

# Configure Poetry 
RUN poetry config virtualenvs.create false

# Install test dependencies only
RUN poetry install --no-interaction --no-root --only testing

# Install core dependencies manually
RUN pip install pydantic==2.10.5 \
    pydantic-settings==2.7.1 \
    gymnasium[mujoco,other]==1.1.0 \
    numpy==2.2.0 \
    comet-ml==3.49.3

# Install CPU-only PyTorch
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url "https://download.pytorch.org/whl/cpu"

# Set working directory permissions to ensure GitHub Actions can write to it
RUN chmod -R 777 /app

# Set the entrypoint to run commands with the dependencies
ENTRYPOINT ["python", "-m"]
