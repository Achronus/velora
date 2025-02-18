FROM python:3.12

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the dependency files first for caching
COPY pyproject.toml poetry.lock ./

# Configure Poetry to install dependencies into the system environment
RUN poetry config virtualenvs.create false

# Install only the documentation dependencies
RUN poetry install --no-interaction --no-root --only docs

# Set permissions for mounted volume
RUN chmod -R 777 /app

# Default command for building and serving docs
ENTRYPOINT ["mkdocs"]
CMD ["serve", "--dev-addr=0.0.0.0:8000"]
