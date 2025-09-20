# ===================================================================
# Docker Configuration for Zomato Restaurant Rating Prediction API
# Uses conda environment for complete dependency management
# ===================================================================

# Build stage
FROM condaforge/mambaforge:latest as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -afy

# Production stage
FROM condaforge/mambaforge:latest as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/envs/newAge/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder stage
COPY --from=builder /opt/conda/envs/newAge /opt/conda/envs/newAge

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY zomato_prediction/ ./zomato_prediction/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY pyproject.toml ./

# Create necessary directories
RUN mkdir -p logs temp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "zomato_prediction.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
