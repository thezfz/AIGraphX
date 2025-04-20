# Containerfile for AIGraphX FastAPI application
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive \
    CONDA_AUTO_UPDATE_CONDA=false \
    PATH="/opt/conda/bin:$PATH"

# Install system dependencies (less frequent changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (less frequent changes)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda init bash && \
    conda --version

# Set the working directory
WORKDIR /app

# --- Environment Setup (depends only on environment.yml) ---
# Copy environment definition file first
COPY environment.yml environment.yml

# Create Conda environment from environment.yml (Handles both Conda and Pip deps)
RUN echo "Creating Conda environment AIGraphX from environment.yml..." && \
    conda env create -f environment.yml --name AIGraphX && \
    # Clean up conda caches after creating environment
    conda clean -afy

# --- Verification Step (depends on environment install) ---
RUN echo "Verifying uvicorn installation in AIGraphX environment..." && \
    # Try using conda run directly (should work if installed correctly)
    conda run -n AIGraphX which uvicorn && \
    conda run -n AIGraphX uvicorn --version && \
    echo "Uvicorn found and executable via conda run." || \
    (echo "ERROR: Uvicorn not found or not executable via conda run! Check environment.yml and installation steps." && exit 1)

# --- Application Code Copy (more frequent changes) ---
# Copy the rest of the application code *after* environment setup
COPY . .

# --- Final Configuration ---
EXPOSE 8000

# Ensure ENTRYPOINT is removed or commented out
# ENTRYPOINT ["/opt/conda/envs/AIGraphX/bin/python", "-m"]

# Use shell form CMD to allow shell variable substitution before conda run
CMD conda run -n AIGraphX --no-capture-output uvicorn aigraphx.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --reload