# Dockerfile
#
# Builds a GPU-ready container with dependencies pre-installed.
# Code is pulled at container startup via entrypoint.sh, not baked in.
#
# Usage:
#   docker build -t orderedlearning .
#   docker build --build-arg BASE_IMAGE_TAG=your/base:tag -t orderedlearning .
#
# RunPod users: The default base image and CMD are configured for RunPod.
# For other platforms, override CMD with your own startup command.

ARG BASE_IMAGE_TAG=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

FROM ${BASE_IMAGE_TAG}

WORKDIR /workspace

# Install git (needed for entrypoint clone/pull)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Remove the requirements file (code is pulled at startup, not baked in)
RUN rm -f requirements.txt

# Copy the startup script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint script to run on container start
ENTRYPOINT ["/entrypoint.sh"]

# RunPod's default start script (JupyterLab, SSH, etc.)
# Override for other platforms: e.g., CMD ["bash"]
CMD ["/start.sh"]
