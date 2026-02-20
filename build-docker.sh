#!/bin/sh

CUDA_NEWER_BASE="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
CUDA_NEWER_TAG="latest-cuda12.8"
CUDA_OLDER_BASE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CUDA_OLDER_TAG="latest-cuda12.4"

IMAGE_NAME="${DOCKER_IMAGE_NAME:-orderedlearning}"

echo "Building Docker image: ${IMAGE_NAME}:${CUDA_NEWER_TAG}..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${CUDA_NEWER_TAG}" --build-arg BASE_IMAGE_TAG="${CUDA_NEWER_BASE}" .

if [ $? -ne 0 ]; then
  echo "Docker build failed! Aborting." >&2
  exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}:${CUDA_OLDER_TAG}..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${CUDA_OLDER_TAG}" --build-arg BASE_IMAGE_TAG="${CUDA_OLDER_BASE}" .

if [ $? -ne 0 ]; then
  echo "Docker build failed! Aborting." >&2
  exit 1
fi

echo "Pushing Docker image: ${IMAGE_NAME}:${CUDA_NEWER_TAG}..."
docker push "${IMAGE_NAME}:${CUDA_NEWER_TAG}"

if [ $? -ne 0 ]; then
  echo "Docker push failed! Aborting." >&2
  exit 1
fi

echo "Pushing Docker image: ${IMAGE_NAME}:${CUDA_OLDER_TAG}..."
docker push "${IMAGE_NAME}:${CUDA_OLDER_TAG}"

if [ $? -ne 0 ]; then
  echo "Docker push failed! Aborting." >&2
  exit 1
fi

echo "Docker build and push successful."
