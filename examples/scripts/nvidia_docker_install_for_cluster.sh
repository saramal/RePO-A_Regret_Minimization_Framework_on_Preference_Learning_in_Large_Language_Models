#!/bin/bash
set -euxo pipefail

# 1. install required package
sudo dnf install -y curl gnupg2

# 2. Setup NVIDIA GPG key & repo
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container.repo

# 3. Install NVIDIA Container Toolkit
# sudo dnf clean expire-cache
sudo dnf install -y nvidia-container-toolkit

# 4. Set NVIDIA container runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 5. restart docker
sudo systemctl restart docker

# check
sudo docker run --rm --runtime=nvidia nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
