#!/bin/bash 

#SBATCH -J docker_test
#SBATCH -p gpu03
  
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00


IMAGE="openrlhf/openrlhf:latest"
export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

docker run --rm --gpus all \
  -e NVIDIA_VISIBLE_DEVICES \
  nvidia/cuda:12.2.0-base nvidia-smi
