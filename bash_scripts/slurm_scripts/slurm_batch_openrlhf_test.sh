#!/bin/bash 

#SBATCH -J docker_test
#SBATCH -p gpu03
  
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

mkdir -p logs

IMAGE="openrlhf/openrlhf:latest"
export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

docker run --rm --gpus all \
  -e NVIDIA_VISIBLE_DEVICES \
  "$IMAGE" bash -lc '
    nvidia-smi;
    python - <<EOF
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
EOF
  '
