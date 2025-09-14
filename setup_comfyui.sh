#!/bin/bash
set -e  # exit if any command fails

# -----------------------------
# System dependencies
# -----------------------------
apt update && apt install -y git curl ffmpeg python3-venv python3-pip

# -----------------------------
# Clone ComfyUI
# -----------------------------
cd /workspace
if [ ! -d "ComfyUI" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git
fi

# -----------------------------
# Virtual environment (Python 3.x system default)
# -----------------------------
cd ComfyUI
python3 -m venv .comfyui
source .comfyui/bin/activate

# -----------------------------
# Base deps
# -----------------------------
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# -----------------------------
# Install PyTorch (CUDA 12.6 build for A40/4090 etc.)
# -----------------------------
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# -----------------------------
# Extra deps
# -----------------------------
pip install onnxruntime-gpu packaging ninja "accelerate>=1.1.1" "diffusers>=0.31.0" "transformers>=4.39.3" triton

# -----------------------------
# Custom nodes
# -----------------------------
cd custom_nodes

# ComfyUI Manager
if [ ! -d "ComfyUI-Manager" ]; then
  git clone https://github.com/ltdrdata/ComfyUI-Manager
fi

# TeaCache
if [ ! -d "ComfyUI-TeaCache" ]; then
  git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git
  cd ComfyUI-TeaCache
  pip install -r requirements.txt
  cd ..
fi

# SageAttention
cd /workspace/ComfyUI
if [ ! -d "SageAttention" ]; then
  git clone https://github.com/thu-ml/SageAttention
  cd SageAttention
  pip install -e .
  cd ..
fi

echo "✅ ComfyUI setup complete!"
