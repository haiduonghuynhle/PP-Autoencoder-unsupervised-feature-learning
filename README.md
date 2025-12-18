# CUDA Autoencoder for CIFAR-10
CSC14120 - Parallel Programming — Final Project

## Overview
This repository contains a CUDA-accelerated convolutional autoencoder for unsupervised feature learning on the CIFAR-10 dataset, plus scripts to extract features and train an SVM classifier.

## Requirements
- Linux or WSL/Windows with CUDA toolchain installed (tested with CUDA-compatible A100)
- GNU Make, GCC, nvcc
- Python 3.8+ with: matplotlib, numpy, seaborn (for analysis & plotting)

## Repository layout
- `src/` - C++/CUDA source files (autoencoder training + kernels)
- `include/` - headers
- `models/` - trained model binaries (saved by training scripts)
- `data/` - CIFAR-10 binary files (downloaded by Make)
- `notebooks/CSC14120_Final_Report_v4.ipynb` - project report and analysis

## Quick setup
1. Download data and build

```bash
# fetch CIFAR-10 and prepare folders
make download_data

# Build everything (adjust Makefile GPU arch for your card if needed)
# For A100 (sm_80) the notebook used: sed -i 's/sm_75/sm_80/g' Makefile
make all

# Optional: build CPU-only binary
make cpu
```

## Training examples
Use the binaries in the repo root after building.

CPU baseline (2% data example used in logs):
```bash
./autoencoder_cpu --train --cpu --epochs 20 --batch-size 32 \
  --model models/autoencoder_cpu.bin --data data/cifar-10-batches-bin
```

GPU (default/basic):
```bash
./autoencoder_gpu --train --epochs 20 --batch-size 64 \
  --model models/autoencoder_gpu.bin --data data/cifar-10-batches-bin
```

GPU optimized (v1):
```bash
./autoencoder_gpu --train --opt-v1 --epochs 20 --batch-size 64 \
  --model models/autoencoder_gpu_opt_v1.bin --data data/cifar-10-batches-bin
```

GPU optimized (v2):
```bash
./autoencoder_gpu --train --opt-v2 --epochs 20 --batch-size 64 \
  --model models/autoencoder_gpu_opt_v2.bin --data data/cifar-10-batches-bin
```

## Feature extraction
Example commands (the notebook used `models/autoencoder_gpu_opt_v2.bin` for extraction):

```bash
# extract using CPU or GPU models
./autoencoder_gpu --extract-features --model models/autoencoder_gpu_opt_v2.bin \
  --data data/cifar-10-batches-bin
```

Feature extraction log highlights (from execution):
- Feature dimension: 8192
- Train features extracted in ~6.07s
- Test features extracted in ~1.20s
- Total feature extraction time: ~7.41s
- Output files: `models/train_features_gpu_opt_v2.bin`, `models/test_features_gpu_opt_v2.bin`

## SVM training (recommended using existing libraries)
Examples used in notebook (cuML / LIBSVM variants shown):

CPU (LIBSVM / scikit-learn example):
```bash
# example script (if using Python + sklearn)
python train_svm.py --train models/train_features_cpu.bin --test models/test_features_cpu.bin
```

GPU-accelerated SVM with cuML (as in notebook):
```bash
# install cuML/cupy for matching CUDA version (example shown in notebook)
pip install --no-cache-dir cupy-cuda12x
pip install --no-cache-dir cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# run cuML-based SVM training
python train_svm_cuml.py --train models/train_features_gpu_opt_v2.bin --test models/test_features_gpu_opt_v2.bin
```

## Demo
Run the Gradio demo using the trained autoencoder and SVM model (example):

```bash
python gradio_demo.py --autoencoder models/autoencoder_gpu.bin --svm models/svm_cuml_model_gpu.pkl --share
```

## Measured performance notes (from logs)
- CPU per-epoch time (2% data): ~1,030s
- GPU Basic per-epoch time: ~95s (≈10.8×)
- GPU Opt V1 per-epoch time: ~49.8s (≈20.7×)
- GPU Opt V2 per-epoch time: ~34.7s (≈29.7×)
- Feature extraction (full dataset on GPU opt v2): 7.41s total (train+test)
- Classification accuracy (SVM on extracted features): ≈67% (as reported in notebook)

## Notes & tips
- If you run on a different GPU, update the `sm_xx` architecture in the `Makefile` before compiling.
- For fast H2D copies, use pinned host memory and set appropriate CUDA stream usage (v2 opt).
- Save and keep the `models/` outputs; `*.bin` weights are used by the notebook for visualization and SVM training.

## Pretrained model

If you want to skip training and use a pretrained model, download the provided weights and place them in the `models/` directory.

Pretrained model [Google Drive](https://drive.google.com/drive/folders/1-CMgUYIaAk7DXq3R6uNtfg9G3CRhXQw2?usp=sharing)

Download instructions (example using `wget` + `gdown`):

```bash
# using gdown (recommended for Google Drive links)
pip install gdown
gdown --id 1A1CHdGLMx0ohzwS1Lwp8xFl4H25Lh2hf -O models/autoencoder_pretrained.bin

# or using your browser: download and move the file to `models/`:
# mv ~/Downloads/autoencoder_pretrained.bin models/
```

After placing the file in `models/`, you can run the demo using the downloaded model. For example:

```bash
python gradio_demo.py --autoencoder models/autoencoder_pretrained.bin --svm models/svm_cuml_model_gpu.pkl --share
```
