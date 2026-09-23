# CAM: Content-Addressable Memory for Self-Supervised Learning

This repository implements **CAM (Content-Addressable Memory)** on top of the Vision Transformer (ViT) architecture. CAM solves the "dead slot" and memory collapse problems in traditional FIFO queues used by contrastive and self-supervised learning methods (like MoCo, DINO, iBOT).

Unlike a standard FIFO queue where representations blindly cycle through memory regardless of utility, CAM introduces **Semantic Refinement**. By actively pruning redundant clone features and preserving unique, highly-utilized representations, CAM drastically increases the information entropy and the temporal receptive field of the memory bank.

## Experimental Findings

In identical ViT-B training runs, **CAM vastly outperforms Pure FIFO queues**:
- **75% fewer dead slots**: Only 11% of CAM's memory slots go unused, compared to 46% in naive FIFO.
- **+171% Target Utility**: Sinkhorn-Knopp target matches skyrocket from 5,171 hits to 14,055 hits.
- **2x Temporal Receptive Field**: Useful items survive an average of 59 steps before being overwritten, compared to 28 steps in FIFO.

## Architecture & Codebase

This project strips out downstream bloat and focuses entirely on the SSL pre-training engine.

Key structural components:
* `cam/layers/memory.py`: The core CAM controller (Sensor, Selector, Actuator).
* `cam/layers/density.py`: The Semantic Razor implementation that prunes redundant clones based on neighborhood density.
* `cam/train/cam_meta_arch.py`: The main meta-architecture integrating the Teacher/Student ViTs and the memory queue.

## Getting Started

To launch a pre-training job using CAM on a SLURM cluster:
```bash
python cam/run/submit.py \
    --nodes 1 \
    --ngpus 8 \
    --config-file cam/configs/ssl_default_config.yaml \
    train.dataset_path=ImageNet:split=TRAIN \
    train.output_dir=./logs
```
