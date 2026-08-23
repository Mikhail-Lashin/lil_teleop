# Dexterous hand teleoperation for RH56DFTP

CV-pipeline for robotic hand teleoperation using HaMeR. The system captures video frames, estimates MANO 3D joint parameters and streams rotation vectors to a control endpoint.

## Prerequisites (SERVER)

- Linux (Ubuntu 20.04 / 22.04)
- NVIDIA GPU (CUDA 11.8+ / CUDA 12.x)
- Python 3.10
- Conda

## Required Assets (SERVER)

The application expects pre-trained weights and MANO model files in the `./_DATA` directory:

```text
_DATA/
├── data/
│   └── mano/
│       └── MANO_RIGHT.pkl
└── hamer_ckpts/
```

## Environment Setup (SERVER)

Create and activate Python environment:

```bash
conda create -n env_lilteleop python=3.10 -y
conda activate env_lilteleop
```

Install PyTorch with CUDA support matching your driver, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --ignore-installed
```

Execute the environment build script:

```bash
chmod +x ./scripts/setup_server_env.sh
./scripts/setup_server_env.sh
```

## Test (SERVER)

```bash
python scripts/test_hamer.py
```