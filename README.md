# Dexterous hand teleoperation for RH56DFTP

![Teleoperation Demo](https://github.com/user-attachments/assets/16e64ba8-e06f-4baf-946e-cc89f3693899)

CV-pipeline for dexterous robotic hand teleoperation. Capture video frames, estimate MANO joint parameters and stream rotation vectors to robotic hand.

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

## Prerequisites (CLIENT)

- Windows 10/11, macOS, or Linux
- Webcam
- Python 3.10 (Conda)

## Environment Setup (CLIENT)

Create and activate Conda environment:
```bash
conda create -n env_lilteleop_client python=3.10 -y
conda activate env_lilteleop_client
```

Install lightweight dependencies:
```bash
pip install -r requirements_client.txt
```

Install pinocchio:
```bash
conda install -c conda-forge pinocchio -y
```

## Run pipeline

On server:

```bash
python src/handpos_server.py
```

On client:

```bash
python src/handpos_client.py
python src/rerun_vis.py         # for rerun visualization
python src/run_rh56dftp.py      # for physical hand teleop
```

## Acknowledgments

Retargeting logic and hand configs in src/retargeting/ are based on dex-retargeting by DexSuite.

Source: https://github.com/dexsuite/dex-retargeting.git