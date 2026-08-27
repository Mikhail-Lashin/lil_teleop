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

## Run pipeline

On server:

```bash
python src/server.py
```

On client:

```bash
python src/client.py
python src/rerun_vis.py
```

## Run tests

Check HaMeR work on server:

```bash
python scripts/test_hamer.py
```

Check zmq bus on client (while pipeline is running):

```bash
python scripts/test_zmq.py
```

Check physical RH56DFTP robot hand on client (while it's connected to client via ethernet):

```bash
python scripts/test_rh56dftp.py
```

## Acknowledgments

Retargeting logic and hand configs in src/retargeting/ are based on dex-retargeting by DexSuite.

Source: https://github.com/dexsuite/dex-retargeting.git