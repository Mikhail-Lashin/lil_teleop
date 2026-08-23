#!/bin/bash
set -e

echo ">>> Setting env for lil_teleop..."

PROJECT_ROOT=$(pwd)
mkdir -p "$PROJECT_ROOT/bin"

# nvcc dummy
cat << 'EOF' > "$PROJECT_ROOT/bin/nvcc"
#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "Cuda compilation tools, release 13.0, V13.0.0"
else
    exec /usr/bin/nvcc "$@"
fi
EOF
chmod +x "$PROJECT_ROOT/bin/nvcc"
export PATH="$PROJECT_ROOT/bin:$PATH"

# basic libs
echo ">>> Installing requirements_server.txt..."
pip install --no-build-isolation -r requirements_server.txt

# chumpy fix
echo ">>> Patching chumpy..."
python3 -c "
import os, glob
for p in glob.glob('$CONDA_PREFIX/lib/python3.10/site-packages/chumpy/__init__.py'):
    with open(p, 'r') as f: text = f.read()
    old = 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf'
    new = 'from numpy import nan, inf\nbool=bool; int=int; float=float; complex=complex; object=object; str=str; unicode=str'
    if old in text:
        with open(p, 'w') as f: f.write(text.replace(old, new))
"

# Detectron2 (pure python)
echo ">>> Building Detectron2..."
rm -rf /tmp/detectron2
git clone --quiet https://github.com/facebookresearch/detectron2.git /tmp/detectron2
python3 -c "
with open('/tmp/detectron2/setup.py', 'r') as f:
    text = f.read()
text = text.replace('ext_modules=get_extensions(),', 'ext_modules=[],')
with open('/tmp/detectron2/setup.py', 'w') as f:
    f.write(text)
"
pip install -e /tmp/detectron2 --no-build-isolation

# ViTPose
echo ">>> Installing ViTPose..."
pip install git+https://github.com/ViTAE-Transformer/ViTPose.git --no-build-isolation

# HaMeR
echo ">>> Installing HaMeR..."
pip install git+https://github.com/geopavlakos/hamer.git --no-deps

echo ">>> Removing temporary files..."
rm -rf "$PROJECT_ROOT/bin"

echo "✅ Environment successfully configured!"