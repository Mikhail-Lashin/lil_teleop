import pybvh
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BVH_FILEPATH = PROJECT_ROOT / "test_data" / "test.bvh"

bvh = pybvh.read_bvh_file(BVH_FILEPATH)
bvh.play()