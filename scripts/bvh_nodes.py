import pybvh

FILEPATH = r"test_data\bvh_misha_no_xyz_chr01_MAYA.bvh"

bvh = pybvh.read_bvh_file(FILEPATH)
poses = bvh.node_positions(centered="skeleton")

print(f"\n>>> node_positions.shape = {poses.shape}")

sorted_nodes = sorted(bvh.node_index.items(), key=lambda x: x[1])

print("\n>>> All nodes:")
for name, idx in sorted_nodes:
    print(f">>> [{idx:02d}] {name:<25}")