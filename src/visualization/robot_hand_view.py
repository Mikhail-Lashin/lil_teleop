import os
import trimesh
import logging
import contextlib
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF

logging.getLogger("yourdfpy").setLevel(logging.ERROR)

MANO_TRANSFORM = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0, 1]
])

class RobotHandView:
    """Render robot hand in Rerun"""
    def __init__(self, urdf_path, retargeter, root_entity):
        self.root_entity = root_entity
        self.retargeter = retargeter  # SeqRetargeting object from dex-retargeting
        self.robot = URDF.load(urdf_path)
        
        self.fnull = open(os.devnull, 'w')
        self._load_meshes(urdf_path)

    def _load_meshes(self, urdf_path):
        print(">>> Loading robot hand meshes in Rerun...")
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        
        for link_name, link in self.robot.link_map.items():
            if link.visuals:
                for visual in link.visuals:
                    if visual.geometry and visual.geometry.mesh and visual.geometry.mesh.filename:
                        clean_path = visual.geometry.mesh.filename.replace("package://", "").replace("file://", "")
                        possible_paths = [
                            os.path.normpath(os.path.join(urdf_dir, clean_path)),
                            os.path.normpath(os.path.join(urdf_dir, "meshes", os.path.basename(clean_path))),
                            os.path.normpath(os.path.join(urdf_dir, "..", clean_path)),
                        ]
                        
                        target_mesh_path = next((p for p in possible_paths if os.path.exists(p)), None)
                        if target_mesh_path:
                            try:
                                m = trimesh.load(target_mesh_path)
                                meshes = m.dump() if hasattr(m, 'dump') else [m]
                                for sub_m in meshes:
                                    if hasattr(sub_m, 'vertices') and hasattr(sub_m, 'faces'):
                                        normals = getattr(sub_m, 'vertex_normals', None)
                                        rr.log(
                                            f"{self.root_entity}/{link_name}",
                                            rr.Mesh3D(
                                                vertex_positions=sub_m.vertices,
                                                triangle_indices=sub_m.faces,
                                                vertex_normals=normals
                                            ),
                                            static=True
                                        )
                            except Exception:
                                pass

    def update(self, joints3d):
        if joints3d is None:
            return
            
        joints_array = np.array(joints3d)
        joints_array = np.array(joints3d) - np.array(joints3d)[0]
        joints_array = joints_array @ MANO_TRANSFORM.T
        
        optimizer = self.retargeter.optimizer

        # compute operator vectors
        if optimizer.retargeting_type == "POSITION":
            ref_value = joints_array[optimizer.target_link_human_indices, :]
        else:
            origin_idx = optimizer.target_link_human_indices[0, :]
            task_idx = optimizer.target_link_human_indices[1, :]
            ref_value = joints_array[task_idx, :] - joints_array[origin_idx, :]

        # compute robot angles and update
        qpos = self.retargeter.retarget(ref_value)
        qpos_dict = dict(zip(optimizer.robot.dof_joint_names, qpos))
        self.robot.update_cfg(qpos_dict)

        # move joints in rerun
        for link_name in self.robot.link_map.keys():
            try:
                transform_matrix = self.robot.get_transform(link_name)
                rr.log(
                    f"{self.root_entity}/{link_name}",
                    rr.Transform3D(
                        translation=transform_matrix[:3, 3],
                        mat3x3=transform_matrix[:3, :3]
                    )
                )
            except Exception:
                pass
    
    def close(self):
        self.fnull.close()