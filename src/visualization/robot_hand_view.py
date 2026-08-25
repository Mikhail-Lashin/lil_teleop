import os
import trimesh
import logging
import contextlib
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

from retargeting.rh56dftp_retargeter_draft import RH56DFTPRetargeter
from utils.one_euro import OneEuroFilter

logging.getLogger("yourdfpy").setLevel(logging.ERROR)

class RobotHandView:
    """Render robot hand in Rerun"""
    def __init__(self, urdf_path, joints, retargeter, root_entity, freq=50, min_cutoff=2.0, beta=0.03, d_cutoff=1.0):
        self.root_entity = root_entity
        self.retargeter = retargeter
        self.robot = self.retargeter.robot
        self.joints = joints
        
        self.filters = {
            j_name: OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for j_name in self.joints
        }
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

    def update(self, mano_params):
        if mano_params is None:
            return
            
        mano_array = np.array(mano_params)
        mano_aa = [R.from_matrix(m).as_rotvec() for m in mano_array] if mano_array.ndim == 3 else mano_array

        # compute angles
        with contextlib.redirect_stdout(self.fnull):
            robot_commands = self.retargeter.compute_robot_angles(mano_aa)
            filtered_commands = {
                j: self.filters[j].filter(val) if j in self.filters else val
                for j, val in robot_commands.items()
            }
            self.robot.update_cfg(filtered_commands)

        # move links
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