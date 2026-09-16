import numpy as np
import rerun as rr
import cv2
import os
import trimesh
import logging
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF
from utils.one_euro import OneEuroFilter

logging.getLogger("yourdfpy").setLevel(logging.ERROR)

R_hamer2urdf = np.array([   # ROTATION ISSUE DEBUG for rh56dftp URDF
    [-1, 0, 0],
    [ 0,-1, 0],
    [ 0, 0, 1] 
])

class CameraView:
    """Render video stream from camera in Rerun"""
    def __init__(self, entity_path):
        self.entity_path = entity_path

    def update(self, image_bytes):
        if image_bytes is None:
            return
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rr.log(self.entity_path, rr.Image(frame_rgb))
            
class HumanHandView:
    """Render operator hand skeleton in Rerun"""
    HUMAN_SKELETON_LINKS = [
        (0, 1), (1, 2), (2, 3), (3, 4),             # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),             # index
        (0, 9), (9, 10), (10, 11), (11, 12),        # middle
        (0, 13), (13, 14), (14, 15), (15, 16),      # ring
        (0, 17), (17, 18), (18, 19), (19, 20)       # pinky
    ]
    DEFAULT_POSE = np.array([
        [0.00,  0.00,  0.00],                                                                   #     0: wrist
        [0.02,  0.02,  0.00], [0.04,  0.04,  0.00], [0.06,  0.05,  0.00], [0.08,  0.05,  0.00], #   1-4: thumb
        [0.03,  0.00,  0.01], [0.07,  0.00,  0.02], [0.11,  0.00,  0.02], [0.15,  0.00,  0.02], #   5-8: index
        [0.03, -0.01,  0.00], [0.08, -0.02,  0.00], [0.13, -0.02,  0.00], [0.17, -0.02,  0.00], #  9-12: middle
        [0.03, -0.02, -0.01], [0.07, -0.03, -0.01], [0.11, -0.03, -0.01], [0.14, -0.03, -0.01], # 13-16: ring
        [0.02, -0.03, -0.02], [0.05, -0.04, -0.02], [0.08, -0.04, -0.02], [0.10, -0.04, -0.02]  # 17-20: pinky
    ])
    
    def __init__(self, root_entity, freq=50, min_cutoff=2.0, beta=0.03, d_cutoff=1.0):
        self.root_entity = root_entity
        self.filter = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def update(self, joints):
        if joints is not None:
            raw_joints = np.array(joints) - np.array(joints)[0]
        else:
            raw_joints = self.DEFAULT_POSE

        # filter
        filtered_joints = self.filter.filter(raw_joints)

        # joint points
        rr.log(
            f"{self.root_entity}/Joints",
            rr.Points3D(
                filtered_joints,
                colors=[[0, 200, 255] for _ in range(len(filtered_joints))],
                radii=0.005
            )
        )

        # links
        bone_strips = [[filtered_joints[start], filtered_joints[end]] for start, end in self.HUMAN_SKELETON_LINKS]
        rr.log(
            f"{self.root_entity}/Skeleton_Bones",
            rr.LineStrips3D(
                bone_strips,
                colors=[[255, 255, 255] for _ in range(len(self.HUMAN_SKELETON_LINKS))],
                radii=0.001
            )
        )
        
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
        joints_array = joints_array @ R_hamer2urdf.T    # ROTATION ISSUE DEBUG for rh56dftp URDF
        
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
            
        # draw target vectors
        if hasattr(optimizer, "origin_link_names"):
            origins = np.array([
                self.robot.get_transform(name)[:3, 3] 
                for name in optimizer.origin_link_names
            ])
        else:
            origins = np.zeros_like(ref_value)
        
        self.draw_target_vectors(origins, ref_value)
        
    def draw_target_vectors(self, origins, vectors):
        rr.log(
            f"{self.root_entity}/Target_Vectors",
            rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                colors=[[255, 255, 0] for _ in range(len(vectors))],
                radii=0.002
            )
        )

    def close(self):
        self.fnull.close()