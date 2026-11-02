from abc import abstractmethod
from typing import List, Optional, Dict

import nlopt
import numpy as np
import torch

from retargeting.kinematics_adaptor import (
    KinematicAdaptor,
    MimicJointKinematicAdaptor,
)
from retargeting.robot_wrapper import RobotWrapper


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )

        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3)
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class InspireLUT1800Optimizer(Optimizer):
    """Retargeting optimizer using precomputed Forward Kinematics Look-Up Tables (LUT).

    This optimizer is specifically tailored for 6-DoF underactuated anthropomorphic hands
    such as the Inspire Robots RH56DFTP (four 1-DoF fingers with mimic joints and one 2-DoF thumb).
    Instead of running iterative non-linear optimization (SLSQP) at each step, it decouples
    the kinematics per finger, evaluates a dense 1D LUT for the four fingers, and combines
    a 2D coarse LUT search with a single Gauss-Newton refinement step for the thumb.

    Args:
        robot: RobotWrapper instance handling Pinocchio forward kinematics.
        target_joint_names: List of active joint names to be optimized.
        target_origin_link_names: List of origin link names for each reference vector.
        target_task_link_names: List of task (fingertip) link names for each reference vector.
        target_link_human_indices: Indices mapping human hand keypoints to the target vectors.
        huber_delta: Smooth L1 loss threshold (kept for configuration compatibility).
        norm_delta: Weight for temporal regularization toward the previous joint positions.
        scaling: Scaling factor applied to the human reference vectors.
        max_register_val: Resolution of the 1D actuator grid (default: 1800).
        thumb_grid_size: Resolution of the 2D coarse grid for the 2-DoF thumb (default: 61).
    """

    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta: float = 0.02,
        norm_delta: float = 4e-3,
        scaling: float = 1.0,
        max_register_val: int = 1800,
        thumb_grid_size: int = 61,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_delta = huber_delta
        self.norm_delta = norm_delta
        self.scaling = scaling
        self.max_register_val = max_register_val
        self.thumb_grid_size = thumb_grid_size

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = [
            self.computed_link_names.index(name) for name in target_origin_link_names
        ]
        self.task_link_indices = [
            self.computed_link_names.index(name) for name in target_task_link_names
        ]

        # Sanity check and cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)
        self.num_vecs = len(target_task_link_names)

        # LUT state and buffers
        self.joint_limits: Optional[np.ndarray] = None
        self.lut_ready: bool = False
        self.f1d_vec_indices: List[int] = []
        self.f1d_joint_indices: List[int] = []
        self.f1d_q_grids: Optional[np.ndarray] = None
        self.f1d_vec_grids: Optional[np.ndarray] = None
        self.thumb_group: Optional[Dict[str, np.ndarray]] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon: float = 1e-3):
        super().set_joint_limit(joint_limits, epsilon)
        self.joint_limits = joint_limits.astype(np.float32)
        self.lut_ready = False

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        super().set_kinematic_adaptor(adaptor)
        # Invalidate LUT so it is rebuilt with the mimic joint adaptor included
        self.lut_ready = False

    def _compute_vectors(self, q_target: np.ndarray, fixed_qpos: np.ndarray) -> np.ndarray:
        """Compute 3D task vectors in the robot frame for a given target joint configuration."""
        qpos = np.zeros(self.num_joints, dtype=np.float64)
        qpos[self.idx_pin2fixed] = fixed_qpos
        qpos[self.idx_pin2target] = q_target

        # Kinematics forwarding for qpos (handles mimic joints)
        if self.adaptor is not None:
            qpos[:] = self.adaptor.forward_qpos(qpos)[:]

        self.robot.compute_forward_kinematics(qpos)
        target_link_poses = [
            self.robot.get_link_pose(index) for index in self.computed_link_indices
        ]
        body_pos = np.array([pose[:3, 3] for pose in target_link_poses], dtype=np.float32)

        origin_link_pos = body_pos[self.origin_link_indices, :]
        task_link_pos = body_pos[self.task_link_indices, :]
        return task_link_pos - origin_link_pos

    def _build_lut(self, fixed_qpos: np.ndarray):
        """Precompute 1D and 2D forward kinematics Look-Up Tables for decoupled finger joints."""
        if self.joint_limits is None:
            raise ValueError("Joint limits must be set via set_joint_limit() before retargeting.")

        q_mid = 0.5 * (self.joint_limits[:, 0] + self.joint_limits[:, 1])
        base_vecs = self._compute_vectors(q_mid, fixed_qpos)

        # Automatically detect which joints affect which fingertip vector
        vec_to_joints: List[List[int]] = [[] for _ in range(self.num_vecs)]
        for j_idx in range(self.opt_dof):
            q_pert = q_mid.copy()
            q_pert[j_idx] = self.joint_limits[j_idx, 1]
            diff = np.linalg.norm(
                self._compute_vectors(q_pert, fixed_qpos) - base_vecs, axis=1
            )
            for v_i in range(self.num_vecs):
                if diff[v_i] > 1e-4:
                    vec_to_joints[v_i].append(j_idx)

        n_1d = self.max_register_val + 1
        n_2d = self.thumb_grid_size

        self.f1d_vec_indices.clear()
        self.f1d_joint_indices.clear()
        f1d_q_list, f1d_vec_list = [], []
        self.thumb_group = None

        for v_i, j_list in enumerate(vec_to_joints):
            if len(j_list) == 1:
                # 1-DoF finger (index, middle, ring, pinky)
                j0 = j_list[0]
                q_grid = np.linspace(
                    self.joint_limits[j0, 0], self.joint_limits[j0, 1], n_1d, dtype=np.float32
                )
                vec_grid = np.zeros((n_1d, 3), dtype=np.float32)
                q_test = q_mid.copy()
                for k, q_val in enumerate(q_grid):
                    q_test[j0] = q_val
                    vec_grid[k] = self._compute_vectors(q_test, fixed_qpos)[v_i]

                self.f1d_vec_indices.append(v_i)
                self.f1d_joint_indices.append(j0)
                f1d_q_list.append(q_grid)
                f1d_vec_list.append(vec_grid)

            elif len(j_list) == 2:
                # 2-DoF thumb (opposition/yaw and flexion/pitch)
                j0, j1 = j_list[0], j_list[1]
                q0_lin = np.linspace(
                    self.joint_limits[j0, 0], self.joint_limits[j0, 1], n_2d, dtype=np.float32
                )
                q1_lin = np.linspace(
                    self.joint_limits[j1, 0], self.joint_limits[j1, 1], n_2d, dtype=np.float32
                )
                vec_2d = np.zeros((n_2d, n_2d, 3), dtype=np.float32)
                q_test = q_mid.copy()
                for i0, val0 in enumerate(q0_lin):
                    q_test[j0] = val0
                    for i1, val1 in enumerate(q1_lin):
                        q_test[j1] = val1
                        vec_2d[i0, i1] = self._compute_vectors(q_test, fixed_qpos)[v_i]

                g0, g1 = np.meshgrid(q0_lin, q1_lin, indexing="ij")
                self.thumb_group = {
                    "vec_idx": v_i,
                    "joints": j_list,
                    "q0_lin": q0_lin,
                    "q1_lin": q1_lin,
                    "vec_2d": vec_2d,
                    "q_flat": np.stack([g0.ravel(), g1.ravel()], axis=1),
                    "vec_flat": vec_2d.reshape(-1, 3),
                }

        if f1d_q_list:
            self.f1d_q_grids = np.stack(f1d_q_list, axis=0)      # (num_1d_fingers, n_1d)
            self.f1d_vec_grids = np.stack(f1d_vec_list, axis=0)  # (num_1d_fingers, n_1d, 3)

        self.lut_ready = True

    def retarget(self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray) -> np.ndarray:
        """
        Compute the retargeting results using precomputed kinematic LUT and local refinement.
        Args:
            ref_value: the reference vectors in cartesian space as input
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names
        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )

        if not self.lut_ready:
            self._build_lut(np.asarray(fixed_qpos, dtype=np.float64))

        last_qpos_arr = np.asarray(last_qpos, dtype=np.float32)
        target_vecs = np.asarray(ref_value, dtype=np.float32) * self.scaling
        out_qpos = last_qpos_arr.copy()

        # Vectorized 1D LUT search for all single-DoF fingers
        if self.f1d_q_grids is not None and self.f1d_vec_grids is not None:
            f_targets = target_vecs[self.f1d_vec_indices][:, None, :]
            diff_fv = self.f1d_vec_grids - f_targets
            pos_f_err = np.einsum("fki,fki->fk", diff_fv, diff_fv)
            last_f_q = last_qpos_arr[self.f1d_joint_indices][:, None]
            reg_f_err = self.norm_delta * ((self.f1d_q_grids - last_f_q) ** 2)
            best_k = np.argmin(pos_f_err + reg_f_err, axis=1)
            out_qpos[self.f1d_joint_indices] = self.f1d_q_grids[
                np.arange(len(best_k)), best_k
            ]

        # 2D coarse LUT search + local Gauss-Newton refinement for the 2-DoF thumb
        if self.thumb_group is not None:
            g = self.thumb_group
            j0, j1 = g["joints"]
            t_vec = target_vecs[g["vec_idx"]]
            last_q2 = np.array([last_qpos_arr[j0], last_qpos_arr[j1]], dtype=np.float32)

            diff_v = g["vec_flat"] - t_vec
            pos_err = np.einsum("ij,ij->i", diff_v, diff_v)
            diff_q = g["q_flat"] - last_q2
            reg_err = self.norm_delta * np.einsum("ij,ij->i", diff_q, diff_q)

            flat_idx = int(np.argmin(pos_err + reg_err))
            q0_lin, q1_lin, vec_2d = g["q0_lin"], g["q1_lin"], g["vec_2d"]
            n0, n1 = len(q0_lin), len(q1_lin)
            bi, bj = divmod(flat_idx, n1)

            # Finite-difference local Jacobian around the best grid cell
            i_next, i_prev = min(bi + 1, n0 - 1), max(bi - 1, 0)
            j_next, j_prev = min(bj + 1, n1 - 1), max(bj - 1, 0)
            dv_d0 = (vec_2d[i_next, bj] - vec_2d[i_prev, bj]) / (
                q0_lin[i_next] - q0_lin[i_prev] + 1e-8
            )
            dv_d1 = (vec_2d[bi, j_next] - vec_2d[bi, j_prev]) / (
                q1_lin[j_next] - q1_lin[j_prev] + 1e-8
            )

            jacobian_2d = np.column_stack([dv_d0, dv_d1])
            residual = t_vec - vec_2d[bi, bj]
            delta_q = np.linalg.solve(
                jacobian_2d.T @ jacobian_2d + 1e-4 * np.eye(2, dtype=np.float32),
                jacobian_2d.T @ residual,
            )

            step0, step1 = abs(q0_lin[1] - q0_lin[0]), abs(q1_lin[1] - q1_lin[0])
            out_qpos[j0] = np.clip(
                q0_lin[bi] + np.clip(delta_q[0], -step0, step0), q0_lin[0], q0_lin[-1]
            )
            out_qpos[j1] = np.clip(
                q1_lin[bj] + np.clip(delta_q[1], -step1, step1), q1_lin[0], q1_lin[-1]
            )

        return out_qpos

    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        raise NotImplementedError(
            "InspireLUT1800Optimizer overrides retarget() directly and does not use NLopt objectives."
        )