from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking
from active_adaptation.envs.mdp.base import Reward as BaseReward

from typing import List, Dict
from omegaconf import DictConfig
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate, axis_angle_from_quat, yaw_quat

import torch


TrackReward = BaseReward[RobotTracking]

class _tracking_keypoint(TrackReward):
    def __init__(self, body_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_keypoint_names
        
        self.sigma = sigma
        body_indices_motion, matched_names_motion = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)
        body_indices_asset, matched_names_asset = resolve_matching_names(body_names, self.command_manager.asset.body_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "body names in motion dataset and robot not matched"
        assert set(matched_names) <= set(self.command_manager.tracking_keypoint_names), "Some body names in motion dataset not found in tracking body names"
        
        self.body_indices_motion = []
        self.body_indices_asset = []
        self.body_names = list(sorted(matched_names))
        self.num_bodies = len(self.body_names)
        for body_name in self.body_names:
            body_idx_motion = self.command_manager.tracking_keypoint_names.index(body_name)
            body_idx_asset = self.command_manager.asset.body_names.index(body_name)

            self.body_indices_motion.append(body_idx_motion)
            self.body_indices_asset.append(body_idx_asset)

        self.tolerance = torch.zeros(len(self.body_names), device=self.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, self.body_names)
            self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.device)
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

    def compute(self):
        raise NotImplementedError


class keypoint_pos_tracking_product(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_pos_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w
        
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

        diff = body_pos_motion_relative - body_pos_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_pos_error(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)

class keypoint_pos_error_local(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w
        
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

        diff = body_pos_motion_relative - body_pos_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_tracking_product(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
    
class keypoint_ori_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_ori_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_ori_asset)
        body_ori_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_ori_motion)

        diff = quat_mul(quat_conjugate(body_ori_motion_relative), body_ori_asset_relative)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_ori_error(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)
    
class keypoint_ori_error_local(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_ori_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_ori_asset)
        body_ori_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_ori_motion)

        diff = quat_mul(quat_conjugate(body_ori_motion_relative), body_ori_asset_relative)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)

class keypoint_lin_vel_tracking_product(_tracking_keypoint):
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset]
        body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
        diff = body_lin_vel_motion - body_lin_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
class keypoint_ang_vel_tracking_product(_tracking_keypoint):
    def compute(self):
        body_ang_vel_asset = self.command_manager.asset.data.body_com_ang_vel_w[:, self.body_indices_asset]
        body_ang_vel_motion = self.command_manager.ref_body_ang_vel_w[:, self.body_indices_motion]
        diff = body_ang_vel_motion - body_ang_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class _tracking_joint(TrackReward):
    def __init__(self, joint_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names
    
        self.sigma = sigma
        joint_indices_asset, matched_names_asset = resolve_matching_names(joint_names, self.command_manager.asset.joint_names)
        joint_indices_motion, matched_names_motion = resolve_matching_names(joint_names, self.command_manager.tracking_joint_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "joint names in motion dataset and robot not matched"
        assert set(matched_names) <= set(self.command_manager.tracking_joint_names), "Some joint names in motion dataset not found in tracking joint names"

        self.joint_indices_motion = []
        self.joint_indices_asset = []
        self.joint_names = list(sorted(matched_names))
        for joint_name in self.joint_names:
            joint_idx_motion = self.command_manager.tracking_joint_names.index(joint_name)
            joint_idx_asset = self.command_manager.asset.joint_names.index(joint_name)

            self.joint_indices_motion.append(joint_idx_motion)
            self.joint_indices_asset.append(joint_idx_asset)

        self.tolerance = torch.zeros(len(self.joint_names), device=self.env.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, matched_names_motion)
            self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.env.device)
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

class joint_pos_tracking_product(_tracking_joint):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
        joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_joints]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
    
class joint_pos_error(_tracking_joint):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
        joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)
    
class joint_vel_tracking_product(_tracking_joint):
    def compute(self):
        joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.joint_indices_asset]
        joint_vel_motion = self.command_manager.ref_joint_vel[:, self.joint_indices_motion]
        diff = joint_vel_motion - joint_vel_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_joints]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class ladder_eef_contact_exp(TrackReward):
    def __init__(
        self,
        pos_sigma: float=0.1,
        pos_tolerance: float=0.0,
        gain: float=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gain = gain
        self.eef_pos_error = torch.zeros(self.num_envs, self.command_manager.num_eefs, device=self.device)

        self.pos_sigma = pos_sigma
        self.pos_tolerance = pos_tolerance
    
    def update(self):
        self.in_range_left = self.command_manager.ref_object_contact_left_hand.squeeze(-1).bool()
        self.in_range_right = self.command_manager.ref_object_contact_right_hand.squeeze(-1).bool()

        eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
        self.eef_pos_error[:] = (eef_pos_diff.norm(dim=-1) - self.pos_tolerance).clamp_min(0.0)

    def compute(self):
        rew = torch.exp(-self.eef_pos_error / self.pos_sigma)
        # shape: [num_envs]
        in_range_left = self.in_range_left.float()
        in_range_right = self.in_range_right.float()
        rew_left = rew[:, 0] * in_range_left * self.gain + 1 - in_range_left
        rew_right = rew[:, 1] * in_range_right * self.gain + 1 - in_range_right
        rew = 0.5 * (rew_left + rew_right)
        return rew.unsqueeze(-1)
