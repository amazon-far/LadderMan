from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking
from active_adaptation.envs.mdp.base import Termination as BaseTermination

import torch
from typing import List
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate, axis_angle_from_quat
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

class _cum_error_mixin:
    def __init__(self, min_steps: int=1, threshold: float=0.25, **kwargs):
        super().__init__(**kwargs)
        self.min_steps = min_steps
        self.threshold = threshold

        with torch.device(self.device):
            self.error = torch.zeros(self.num_envs)
            self.__exceeded = torch.zeros(self.num_envs, dtype=bool)
            self.__cum_steps = torch.zeros(self.num_envs, dtype=torch.int32)
        
    def update(self):
        self.__exceeded = self.error >= self.threshold
        self.__cum_steps[self.__exceeded] += 1
        self.__cum_steps[~self.__exceeded] = 0

    def reset(self, env_ids):
        self.__cum_steps[env_ids] = 0
    
    def __call__(self):
        return (self.__cum_steps >= self.min_steps).unsqueeze(-1)
        
    @property
    def cum_steps(self):
        return self.__cum_steps

RobotTrackTermination = BaseTermination[RobotTracking]


class cum_foot_both_lost_contact(_cum_error_mixin, RobotTrackTermination):
    def __init__(
        self,
        body_names: str | List[str] = ".*ankle_roll_link",
        min_upward_force: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contact_sensor = self.env.scene["contact_forces"]
        body_ids, _ = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self.min_upward_force = min_upward_force

    def update(self):
        contact_time = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.env.step_dt
        net_forces_w = self.contact_sensor.data.net_forces_w[:, self.body_ids]
        upward_support = net_forces_w[..., 2] > self.min_upward_force
        in_contact = contact_time & upward_support
        both_lost = (~in_contact).all(dim=1)
        self.error[:] = 2 * both_lost.float()
        super().update()

class cum_foot_any_lost_contact(_cum_error_mixin, RobotTrackTermination):
    def __init__(
        self,
        body_names: str | List[str] = ".*ankle_roll_link",
        min_upward_force: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contact_sensor = self.env.scene["contact_forces"]
        body_ids, _ = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self.min_upward_force = min_upward_force

    def update(self):
        contact_time = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.env.step_dt
        net_forces_w = self.contact_sensor.data.net_forces_w[:, self.body_ids]
        upward_support = net_forces_w[..., 2] > self.min_upward_force
        in_contact = contact_time & upward_support
        any_lost = (~in_contact).any(dim=1)
        self.error[:] = 2 * any_lost.float()
        super().update()

class cum_body_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_pos_w = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        robot_body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_pos_w - robot_body_pos_w).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()

class cum_body_z_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_pos_w = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        robot_body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_pos_w - robot_body_pos_w)[..., 2].abs()
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()
    
class cum_body_ori_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_quat_w = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]
        robot_body_quat_w = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_quat_diff = quat_mul(quat_conjugate(ref_body_quat_w), robot_body_quat_w)
        body_ori_error = axis_angle_from_quat(body_quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()
    
class cum_body_pos_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_pos_w = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        ref_root_pos_w = self.command_manager.ref_root_pos_w[:, None, :].clone()
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, :]
        
        robot_body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        robot_root_pos_w = self.command_manager.asset.data.root_link_pos_w[:, None, :].clone()
        robot_root_quat_w = self.command_manager.asset.data.root_link_quat_w[:, None, :]
        
        ref_root_pos_w[..., 2] = 0.0
        robot_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_pos_local = quat_apply_inverse(ref_root_quat_w, ref_body_pos_w - ref_root_pos_w)
        robot_body_pos_local = quat_apply_inverse(robot_root_quat_w, robot_body_pos_w - robot_root_pos_w)

        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_pos_local - robot_body_pos_local).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()
    
class cum_body_ori_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_quat_w = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, :]
        
        robot_body_quat_w = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        robot_root_quat_w = self.command_manager.asset.data.root_link_quat_w[:, None, :]
        
        ref_root_quat_w = yaw_quat(ref_root_quat_w).expand_as(ref_body_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w).expand_as(robot_body_quat_w)

        ref_body_quat_local = quat_mul(quat_conjugate(ref_root_quat_w), ref_body_quat_w)
        robot_body_quat_local = quat_mul(quat_conjugate(robot_root_quat_w), robot_body_quat_w)

        # shape: [num_envs, num_tracking_bodies, 3]
        body_quat_diff = quat_mul(quat_conjugate(ref_body_quat_local), robot_body_quat_local)
        body_ori_error = axis_angle_from_quat(body_quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()
    



class cum_ladder_lost_contact_steps(_cum_error_mixin, RobotTrackTermination):
    def __init__(
        self,
        body_names: str | List[str] = ".*wrist_yaw_link",
        pos_thres: float = 0.05,
        min_upward_force: float = 5.0,
        max_force_angle_deg: float = 60.0,
        threshold: float = 1.0,
        strict_position: bool = False,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.contact_sensor = self.env.scene["contact_forces"]
        body_ids, _ = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self.pos_thres = pos_thres
        self.min_upward_force = min_upward_force
        self.max_force_angle_deg = max_force_angle_deg
        self.strict_position = strict_position
    
    def update(self):
        eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
        contact_pos = eef_pos_diff.norm(dim=-1) < self.pos_thres
        if self.strict_position:
            contact_pos = contact_pos & (eef_pos_diff[..., -1] > 0.0)

        net_forces_w = self.contact_sensor.data.net_forces_w[:, self.body_ids]
        force_norm = net_forces_w.norm(dim=-1)
        eps = torch.finfo(force_norm.dtype).eps
        force_dir_z = net_forces_w[..., 2] / (force_norm + eps)
        max_angle_rad = torch.deg2rad(
            torch.tensor(self.max_force_angle_deg, device=net_forces_w.device, dtype=net_forces_w.dtype)
        )
        angle_ok = force_dir_z > torch.cos(max_angle_rad)
        proj_z_ok = net_forces_w[..., 2] > self.min_upward_force
        upward_support = angle_ok & proj_z_ok

        in_contact = contact_pos & upward_support
        in_contact_left = in_contact[:, 0]
        in_contact_right = in_contact[:, 1]
        in_range_left = self.command_manager.ref_object_contact_left_hand.squeeze(-1).bool()
        in_range_right = self.command_manager.ref_object_contact_right_hand.squeeze(-1).bool()

        lost_contact_left = in_range_left & (~in_contact_left)
        lost_contact_right = in_range_right & (~in_contact_right)
        lost_contact = lost_contact_left | lost_contact_right
        self.error[:] = 2 * lost_contact.float()
        super().update()