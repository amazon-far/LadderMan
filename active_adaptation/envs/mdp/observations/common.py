from active_adaptation.envs.mdp.base import Observation
import active_adaptation.utils.symmetry as sym_utils

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.assets import Articulation

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

class root_ang_vel_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()
    
    def reset(self, env_ids):
        root_ang_vel_b = self.asset.data.root_ang_vel_b[env_ids]
        root_ang_vel_b = root_ang_vel_b.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer[env_ids] = root_ang_vel_b

    def update(self):
        root_ang_vel_b = self.asset.data.root_ang_vel_b
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = root_ang_vel_b

    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

class projected_gravity_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()
    
    def reset(self, env_ids):
        projected_gravity_b = self.asset.data.projected_gravity_b[env_ids]
        projected_gravity_b = projected_gravity_b.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)
        self.buffer[env_ids] = self.asset.data.projected_gravity_b[env_ids].unsqueeze(1)
    
    def update(self):
        projected_gravity_b = self.asset.data.projected_gravity_b
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = projected_gravity_b
    
    def compute(self):
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

class joint_pos_history(Observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        history_steps: list[int]=[1], 
        noise_std: float=0.,
        set_to_zero_joint_names: str | None=None
    ):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(noise_std, 0.)
        self.asset: Articulation = self.env.scene["robot"]
        from active_adaptation.envs.mdp.action import JointPosition
        action_manager: JointPosition = self.env.action_manager
        self.joint_pos_offset = action_manager.offset
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.joint_mask = torch.ones(self.num_joints, device=self.device)
        if set_to_zero_joint_names is not None:
            from isaaclab.utils import resolve_matching_names
            set_to_zero_joint_ids, _ = resolve_matching_names(set_to_zero_joint_names, self.joint_names)
            self.joint_mask[set_to_zero_joint_ids] = 0.
        self.joint_mask = self.joint_mask.unsqueeze(0).unsqueeze(0) # [1, 1, J]

        shape = (self.num_envs, self.buffer_size, self.num_joints)
        self.joint_pos = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.buffer = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    def reset(self, env_ids):
        joint_pos = self.asset.data.joint_pos[env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)]
        joint_pos2 = self.asset.data.joint_pos[env_ids][:, self.joint_ids]
        assert torch.allclose(joint_pos, joint_pos2)
        self.buffer[env_ids] = self.asset.data.joint_pos[env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)].unsqueeze(1)
    
    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        joint_pos = self.joint_pos.mean(1)
        if self.noise_std > 0:
            joint_pos = random_noise(joint_pos, self.noise_std)
        self.buffer[:, 0] = joint_pos
    
    def compute(self):
        joint_pos = self.buffer - self.joint_pos_offset[:, self.joint_ids].unsqueeze(1)
        joint_pos = joint_pos * self.joint_mask
        joint_pos_selected = joint_pos[:, self.history_steps]
        return joint_pos_selected.reshape(self.num_envs, -1)
 
class prev_actions(Observation):
    def __init__(self, env, steps: int=1, flatten: bool=True, permute: bool=False):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.permute = permute
        self.action_manager = self.env.action_manager
    
    def compute(self):
        action_buf = self.action_manager.action_buf[:, :, :self.steps].clone()
        if self.permute:
            action_buf = action_buf.permute(0, 2, 1)
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def symmetry_transforms(self):
        assert self.permute
        transform = self.action_manager.symmetry_transforms()
        return transform.repeat(self.steps)


class applied_action(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = self.env.action_manager

    def compute(self) -> torch.Tensor:
        return self.action_manager.applied_action

    def symmetry_transforms(self):
        transform = self.action_manager.symmetry_transforms()
        return transform


class applied_torque(Observation):
    def __init__(self, env, joint_names: str=".*"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
    
    def compute(self) -> torch.Tensor:
        applied_efforts = self.asset.data.applied_torque
        return applied_efforts[:, self.joint_ids]
    
    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform
