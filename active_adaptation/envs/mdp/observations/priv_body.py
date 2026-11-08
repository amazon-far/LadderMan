from active_adaptation.envs.mdp.base import Observation

import torch
import active_adaptation.utils.symmetry as sym_utils
from active_adaptation.utils.math import EMA
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation

class body_pos_b(Observation):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.update()
        if self.env.backend == "mujoco":
            self.feet_marker_0 = self.env.scene.create_sphere_marker(0.05, [1, 0, 0, 0.5])
            self.feet_marker_1 = self.env.scene.create_sphere_marker(0.05, [1, 0, 0, 0.5])

    def update(self):
        self.root_quat_w = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        self.root_pos_w = self.asset.data.root_pos_w.unsqueeze(1).clone()
        # TODO: now assume ground height is 0
        self.root_pos_w[..., 2] = 0.0
        self.body_pos_w = self.asset.data.body_pos_w[:, self.body_indices]
        
    def compute(self):
        body_pos_b = quat_apply_inverse(self.root_quat_w, self.body_pos_w - self.root_pos_w)
        return body_pos_b.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names)
    
    def debug_draw(self):
        if self.env.backend == "mujoco":
            self.feet_marker_0.geom.pos = self.asset.data.body_pos_w[0, self.body_indices[0]]
            self.feet_marker_1.geom.pos = self.asset.data.body_pos_w[0, self.body_indices[1]]


class body_vel_b(Observation):
    def __init__(self, env, body_names: str, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.update()
    
    def update(self):
        if self.yaw_only:
            self.root_quat_w = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            self.root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        self.body_vel_w = self.asset.data.body_vel_w[:, self.body_indices]
        
    def compute(self):
        body_lin_vel_b = quat_apply_inverse(self.root_quat_w, self.body_vel_w[:, :, :3])
        body_ang_vel_b = quat_apply_inverse(self.root_quat_w, self.body_vel_w[:, :, 3:])
        return body_lin_vel_b.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names)


    


   

class root_linvel_b(Observation):
    def __init__(self, env, gammas=(0.,), yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.ema = EMA(self.asset.data.root_lin_vel_w, gammas=gammas)
        self.ema.update(self.asset.data.root_lin_vel_w)
        self.update()
    
    def reset(self, env_ids: torch.Tensor):
        self.ema.reset(env_ids)
    
    def post_step(self, substep):
        self.ema.update(self.asset.data.root_lin_vel_w)
    
    def update(self):
        if self.yaw_only:
            self.quat = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
        else:
            self.quat = self.asset.data.root_quat_w.unsqueeze(1)

    def compute(self) -> torch.Tensor:
        linvel = self.ema.ema
        linvel = quat_apply_inverse(self.quat, linvel)
        return linvel.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform


class body_height(Observation):
    # this will use ray casting to compute the height of the ground under the body
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.as_tensor(self.body_ids, device=self.device)
    
    def compute(self):
        body_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
        body_height = body_pos_w[:, :, 2] - self.env.get_ground_height_at(body_pos_w)
        return body_height.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))
