import torch
import numpy as np
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple, Sequence, List

import isaaclab.utils.string as string_utils


if TYPE_CHECKING:
    from isaaclab.assets import Articulation


from active_adaptation.envs.mdp.base import Randomization

RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]


class ladder_inclination_me_randomization(Randomization):
    """
    Multi-expert ladder assignment: every env gets one expert bucket (fixed for the whole run)
    and its ladder geometry is re-sampled around that bucket at every reset.

    Buckets are the Cartesian product ``inclination_deg_list x spacing_list`` in
    inclination-major order (outer loop over inclinations, inner loop over spacings), e.g.
    ``[60, 65] x [0.22, 0.25]`` -> ``(60, 0.22), (60, 0.25), (65, 0.22), (65, 0.25)``.
    Without ``spacing_list`` the rung spacing of ``command.ladder_config`` is used for all buckets.

    Exposes, for expert routing:
        env.ladder_inclination_rad: [num_envs, 1], sampled inclination (rad, incl. noise)
        env.ladder_spacing:         [num_envs, 1], sampled rung spacing (m, incl. noise)
        env.ladder_me_experts:      list of (inclination_deg, spacing) per bucket
    Geometry sampling and pose update match ladder_geometry_randomization (per-rung x/z offsets,
    base x/z noise, additive inclination / spacing noise).
    """

    def __init__(
        self,
        inclination_deg_list: Sequence[float],
        spacing_list: Sequence[float] | None = None,
        assignment: str = "round_robin",
        rung_spacing_noise: Tuple[float, float] | None = None,
        rung_offset_x_noise: Tuple[float, float] | None = None,
        rung_offset_z_noise: Tuple[float, float] | None = None,
        inclination_noise: Tuple[float, float] | None = None,
        base_x_noise: Tuple[float, float] | None = None,
        base_z_noise: Tuple[float, float] | None = None,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if inclination_deg_list is None or len(inclination_deg_list) == 0:
            raise ValueError("ladder_inclination_me_randomization requires non-empty inclination_deg_list.")
        self.inclination_deg_list: List[float] = [float(x) for x in inclination_deg_list]

        if assignment not in ("round_robin", "random"):
            raise ValueError(f"Invalid assignment={assignment}. Expected 'round_robin' or 'random'.")
        self.assignment = assignment

        self.ladder = self.env.ladder
        self.ladder_config = self.env.cfg.command.ladder_config
        self.num_rungs = int(self.ladder_config.get("num_rungs", 6))
        self.rung_length = float(self.ladder_config.get("rung_length", 0.6))
        self.rung_width = float(self.ladder_config.get("rung_width", 0.08))
        self.rung_height = float(self.ladder_config.get("rung_height", 0.02))
        bp = self.ladder_config.get("base_position", (0.2, 0.0, 0.1))
        self.base_position = torch.tensor(
            bp if isinstance(bp, (list, tuple)) else [bp[0], bp[1], bp[2]],
            device=self.device,
            dtype=torch.float32,
        )
        self.rung_spacing = float(self.ladder_config.get("rung_spacing", 0.25))
        self.align_rungs_to_incline = bool(self.ladder_config.get("align_rungs_to_incline", False))

        if spacing_list is not None and len(spacing_list) == 0:
            raise ValueError("ladder_inclination_me_randomization: spacing_list must be non-empty (or null).")
        self.spacing_list: List[float] = [float(x) for x in spacing_list] if spacing_list is not None else [self.rung_spacing]
        self.experts = [(incl, sp) for incl in self.inclination_deg_list for sp in self.spacing_list]
        self.num_experts = len(self.experts)

        def _pair(v):
            return (float(v[0]), float(v[1])) if v is not None and len(v) >= 2 else (0.0, 0.0)

        self.rung_spacing_noise = _pair(rung_spacing_noise)
        self.rung_offset_x_noise = _pair(rung_offset_x_noise)
        self.rung_offset_z_noise = _pair(rung_offset_z_noise)
        self.inclination_noise = _pair(inclination_noise)
        self.base_x_noise = _pair(base_x_noise)
        self.base_z_noise = _pair(base_z_noise)

        # Object order must match create_ladder_cfg: rung_0 .. rung_{n-1}, rail_left, rail_right
        self.object_names = [f"rung_{i}" for i in range(self.num_rungs)] + ["rail_left", "rail_right"]
        self.num_objects = len(self.object_names)
        self.enabled = enabled

        with torch.device(self.device):
            if assignment == "round_robin":
                bucket_ids = torch.arange(self.env.num_envs, device=self.device) % self.num_experts
            else:
                bucket_ids = torch.randint(0, self.num_experts, (self.env.num_envs,), device=self.device)
            self.bucket_ids = bucket_ids.long()
            self.env.ladder_me_bucket_ids = self.bucket_ids
            self.env.ladder_me_experts = list(self.experts)
            self.env.ladder_inclination_rad = torch.zeros(self.env.num_envs, 1, device=self.device)
            self.env.ladder_spacing = torch.full((self.env.num_envs, 1), self.rung_spacing, device=self.device)
            self._expert_incl_deg = torch.tensor([e[0] for e in self.experts], dtype=torch.float32)
            self._expert_spacing = torch.tensor([e[1] for e in self.experts], dtype=torch.float32)
        print(f"[ladder_inclination_me_randomization] {self.num_experts} experts (inclination deg, rung spacing m): {self.experts}")

    def reset(self, env_ids: torch.Tensor):
        if not self.enabled:
            return
        if len(env_ids) == 0:
            return
        N = len(env_ids)
        device = self.device

        expert_ids = self.bucket_ids[env_ids]  # [N]
        inclination_deg = self._expert_incl_deg.to(device)[expert_ids]  # [N]

        inc = torch.zeros(N, device=device)
        inc.uniform_(self.inclination_noise[0], self.inclination_noise[1])
        inclination_deg = inclination_deg + inc

        angle_rad = torch.deg2rad(inclination_deg)
        self.env.ladder_inclination_rad[env_ids] = angle_rad.unsqueeze(-1)

        sp = torch.zeros(N, device=device)
        sp.uniform_(self.rung_spacing_noise[0], self.rung_spacing_noise[1])
        rung_spacing = self._expert_spacing.to(device)[expert_ids] + sp
        self.env.ladder_spacing[env_ids] = rung_spacing.unsqueeze(-1)

        ro_x = torch.zeros(N, self.num_rungs, device=device)
        ro_x.uniform_(self.rung_offset_x_noise[0], self.rung_offset_x_noise[1])
        rung_offset_x = ro_x
        ro_z = torch.zeros(N, self.num_rungs, device=device)
        ro_z.uniform_(self.rung_offset_z_noise[0], self.rung_offset_z_noise[1])
        rung_offset_z = ro_z

        base_noise = torch.zeros(N, 3, device=device)
        base_noise[:, 0].uniform_(self.base_x_noise[0], self.base_x_noise[1])
        base_noise[:, 2].uniform_(self.base_z_noise[0], self.base_z_noise[1])
        base_pos = self.base_position.unsqueeze(0).expand(N, 3) + base_noise
        base_pos = base_pos + self.env.scene.env_origins[env_ids]

        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        ladder_direction = torch.stack([cos_a, torch.zeros(N, device=device), sin_a], dim=-1)

        idx_r = torch.arange(self.num_rungs, device=device, dtype=torch.float32)
        offset = idx_r.unsqueeze(0) * rung_spacing.unsqueeze(1)
        dx = ladder_direction[:, 0:1] * offset + rung_offset_x
        dy = ladder_direction[:, 1:2] * offset
        dz = ladder_direction[:, 2:3] * offset + rung_offset_z
        rung_pos = base_pos.unsqueeze(1) + torch.stack([dx, dy, dz], dim=-1)

        rail_length = (self.num_rungs - 1) * rung_spacing if self.num_rungs > 1 else rung_spacing
        rail_center = base_pos + ladder_direction * (rail_length.unsqueeze(1) * 0.5)
        offset_dist = (self.rung_length + self.rung_height) * 0.5
        side_dir = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(N, 3)
        left_pos = rail_center - side_dir * offset_dist
        right_pos = rail_center + side_dir * offset_dist
        rail_pos = torch.stack([left_pos, right_pos], dim=1)

        positions = torch.cat([rung_pos, rail_pos], dim=1)

        obj_ids = torch.arange(self.num_objects, device=device).unsqueeze(1)
        indices = (obj_ids * self.num_envs + env_ids.unsqueeze(0)).reshape(-1)
        pos_flat = positions.transpose(0, 1).reshape(-1, 3)
        pose_flat = self.ladder.root_physx_view.get_transforms()
        pose_flat[indices, :3] = pos_flat
        self.ladder.root_physx_view.set_transforms(pose_flat, indices)


class ladder_geometry_randomization(Randomization):
    """
    Per-env ladder geometry randomization at reset: samples noise for inclination,
    rung_spacing, rung_offset, base_position and recomputes ladder poses for each env.
    Ladder collision shape sizes (rung_width, rung_height) are unchanged at runtime.
    """
    def __init__(
        self,
        rung_spacing_noise: Tuple[float, float] | None = None,
        rung_offset_x_noise: Tuple[float, float] | None = None,
        rung_offset_z_noise: Tuple[float, float] | None = None,
        inclination_noise: Tuple[float, float] | None = None,
        base_x_noise: Tuple[float, float] | None = None,
        base_z_noise: Tuple[float, float] | None = None,
        enabled: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ladder = self.env.ladder
        self.ladder_config = self.env.cfg.command.ladder_config
        self.num_rungs = int(self.ladder_config.get("num_rungs", 6))
        self.rung_length = float(self.ladder_config.get("rung_length", 0.6))
        self.rung_width = float(self.ladder_config.get("rung_width", 0.08))
        self.rung_height = float(self.ladder_config.get("rung_height", 0.02))
        bp = self.ladder_config.get("base_position", (0.2, 0.0, 0.1))
        self.base_position = torch.tensor(bp if isinstance(bp, (list, tuple)) else [bp[0], bp[1], bp[2]], device=self.device, dtype=torch.float32)
        self.rung_spacing = float(self.ladder_config.get("rung_spacing", 0.25))
        self.rung_offset = float(self.ladder_config.get("rung_offset", 0.0))
        self.inclination_deg = float(self.ladder_config.get("inclination_deg", 60.0))
        self.align_rungs_to_incline = bool(self.ladder_config.get("align_rungs_to_incline", False))

        def _pair(v):
            return (float(v[0]), float(v[1])) if v is not None and len(v) >= 2 else (0.0, 0.0)
        def _triple(v):
            if v is not None and len(v) >= 3:
                return (float(v[0]), float(v[1]), float(v[2]))
            return (0.0, 0.0, 0.0)
        self.rung_spacing_noise = _pair(rung_spacing_noise)
        self.rung_offset_x_noise = _pair(rung_offset_x_noise)
        self.rung_offset_z_noise = _pair(rung_offset_z_noise)
        self.inclination_noise = _pair(inclination_noise)
        self.base_x_noise = _pair(base_x_noise)
        self.base_z_noise = _pair(base_z_noise)

        # Object order must match create_ladder_cfg: rung_0 .. rung_{n-1}, rail_left, rail_right
        self.object_names = [f"rung_{i}" for i in range(self.num_rungs)] + ["rail_left", "rail_right"]
        self.num_objects = len(self.object_names)

        self.enabled = enabled

    def reset(self, env_ids: torch.Tensor):
        if not self.enabled:
            return
        if len(env_ids) == 0:
            return
        N = len(env_ids)
        device = self.device

        # Sample noise per env (additive)
        inc = torch.zeros(N, device=device)
        inc.uniform_(self.inclination_noise[0], self.inclination_noise[1])
        inclination_deg = self.inclination_deg + inc

        sp = torch.zeros(N, device=device)
        sp.uniform_(self.rung_spacing_noise[0], self.rung_spacing_noise[1])
        rung_spacing = self.rung_spacing + sp

        ro_x = torch.zeros(N, self.num_rungs, device=device)
        ro_x.uniform_(self.rung_offset_x_noise[0], self.rung_offset_x_noise[1])
        rung_offset_x = ro_x
        ro_z = torch.zeros(N, self.num_rungs, device=device)
        ro_z.uniform_(self.rung_offset_z_noise[0], self.rung_offset_z_noise[1])
        rung_offset_z = ro_z

        # base_z_noise: (min, max) additive noise on base z
        base_noise = torch.zeros(N, 3, device=device)
        base_noise[:, 0].uniform_(self.base_x_noise[0], self.base_x_noise[1])
        base_noise[:, 2].uniform_(self.base_z_noise[0], self.base_z_noise[1])
        base_pos = self.base_position.unsqueeze(0).expand(N, 3) + base_noise
        base_pos = base_pos + self.env.scene.env_origins[env_ids]

        angle_rad = torch.deg2rad(inclination_deg)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        ladder_direction = torch.stack([cos_a, torch.zeros(N, device=device), sin_a], dim=-1)

        # Rung positions [N, num_rungs, 3]
        idx_r = torch.arange(self.num_rungs, device=device, dtype=torch.float32)  # [num_rungs]
        offset = idx_r.unsqueeze(0) * rung_spacing.unsqueeze(1)  # [N, num_rungs]

        dx = ladder_direction[:, 0:1] * offset + rung_offset_x  # [N, num_rungs]
        dy = ladder_direction[:, 1:2] * offset  # [N, num_rungs]
        dz = ladder_direction[:, 2:3] * offset + rung_offset_z  # [N, num_rungs]
        rung_pos = base_pos.unsqueeze(1) + torch.stack([dx, dy, dz], dim=-1)  # [N, num_rungs, 3]

        rail_length = (self.num_rungs - 1) * rung_spacing if self.num_rungs > 1 else rung_spacing
        rail_center = base_pos + ladder_direction * (rail_length.unsqueeze(1) * 0.5)
        offset_dist = (self.rung_length + self.rung_height) * 0.5
        side_dir = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(N, 3)
        left_pos = rail_center - side_dir * offset_dist
        right_pos = rail_center + side_dir * offset_dist
        rail_pos = torch.stack([left_pos, right_pos], dim=1)

        positions = torch.cat([rung_pos, rail_pos], dim=1)

        obj_ids = torch.arange(self.num_objects, device=device).unsqueeze(1)
        indices = (obj_ids * self.num_envs + env_ids.unsqueeze(0)).reshape(-1)
        pos_flat = positions.transpose(0, 1).reshape(-1, 3)
        pose_flat = self.ladder.root_physx_view.get_transforms()
        pose_flat[indices, :3] = pos_flat
        self.ladder.root_physx_view.set_transforms(pose_flat, indices)


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range = (0.6, 1.0),
        dynamic_friction_range = (0.6, 1.0),
        restitution_range=(0.0, 0.2),
        homogeneous: bool=False
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.homogeneous = homogeneous
        
        self.default_materials = (
            self.asset.root_physx_view.get_material_properties()
        )
        
        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum([0,] + num_shapes_per_body)
        self.shape_ids = torch.cat([
            torch.arange(cumsum[i], cumsum[i+1]) 
            for i in self.body_ids
        ])
        self.num_buckets = 64
        self.static_friction_buckets = sample_uniform((self.num_buckets,), *self.static_friction_range)
        self.dynamic_friction_buckets = sample_uniform((self.num_buckets,), *self.dynamic_friction_range)
        self.restitution_buckets = sample_uniform((self.num_buckets,), *self.restitution_range)

    def startup(self):
        logging.info(f"Randomize body materials of {self.body_names} upon startup.")

        materials = self.default_materials.clone()
        if self.homogeneous:
            shape = (self.num_envs, 1)
        else:
            shape = (self.num_envs, len(self.shape_ids))
        materials[:, self.shape_ids, 0] = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        materials[:, self.shape_ids, 1] = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        materials[:, self.shape_ids, 2] = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]

        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_material_properties(materials.flatten(), indices)
        self.asset.data.body_materials = materials.to(self.device)


class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        self.body_ids, self.body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        self.mass_ranges = torch.tensor(values)
        print(self.body_names)

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        masses = self.asset.data.default_mass.clone()
        inertias = self.asset.data.default_inertia.clone()
        print(f"Default masses: {masses[0]}")
        scale = uniform(
            self.mass_ranges[:, 0].expand_as(masses[:, self.body_ids]),
            self.mass_ranges[:, 1].expand_as(masses[:, self.body_ids])
        )
        masses[:, self.body_ids] *= scale
        inertias[:, self.body_ids] *= scale.unsqueeze(-1)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_masses(masses, indices)
        self.asset.root_physx_view.set_inertias(inertias, indices)
        assert torch.allclose(self.asset.root_physx_view.get_masses(), masses)

class perturb_body_com(Randomization):
    def __init__(self, env, body_names, com_range=(-0.05, 0.05)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.com_range = com_range
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.ALL_INDICES = torch.arange(self.asset.num_instances)
    
    def startup(self):
        coms = self.asset.root_physx_view.get_coms()
        rand_offset = sample_uniform((self.asset.num_instances, len(self.body_ids), 3), *self.com_range)
        coms[:, self.body_ids, :3] += rand_offset
        self.asset.root_physx_view.set_coms(coms, indices=self.ALL_INDICES)


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[env_ids, :, 0], self.offset_range[env_ids, :, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)


