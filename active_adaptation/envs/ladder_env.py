import os
import json
import math
import torch

import active_adaptation
from active_adaptation.envs.base import _Env


def create_ladder_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Ladder",
    num_rungs: int = 20,
    rung_length: float = 0.6,
    rung_width: float = 0.05,
    rung_height: float = 0.02,
    rung_offset: float = 0.0,
    rung_mass: float = 0.5,
    base_position: tuple[float, float, float] = (0.2, 0.0, 0.1),
    rung_spacing: float = 0.23,
    inclination_deg: float = 60.0,
    align_rungs_to_incline: bool = False,
    contact_idx: int = 2,
):
    """Build the ladder as a collection of kinematic cuboids.

    Object order in the collection: ``rung_0 .. rung_{num_rungs-1}`` followed by the
    left and right rails (keys ``rung_{num_rungs}`` and ``rung_{num_rungs+1}``).
    ``contact_idx`` is consumed by the tracking command, not here.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

    def quat_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    angle_rad = math.radians(inclination_deg)
    ladder_direction = (math.cos(angle_rad), 0.0, math.sin(angle_rad))

    rung_size = (rung_length, rung_width, rung_height)
    if align_rungs_to_incline:
        rung_rotation = (math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0)
    else:
        rung_rotation = (1.0, 0.0, 0.0, 0.0)
    # rotate about z so that the rungs run along the world y axis
    z_rot = (math.cos(-math.pi * 0.25), 0.0, 0.0, math.sin(-math.pi * 0.25))
    rung_rotation = quat_mul(z_rot, rung_rotation)

    def kinematic_cuboid(size, mass, pos, rot, path):
        return RigidObjectCfg(
            prim_path=path,
            spawn=sim_utils.CuboidCfg(
                size=size,
                activate_contact_sensors=False,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,  # fixed in place, but can still be moved programmatically
                    rigid_body_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.2, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )

    ladder_cfg_list = []
    for idx in range(num_rungs):
        offset = idx * rung_spacing
        rung_position = (
            base_position[0] + ladder_direction[0] * offset + idx * rung_offset,
            base_position[1] + ladder_direction[1] * offset,
            base_position[2] + ladder_direction[2] * offset,
        )
        ladder_cfg_list.append(
            kinematic_cuboid(rung_size, rung_mass, rung_position, rung_rotation, f"{prim_path}_rung_{idx}")
        )

    # two side rails along the ladder axis
    rail_length = (num_rungs - 1) * rung_spacing if num_rungs > 1 else rung_spacing
    rail_size = (rail_length, rung_height, rung_width * math.sin(angle_rad))
    # pitch by the inclination so that the local x axis follows the ladder axis, then rotate about z
    rail_rotation = (math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0)
    rail_rotation = quat_mul((math.cos(-0.5 * math.pi), 0.0, 0.0, math.sin(-0.5 * math.pi)), rail_rotation)
    rail_center = tuple(base_position[i] + ladder_direction[i] * (rail_length * 0.5) for i in range(3))
    offset_dist = (rung_length + rail_size[1]) * 0.5
    left_pos = (rail_center[0], rail_center[1] - offset_dist, rail_center[2])
    right_pos = (rail_center[0], rail_center[1] + offset_dist, rail_center[2])
    for side, rail_position in (("left", left_pos), ("right", right_pos)):
        ladder_cfg_list.append(
            kinematic_cuboid(rail_size, rung_mass * num_rungs, rail_position, rail_rotation, f"{prim_path}_rail_{side}")
        )

    return RigidObjectCollectionCfg(
        rigid_objects={f"rung_{idx}": cfg for idx, cfg in enumerate(ladder_cfg_list)}
    )


class LadderEnv(_Env):
    """G1 humanoid + procedurally generated ladder on a flat plane (Isaac Lab)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene.articulations["robot"]

        if self.sim.has_gui():
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
            from isaaclab.envs import ViewerCfg
            # IsaacLab's UI helpers expect these attributes.
            self.lookat_env_i = (
                self.scene._default_env_origins.cpu()
                - torch.tensor(self.cfg.viewer.lookat)
            ).norm(dim=-1).argmin().item()
            self.cfg.viewer.env_index = self.lookat_env_i
            self.manager_visualizers = {}
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env")
            )
            look_at_env_id = self.lookat_env_i
            self.sim.set_camera_view(
                eye=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

        self.action_buf: torch.Tensor = self.action_manager.action_buf
        self.last_action: torch.Tensor = self.action_manager.applied_action

    def setup_scene(self):
        import active_adaptation.envs.scene as scene
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from active_adaptation.assets import ROBOTS, get_asset_meta
        from active_adaptation.envs.terrain import TERRAINS

        env_spacing = self.cfg.viewer.get("env_spacing", 2.0)
        scene_cfg = InteractiveSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing, replicate_physics=False)
        scene_cfg.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        scene_cfg.robot: ArticulationCfg = ROBOTS[self.cfg.robot.name]
        scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_type = self.cfg.robot.get("robot_type", self.cfg.robot.name)
        spawn_cfg = scene_cfg.robot.spawn
        # UrdfFileCfg uses `asset_path` for the URDF file.
        spawn_cfg.asset_path = spawn_cfg.asset_path.format(ROBOT_TYPE=robot_type)

        command_self_collision = self.cfg.command.get("self_collision", None)
        if command_self_collision is not None:
            enable_self_collision = bool(command_self_collision)
            spawn_cfg.self_collision = enable_self_collision
            if spawn_cfg.articulation_props is not None:
                spawn_cfg.articulation_props.enabled_self_collisions = enable_self_collision

        scene_cfg.ladder = create_ladder_cfg(**self.cfg.command.ladder_config)

        scene_cfg.terrain = TERRAINS[self.cfg.terrain]
        scene_cfg.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|wrist_.*)_link",
            history_length=3,
            track_air_time=True
        )

        sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim.isaac_physics_dt,
            render=sim_utils.RenderCfg(rendering_mode="quality"),
            device=f"cuda:{active_adaptation.get_local_rank()}"
        )
        sim_cfg.physx.gpu_found_lost_pairs_capacity = 2538320
        sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 61999079
        sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
        sim_cfg.physx.enable_stabilization = False

        self.sim, self.scene = scene.create_isaaclab_sim_and_scene(sim_cfg, scene_cfg)
        self.ladder = self.scene.rigid_object_collections["ladder"]

        # set camera view for "/OmniverseKit_Persp" camera
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        try:
            import omni.replicator.core as rep
            self._render_product = rep.create.render_product(
                "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
            )
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        except ModuleNotFoundError:
            print("Set app.enable_cameras=true to use cameras.")

        try:
            from active_adaptation.utils.debug import DebugDraw
            self.debug_draw = DebugDraw()
            print("[INFO] Debug Draw API enabled.")
        except ModuleNotFoundError:
            pass

        # joint / actuator metadata, consumed by `scripts/play.py` when exporting a policy
        asset_meta = get_asset_meta(self.scene["robot"])
        path = os.path.join(os.getcwd(), "asset_meta.json")
        print(f"Saving asset meta to {path}")
        with open(path, "w") as f:
            json.dump(asset_meta, f, indent=4)

    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        self.stats[env_ids] = 0.
