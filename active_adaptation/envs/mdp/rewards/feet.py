from active_adaptation.envs.mdp.base import Reward

import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.sensors import ContactSensor
    
class feet_slip(Reward):
    def __init__(
        self, env, body_names: str, weight: float, tolerance: float = 0.0, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

        self.tolerance = tolerance

    def compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.02
        )
        feet_vel = self.asset.data.body_lin_vel_w[:, self.articulation_body_ids, :2]
        feet_vel = (feet_vel.norm(dim=-1) - self.tolerance).clamp(min=0.0, max=1.0)
        slip = (in_contact * feet_vel).sum(dim=1, keepdim=True)
        return -slip


        

class impact_force_l2(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = (
            self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        )
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing impact forces on {self.body_names}.")

    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        contact_forces = self.contact_sensor.data.net_forces_w_history.norm(
            dim=-1
        ).mean(1)
        force = contact_forces[:, self.body_ids] / self.default_mass_total
        return -(force.square() * first_contact).clamp_max(10.0).sum(1, True)


class feet_air_time(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        thres: float,
        weight: float,
        enabled: bool = True,
        soft_discount: float = 1.0,
        condition_on_linvel: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.condition_on_linvel = condition_on_linvel
        self.soft_discount = soft_discount

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.reward = torch.zeros(self.num_envs, 1, device=self.env.device)

        if self.env.backend != "isaac":
            return
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        self.vis_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Feet_contact",
                markers={"feet": sim_utils.SphereCfg(
                    radius=0.06, 
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)))}
            )
        )
        self.vis_marker_pos_w = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.env.device)

    def compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        self.reward = torch.sum(
            (last_air_time - self.thres).clamp_max(0.0) * first_contact, dim=1, keepdim=True
        )
        self.reward *= ~self.env.command_manager.is_standing_env
        violation = ((last_air_time < self.thres) & first_contact).any(dim=1)
        self.env.discount[violation] = self.soft_discount
        return self.reward
    
    def debug_draw(self):
        if self.env.backend != "isaac":
            return
        self.vis_marker_pos_w.fill_(-100)
        feet_pos_w = self.asset.data.body_pos_w[:, self.articulation_body_ids]
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        self.vis_marker_pos_w[first_contact] = feet_pos_w[first_contact]
        self.vis_marker.visualize(
            translations=self.vis_marker_pos_w.reshape(-1, 3),
        )


class feet_contact_count(Reward):
    def __init__(
        self, env, body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
        )

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)
