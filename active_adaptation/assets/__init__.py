import os

from isaaclab.assets import Articulation

from .g1 import G1_CYLINDER_CFG

ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CYLINDER_CFG,
}


def get_asset_meta(asset: Articulation):
    """Joint / body ordering, default pose and actuator gains of an initialized articulation."""
    if not asset.is_initialized:
        raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
    meta = {
        "init_state": asset.cfg.init_state.to_dict(),
        "body_names_isaac": asset.body_names,
        "joint_names_isaac": asset.joint_names,
        "actuators": {},
    }
    meta["default_joint_pos"] = asset.data.default_joint_pos[0].tolist()
    meta["stiffness"] = asset.data.joint_stiffness[0].tolist()
    meta["damping"] = asset.data.joint_damping[0].tolist()
    for actuator_name, actuator in asset.actuators.items():
        meta["actuators"][actuator_name] = actuator.cfg.to_dict()
    return meta
