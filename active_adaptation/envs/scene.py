import builtins

from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


def create_isaaclab_sim_and_scene(sim_cfg: SimulationCfg, scene_cfg: InteractiveSceneCfg):
    if SimulationContext.instance() is not None:
        raise RuntimeError("Simulation context already exists. Cannot create a new one.")
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
        sim.reset()
    sim.step(render=sim.has_gui())
    return sim, scene
