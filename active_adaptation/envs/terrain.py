import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter as _TerrainImporter, TerrainImporterCfg


class TerrainImporter(_TerrainImporter):
    """Minimal terrain importer: only a flat ground plane is used for ladder climbing."""

    def __init__(self, cfg: TerrainImporterCfg):
        cfg.validate()
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        self._terrain_flat_patches = dict()

        if self.cfg.terrain_type != "plane":
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' is not supported (only 'plane').")
        self.import_ground_plane("terrain")
        self.configure_env_origins()
        self.set_debug_vis(self.cfg.debug_vis)


PLANE_TERRAIN_CFG = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/ground",
    terrain_type="plane",
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
)

TERRAINS = {"plane": PLANE_TERRAIN_CFG}
