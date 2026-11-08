"""Warp-based depth camera observation for FAR-HDMI.

Adapted from FAR-Holosoma's WarpDepthCamera.  Uses NVIDIA Warp ray casting
against the terrain warp mesh to produce per-environment depth images at GPU
speed, without requiring RTX rendering or TiledCamera.

Supports **multiple cameras** rendered in one kernel launch via the ``cam_id``
dimension.  Configure via ``camera_poses`` dict mapping camera names to their
body-link mounting specifications.

YAML example (replaces ``depth_camera`` in the ``depth`` observation group)::

    depth:
      warp_depth_camera:
        camera_poses:
          front_depth:
            body_link: d435_link
            offset: [0.0, 0.0, 0.0]
            rotation_deg: [0.0, 0.0, 0.0]
        horizontal_fov: 87.0
        width: 48
        height: 27
        far_clip: 4.0
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from active_adaptation.envs.mdp.base import Observation
from .warp_kernels import (
    depth_range_kernel,
    depth_range_dynamic_kernel,
    depth_range_per_env_fov_kernel,
    depth_range_dynamic_per_env_fov_kernel,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

DEFAULT_EULER_FRAME_ROT_DEG = [-90.0, 0.0, -90.0]


def _convert_to_warp_mesh(
    vertices: np.ndarray, faces: np.ndarray, device: str,
) -> wp.Mesh:
    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )


def _make_cuboid_verts_faces(half_extents: List[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (8, 3) vertices and (12, 3) face indices for an origin-centred cuboid."""
    hx, hy, hz = half_extents
    verts = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return verts, faces


class warp_depth_camera(Observation):
    """Warp ray-cast depth camera observation.

    Produces depth images by casting rays from a pinhole camera model against
    the terrain warp mesh.  Multiple cameras are supported via the
    ``camera_poses`` dict and are rendered in parallel in a single GPU kernel
    launch.

    Output shape: ``(num_envs, num_cameras, height, width)``.

    Parameters
    ----------
    camera_poses : dict
        Mapping of camera name to pose specification.  Each value is a dict
        with keys ``body_link`` (str), ``offset`` (list of 3 floats),
        ``rotation_deg`` (list of 3 floats: roll, pitch, yaw in degrees).
    horizontal_fov : float
        Horizontal field of view in degrees.
    width, height : int
        Output depth image resolution.
    far_clip : float
        Maximum ray distance in metres.
    min_depth : float
        Minimum depth clamp value.
    max_depth : float or None
        Maximum depth clamp value (defaults to ``far_clip``).
    euler_frame_rot_deg : list of 3 floats or None
        Euler angles ``[roll, pitch, yaw]`` in degrees that rotate from the
        warp camera convention (z-forward) into the robot body convention
        (x-forward).  Default ``[-90, 0, -90]``.
    calculate_depth : bool
        If True (default), project ray range onto the principal axis to get
        true depth; False returns raw range.
    noise_std : float
        Per-step additive Gaussian noise standard deviation.
    episodic_noise_range : list of 2 floats
        Per-episode additive noise sampled uniformly from this range.
    delay_range : list of 2 ints
        Frame delay range ``[min, max]``.  A random integer delay (inclusive)
        is sampled per environment on reset.
    fov_randomization : dict or None
        Per-environment horizontal-FOV randomization. Keys:
        ``enable`` (bool), ``min_fov_deg`` (float), ``max_fov_deg`` (float).
    dynamic_meshes : dict or None
        Robot-body ray casting for self-occlusion.  Keys:
        ``enable`` (bool), ``body_meshes`` (dict mapping mesh_file -> body_name),
        ``meshes_root`` (str, base directory for mesh files).
    scene_meshes : dict or None
        Additional scene rigid-object collections to include in ray-casting
        (e.g. ladders, obstacles).  Keys are collection names (matching
        ``env.scene.rigid_object_collections``), values are dicts with
        ``shapes``: a dict mapping a shape name to
        ``{half_extents: [hx, hy, hz], object_indices: [0, 1, ...]}``.
    show : bool
        If True, display depth images via OpenCV windows every time
        ``compute()`` is called.  Works during both training and evaluation.
        Default is False.
    visualize_noised_depth : bool
        If True, OpenCV / ``get_visualization`` / ``debug_draw`` use the same
        depth as observations (after delay, Gaussian noise, episodic offset,
        clamp, pixel dropout, and peripheral occlusion when enabled).  If False,
        visualization uses the raw ray-cast ``depth_buffer`` only.
    pixel_dropout_prob : float
        Per-pixel dropout probability in ``[0, 1]`` applied after noise and
        episodic offset, after ``nan_to_num`` / ``clamp``.  Dropped pixels are
        set to ``pixel_dropout_fill`` (default ``max_depth``).  ``0`` disables.
    pixel_dropout_fill : float or None
        Depth value written to dropped pixels.  If None, uses ``max_depth``
        (simulates missing returns / max range).
    pose_randomization : dict or None
        Per-episode randomization of each camera's mount ``offset`` (metres)
        and ``rotation_deg`` (roll, pitch, yaw in degrees) in the body link
        frame, on top of the nominal values in ``camera_poses``.  Keys:
        ``enable`` (bool), ``offset_range`` (list of 3 ``[low, high]`` pairs
        for x, y, z), ``rotation_deg_range`` (3 pairs for roll, pitch, yaw).
        Omitted ranges default to ``[0, 0]`` per axis.  Resampled on each
        ``reset`` (subset of envs).
    peripheral_depth_occlusion : dict or None
        Stochastic **lateral** masking: each step (when active) samples an
        integer column width and replaces the left and right peripheral bands
        with **uniform random depth** in ``[d_near, d_far]`` (independent per
        pixel), to discourage reliance on side context.  Keys: ``enable``
        (bool), ``mask_width_range`` ``[min_cols, max_cols]`` inclusive per
        side (clamped so strips do not overlap), ``depth_range``
        ``[d_near, d_far]`` (defaults to ``min_depth`` / ``max_depth``),
        ``apply_probability`` (float in ``[0, 1]``, per-env Bernoulli whether
        to apply this step).
    """

    def __init__(
        self,
        env: Any,
        camera_poses: Dict[str, Any],
        horizontal_fov: float = 89.3,
        width: int = 640,
        height: int = 480,
        target_width: int | None = None,
        target_height: int | None = None,
        far_clip: float = 2.0,
        min_depth: float = 0.1,
        max_depth: float | None = None,
        euler_frame_rot_deg: List[float] | None = None,
        calculate_depth: bool = True,
        noise_std: float = 0.003,
        additional_noise_std: float = 0.002,
        episodic_noise_range: List[float] | Tuple[float, float] = (-0.0, 0.0),
        delay_range: List[int] | Tuple[int, int] = (0, 0),
        fov_randomization: Dict[str, Any] | None = None,
        pose_randomization: Dict[str, Any] | None = None,
        peripheral_depth_occlusion: Dict[str, Any] | None = None,
        dynamic_meshes: Dict[str, Any] | None = None,
        scene_meshes: Dict[str, Any] | None = None,
        show: bool = False,
        visualize_noised_depth: bool = False,
        pixel_dropout_prob: float = 0.05,
        pixel_dropout_fill: float | None = None,
    ):
        super().__init__(env)

        if euler_frame_rot_deg is None:
            euler_frame_rot_deg = list(DEFAULT_EULER_FRAME_ROT_DEG)
        if max_depth is None:
            max_depth = far_clip

        # -- parse camera poses (handle both dict and OmegaConf DictConfig) --
        camera_poses = dict(camera_poses)
        self.camera_names: list[str] = list(camera_poses.keys())
        self._num_sensors: int = len(self.camera_names)
        self._poses_cfg = {k: dict(v) for k, v in camera_poses.items()}

        # -- camera params --
        self.horizontal_fov = horizontal_fov
        self.ray_width = width
        self.ray_height = height
        self.target_width = target_width if target_width is not None else width
        self.target_height = target_height if target_height is not None else height
        ih, iw = self.ray_height, self.ray_width
        oh, ow = self.target_height, self.target_width
        if oh != ih or ow != iw:
            dev = self.device
            self._depth_nn_y_idx = (
                torch.arange(oh, device=dev, dtype=torch.long) * ih // oh
            ).clamp(0, ih - 1)
            self._depth_nn_x_idx = (
                torch.arange(ow, device=dev, dtype=torch.long) * iw // ow
            ).clamp(0, iw - 1)
        else:
            self._depth_nn_y_idx = None
            self._depth_nn_x_idx = None

        self.far_plane = far_clip
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.calculate_depth = calculate_depth
        self.euler_frame_rot_deg = euler_frame_rot_deg
        self._show = show
        self._visualize_noised_depth = visualize_noised_depth
        self._last_obs_depth: torch.Tensor | None = None

        # -- noise / DR --
        self.noise_std = noise_std
        self.additional_noise_std = additional_noise_std
        self.episodic_noise_range = tuple(episodic_noise_range)
        self.delay_range = tuple(delay_range)
        self.pixel_dropout_prob = float(pixel_dropout_prob)
        self.pixel_dropout_fill = pixel_dropout_fill

        # -- FOV randomization --
        self.fov_cfg = dict(fov_randomization) if fov_randomization else {"enable": False}
        self._randomize_fov: bool = self.fov_cfg.get("enable", False)

        # -- camera pose randomization (offset + rotation_deg in link frame) --
        self.pose_rand_cfg = dict(pose_randomization) if pose_randomization else {"enable": False}
        self._randomize_pose: bool = self.pose_rand_cfg.get("enable", False)

        # -- peripheral depth occlusion (lateral bands + random depth fill) --
        poc = dict(peripheral_depth_occlusion) if peripheral_depth_occlusion else {"enable": False}
        self._periph_occ_enable = bool(poc.get("enable", False))
        mw = poc.get("mask_width_range", [0, 0])
        w0, w1 = int(mw[0]), int(mw[1])
        if w0 > w1:
            w0, w1 = w1, w0
        self._periph_mask_w_min = w0
        self._periph_mask_w_max = w1
        dr = poc.get("depth_range", None)
        if dr is None:
            self._periph_d_near = float(min_depth)
            self._periph_d_far = float(max_depth)
        else:
            self._periph_d_near = float(dr[0])
            self._periph_d_far = float(dr[1])
        if self._periph_d_near > self._periph_d_far:
            self._periph_d_near, self._periph_d_far = self._periph_d_far, self._periph_d_near
        self._periph_apply_prob = min(1.0, max(0.0, float(poc.get("apply_probability", 1.0))))

        # -- dynamic meshes --
        self.dyna_cfg = dict(dynamic_meshes) if dynamic_meshes else {"enable": False}
        self._use_dynamic_meshes: bool = self.dyna_cfg.get("enable", False)

        # -- scene meshes (static objects like ladders) --
        self._scene_meshes_cfg: Dict[str, Any] = {}
        self._shape_noise_cfg: Dict[str, Dict[str, Any]] = {}
        if scene_meshes is not None:
            self._scene_meshes_cfg = {k: dict(v) for k, v in scene_meshes.items()}
            for coll_cfg in self._scene_meshes_cfg.values():
                sn = dict(coll_cfg).get("shape_noise")
                if sn:
                    for shape_name, noise_cfg in sn.items():
                        self._shape_noise_cfg[shape_name] = dict(noise_cfg)

        # -- internal state --
        self._graph: wp.Graph | None = None
        self._num_dyna_bodies: int = 0
        self._scene_coll_updates: list[tuple] = []
        self.asset: Articulation = self.env.scene["robot"]

        self.hand_offsets = torch.tensor([[0.0415, 0.003, 0.0], [0.0415, -0.003, 0.0]], device=self.device, dtype=torch.float32)

        self._init_camera_matrices()
        self._init_tensors()

    # ------------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------------

    def _init_camera_matrices(self) -> None:
        W, H = float(self.ray_width), float(self.ray_height)
        self._c_x = int(W / 2.0)
        self._c_y = int(H / 2.0)

        if self._randomize_fov:
            self._K_inv = None
            self._init_per_env_fov()
        else:
            self._K_inv_array = None
            self._wp_K_inv_array = None

            u_0, v_0 = W / 2.0, H / 2.0
            hfov_rad = math.radians(self.horizontal_fov)
            f = W / 2.0 / math.tan(hfov_rad / 2.0)
            vfov = 2.0 * math.atan(H / (2.0 * f))
            alpha_u = u_0 / math.tan(hfov_rad / 2.0)
            alpha_v = v_0 / math.tan(vfov / 2.0)

            K = wp.mat44(
                alpha_u, 0.0, u_0, 0.0,
                0.0, alpha_v, v_0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            )
            self._K_inv = wp.inverse(K)

    # ------------------------------------------------------------------
    # Per-environment FOV randomisation
    # ------------------------------------------------------------------

    def _init_per_env_fov(self) -> None:
        N, S = self.num_envs, self._num_sensors
        dev = self.device

        min_deg = self.fov_cfg.get("min_fov_deg", self.horizontal_fov - 2.0)
        max_deg = self.fov_cfg.get("max_fov_deg", self.horizontal_fov + 2.0)

        self._min_fov_rad = torch.full((N, S), math.radians(min_deg), device=dev)
        self._max_fov_rad = torch.full((N, S), math.radians(max_deg), device=dev)
        self._fov_rad = torch.zeros(N, S, device=dev)

        self._resample_fov()

    def _resample_fov(self, env_ids: torch.Tensor | None = None) -> None:
        if not self._randomize_fov:
            return

        if env_ids is None:
            self._fov_rad[:] = (
                (self._max_fov_rad - self._min_fov_rad) * torch.rand_like(self._fov_rad)
                + self._min_fov_rad
            )
        else:
            n = env_ids.shape[0]
            self._fov_rad[env_ids] = (
                (self._max_fov_rad[env_ids] - self._min_fov_rad[env_ids])
                * torch.rand(n, self._num_sensors, device=self.device)
                + self._min_fov_rad[env_ids]
            )

        self._rebuild_K_inv_array()
        self._graph = None

    def _rebuild_K_inv_array(self) -> None:
        W, H = float(self.ray_width), float(self.ray_height)
        u_0, v_0 = W / 2.0, H / 2.0

        fov_half = self._fov_rad / 2.0
        tan_half = torch.tan(fov_half)
        alpha_u = u_0 / tan_half
        f = (W / 2.0) / tan_half
        vfov = 2.0 * torch.atan(H / (2.0 * f))
        alpha_v = v_0 / torch.tan(vfov / 2.0)

        N, S = self._fov_rad.shape
        K = torch.zeros(N, S, 4, 4, device=self.device, dtype=torch.float32)
        K[..., 0, 0] = alpha_u
        K[..., 0, 2] = u_0
        K[..., 1, 1] = alpha_v
        K[..., 1, 2] = v_0
        K[..., 2, 2] = 1.0
        K[..., 3, 3] = 1.0

        K_inv = torch.linalg.inv(K)  # (N, S, 4, 4)
        self._K_inv_array = K_inv.reshape(N, S, 16).contiguous()
        self._wp_K_inv_array = wp.from_torch(self._K_inv_array, dtype=wp.float32)

    # ------------------------------------------------------------------
    # Camera pose randomisation (mount offset + rotation in link frame)
    # ------------------------------------------------------------------

    def _resample_pose(self, env_ids: torch.Tensor | None = None) -> None:
        """Resample per-env, per-camera additive offset (m) and rotation (deg)."""
        if not self._randomize_pose:
            return

        from isaaclab.utils.math import quat_from_euler_xyz

        N = self.num_envs
        dev = self.device
        if env_ids is None:
            env_ids = torch.arange(N, device=dev, dtype=torch.long)
        n = int(env_ids.shape[0])
        S = self._num_sensors

        olow = self._pose_offset_low.view(1, 1, 3)
        ohigh = self._pose_offset_high.view(1, 1, 3)
        pos_delta = olow + torch.rand(n, S, 3, device=dev, dtype=torch.float32) * (
            ohigh - olow
        )

        rlow = self._pose_rot_deg_low.view(1, 1, 3)
        rhigh = self._pose_rot_deg_high.view(1, 1, 3)
        rot_delta_deg = rlow + torch.rand(n, S, 3, device=dev, dtype=torch.float32) * (
            rhigh - rlow
        )
        rot_delta_rad = torch.deg2rad(rot_delta_deg)

        for s in range(S):
            nom_p = self._nominal_local_pos[s]
            self._local_pos_list[s][env_ids] = nom_p.expand(n, -1) + pos_delta[:, s, :]

            nom_r = self._nominal_rot_rad[s]
            eff_rad = nom_r.expand(n, -1) + rot_delta_rad[:, s, :]
            lq = quat_from_euler_xyz(eff_rad[:, 0], eff_rad[:, 1], eff_rad[:, 2])
            self._local_quat_list[s][env_ids] = lq

    # ------------------------------------------------------------------
    # Tensor / buffer initialisation
    # ------------------------------------------------------------------

    def _init_tensors(self) -> None:
        from isaaclab.utils.math import quat_from_euler_xyz

        N = self.num_envs
        S = self._num_sensors
        dev = self.device

        # Depth output buffer: (N, S, H, W)
        self.depth_buffer = torch.zeros(
            N, S, self.ray_height, self.ray_width,
            device=dev, dtype=torch.float32,
        )

        # Segmentation output buffer: (N, S, H, W)
        # 0=terrain/no-hit, 1+=baked shape labels, -1=robot body
        self.seg_buffer = torch.zeros(
            N, S, self.ray_height, self.ray_width,
            device=dev, dtype=torch.int32,
        )
        self._wp_seg = wp.from_torch(self.seg_buffer, dtype=wp.int32)

        # Frame-convention quaternion (camera z-fwd -> robot x-fwd)
        # Stored in Isaac Lab convention (w, x, y, z)
        frame_deg = torch.tensor(self.euler_frame_rot_deg, device=dev, dtype=torch.float32)
        frame_rad = torch.deg2rad(frame_deg)
        frame_quat = quat_from_euler_xyz(
            frame_rad[0:1], frame_rad[1:2], frame_rad[2:3]
        )  # (1, 4) in (w, x, y, z)
        self._frame_quat = frame_quat.expand(N, -1).contiguous()

        # Per-camera local poses relative to their body links
        # Quaternions stored in Isaac Lab convention (w, x, y, z)
        self._local_pos_list: list[torch.Tensor] = []
        self._local_quat_list: list[torch.Tensor] = []
        self._body_link_names: list[str] = []
        self._nominal_local_pos: list[torch.Tensor] = []
        self._nominal_rot_rad: list[torch.Tensor] = []

        if self._randomize_pose:
            orng = self.pose_rand_cfg.get(
                "offset_range", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            )
            rrng = self.pose_rand_cfg.get(
                "rotation_deg_range", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            )
            self._pose_offset_low = torch.tensor(
                [float(a[0]) for a in orng], device=dev, dtype=torch.float32,
            )
            self._pose_offset_high = torch.tensor(
                [float(a[1]) for a in orng], device=dev, dtype=torch.float32,
            )
            self._pose_rot_deg_low = torch.tensor(
                [float(a[0]) for a in rrng], device=dev, dtype=torch.float32,
            )
            self._pose_rot_deg_high = torch.tensor(
                [float(a[1]) for a in rrng], device=dev, dtype=torch.float32,
            )

        for name, pose_cfg in self._poses_cfg.items():
            offset = pose_cfg.get("offset", [0.0, 0.0, 0.0])
            rotation_deg = pose_cfg.get("rotation_deg", [0.0, 0.0, 0.0])
            body_link = pose_cfg["body_link"]

            self._body_link_names.append(body_link)

            nom_p = torch.tensor(offset, device=dev, dtype=torch.float32).unsqueeze(0)
            rot_deg = torch.tensor(rotation_deg, device=dev, dtype=torch.float32)
            nom_r = torch.deg2rad(rot_deg).unsqueeze(0)
            if self._randomize_pose:
                self._nominal_local_pos.append(nom_p)
                self._nominal_rot_rad.append(nom_r)

            if self._randomize_pose:
                self._local_pos_list.append(torch.zeros(N, 3, device=dev, dtype=torch.float32))
                lq_ident = torch.zeros(N, 4, device=dev, dtype=torch.float32)
                lq_ident[:, 0] = 1.0
                self._local_quat_list.append(lq_ident)
            else:
                lp = nom_p.expand(N, -1).contiguous()
                self._local_pos_list.append(lp)
                lq = quat_from_euler_xyz(
                    nom_r[0, 0:1], nom_r[0, 1:2], nom_r[0, 2:3],
                )
                self._local_quat_list.append(lq.expand(N, -1).contiguous())

        if self._randomize_pose:
            self._resample_pose(None)

        # Global camera poses for the warp kernel.
        # Positions: standard (x, y, z).
        # Quaternions: stored in warp convention (x, y, z, w).
        self._global_pos = torch.zeros(N, S, 3, device=dev, dtype=torch.float32)
        self._global_quat = torch.zeros(N, S, 4, device=dev, dtype=torch.float32)
        self._global_quat[..., 3] = 1.0  # identity in warp (x,y,z,w)

        # Zero-copy warp views over the torch tensors
        self._wp_pos = wp.from_torch(self._global_pos, dtype=wp.vec3)
        self._wp_quat = wp.from_torch(self._global_quat, dtype=wp.quat)
        self._wp_pixels = wp.from_torch(self.depth_buffer, dtype=wp.float32)

        # Frame-delay buffer
        max_delay = max(self.delay_range[1], 0)
        self._has_delay = max_delay > 0
        if self._has_delay:
            self._delay_buffer = torch.zeros(
                N, max_delay + 1, S, self.ray_height, self.ray_width,
                device=dev, dtype=torch.float32,
            )
            self._delay = torch.zeros(N, dtype=torch.long, device=dev)
            self._all_env_ids = torch.arange(N, dtype=torch.long, device=dev)

        # Episodic noise
        self._episodic_noise = torch.zeros(N, dtype=torch.float32, device=dev)

        # Per-shape episodic depth offsets (populated for shapes with episodic_offset_range)
        self._shape_episodic_offset: Dict[str, torch.Tensor] = {}
        for shape_name, noise_cfg in self._shape_noise_cfg.items():
            noise_cfg = dict(noise_cfg)
            if noise_cfg.get("episodic_offset_range"):
                self._shape_episodic_offset[shape_name] = torch.zeros(
                    N, dtype=torch.float32, device=dev,
                )

    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """One-time setup: acquire terrain mesh, create dynamic scene meshes, and resolve body indices."""
        self._terrain_wp_mesh = self.env.ground_mesh
        self._terrain_mesh_id = self._terrain_wp_mesh.id

        # Terrain-only face labels (all zeros; scene objects are now dynamic)
        num_terrain_faces = self._terrain_wp_mesh.indices.shape[0] // 3
        self._wp_face_labels = wp.zeros(num_terrain_faces, dtype=wp.int32, device=str(self.device))

        # Build shape label mapping for segmentation
        self.shape_labels: dict[str, int] = {}
        next_label = 1
        for coll_cfg in self._scene_meshes_cfg.values():
            for shape_name in dict(coll_cfg).get("shapes", {}):
                if shape_name not in self.shape_labels:
                    self.shape_labels[shape_name] = next_label
                    next_label += 1

        self._body_indices: list[int] = []
        for body_link_name in self._body_link_names:
            body_ids, body_names = self.asset.find_bodies(body_link_name)
            assert len(body_ids) > 0, (
                f"warp_depth_camera: body link '{body_link_name}' not found on robot"
            )
            self._body_indices.append(body_ids[0])

        # --- Dynamic meshes: robot bodies ---
        all_wp_meshes: list[wp.Mesh] = []
        all_mesh_labels: list[int] = []
        self._num_robot_bodies = 0

        if self._use_dynamic_meshes:
            robot_meshes, robot_body_indices = self._load_robot_meshes()
            all_wp_meshes.extend(robot_meshes)
            all_mesh_labels.extend([-1] * len(robot_meshes))
            self._num_robot_bodies = len(robot_meshes)
            self._dyna_body_idx_tensor = torch.tensor(
                robot_body_indices, dtype=torch.long, device=self.device,
            )
        else:
            self._dyna_body_idx_tensor = torch.zeros(0, dtype=torch.long, device=self.device)

        # --- Dynamic meshes: scene objects (e.g. ladder rungs/rails) ---
        self._scene_coll_updates: list[tuple] = []
        self._num_scene_objects = 0

        if self._scene_meshes_cfg:
            wp_device = str(self.device)
            for coll_name, coll_cfg in self._scene_meshes_cfg.items():
                coll_cfg = dict(coll_cfg)
                collection = self.env.scene.rigid_object_collections.get(coll_name, None)
                assert collection is not None, (
                    f"warp_depth_camera: scene collection '{coll_name}' not found"
                )
                shapes = dict(coll_cfg.get("shapes", {}))

                shape_wp_meshes: dict[str, wp.Mesh] = {}
                for shape_name, shape_cfg in shapes.items():
                    shape_cfg = dict(shape_cfg)
                    verts, faces = _make_cuboid_verts_faces(list(shape_cfg["half_extents"]))
                    shape_wp_meshes[shape_name] = _convert_to_warp_mesh(verts, faces, wp_device)

                start = len(all_wp_meshes)
                obj_indices_list: list[int] = []
                for shape_name, shape_cfg in shapes.items():
                    shape_cfg = dict(shape_cfg)
                    label = self.shape_labels[shape_name]
                    wp_mesh = shape_wp_meshes[shape_name]
                    for obj_idx in list(shape_cfg["object_indices"]):
                        all_wp_meshes.append(wp_mesh)
                        all_mesh_labels.append(label)
                        obj_indices_list.append(obj_idx)

                count = len(obj_indices_list)
                obj_idx_tensor = torch.tensor(
                    obj_indices_list, dtype=torch.long, device=self.device,
                )
                self._scene_coll_updates.append((collection, obj_idx_tensor, start, count))

            self._num_scene_objects = len(all_wp_meshes) - self._num_robot_bodies

        # --- Combine all dynamic meshes ---
        self._num_dyna_bodies = len(all_wp_meshes)
        if self._num_dyna_bodies > 0:
            self._wp_robot_mesh_ids = wp.array(
                [m.id for m in all_wp_meshes], dtype=wp.uint64, device=str(self.device),
            )
            self._wp_dyna_mesh_labels = wp.array(
                all_mesh_labels, dtype=wp.int32, device=str(self.device),
            )
            N = self.num_envs
            B = self._num_dyna_bodies
            self._dyna_body_pos = torch.zeros(N, B, 3, device=self.device, dtype=torch.float32)
            self._dyna_body_quat = torch.zeros(N, B, 4, device=self.device, dtype=torch.float32)
            self._dyna_body_quat[..., 3] = 1.0
            self._wp_dyna_body_pos = wp.from_torch(self._dyna_body_pos, dtype=wp.vec3)
            self._wp_dyna_body_quat = wp.from_torch(self._dyna_body_quat, dtype=wp.quat)

        cam_details = ", ".join(
            f"{name}@'{link}'(idx={idx})"
            for name, link, idx in zip(
                self.camera_names, self._body_link_names, self._body_indices
            )
        )
        dyna_info = ""
        if self._num_robot_bodies > 0:
            dyna_info += f", robot_bodies={self._num_robot_bodies}"
        if self._num_scene_objects > 0:
            dyna_info += f", scene_objects={self._num_scene_objects}(dynamic)"
        if self.shape_labels:
            dyna_info += f", labels={self.shape_labels}"
        pose_info = ", pose_randomization=on" if self._randomize_pose else ""
        print(
            f"warp_depth_camera startup: cameras=[{cam_details}], "
            f"resolution={self.ray_width}x{self.ray_height}, "
            f"far={self.far_plane}m, fov={self.horizontal_fov}°{dyna_info}{pose_info}"
        )

    def _load_robot_meshes(self) -> tuple[list[wp.Mesh], list[int]]:
        """Load robot body meshes and resolve their rigid-body indices."""
        import trimesh
        import active_adaptation

        body_meshes: dict[str, str] = self.dyna_cfg.get("body_meshes", {})
        meshes_root: str = self.dyna_cfg.get("meshes_root", "")

        if not os.path.isabs(meshes_root):
            project_root = os.path.dirname(os.path.dirname(active_adaptation.ASSET_PATH))
            meshes_root = os.path.join(project_root, meshes_root)

        wp_meshes: list[wp.Mesh] = []
        body_indices: list[int] = []
        wp_device = str(self.device)

        for mesh_file, body_name in body_meshes.items():
            mesh_path = os.path.join(meshes_root, mesh_file)
            mesh = trimesh.load(mesh_path)
            wp_meshes.append(_convert_to_warp_mesh(mesh.vertices, mesh.faces, wp_device))
            body_ids, _ = self.asset.find_bodies(body_name)
            assert len(body_ids) > 0, (
                f"warp_depth_camera: dynamic mesh body '{body_name}' not found"
            )
            body_indices.append(body_ids[0])

        return wp_meshes, body_indices

    def update(self) -> None:
        """Update global camera poses from the current rigid-body state."""
        from isaaclab.utils.math import quat_mul, quat_apply

        for s, body_idx in enumerate(self._body_indices):
            # Isaac Lab: body state in (w, x, y, z) quaternion convention
            body_pos = self.asset.data.body_pos_w[:, body_idx, :]   # (N, 3)
            body_quat = self.asset.data.body_quat_w[:, body_idx, :]  # (N, 4) wxyz

            cam_pos = quat_apply(body_quat, self._local_pos_list[s]) + body_pos
            combined_local = quat_mul(self._local_quat_list[s], self._frame_quat)
            cam_quat = quat_mul(body_quat, combined_local)  # (N, 4) wxyz

            self._global_pos[:, s, :] = cam_pos
            # Convert Isaac Lab (w,x,y,z) -> warp (x,y,z,w)
            self._global_quat[:, s, :3] = cam_quat[:, 1:]
            self._global_quat[:, s, 3] = cam_quat[:, 0]

        if self._use_dynamic_meshes and self._num_robot_bodies > 0:
            body_pos_w = self.asset.data.body_pos_w
            body_quat_w = self.asset.data.body_quat_w
            dyna_pos = body_pos_w[:, self._dyna_body_idx_tensor, :]
            dyna_quat = body_quat_w[:, self._dyna_body_idx_tensor, :]
            # TODO: important! hand mesh are collapsed into the wrist mesh, so we need to offset the hand mesh by the hand offset
            dyna_pos[:, -2:] = quat_apply(
                dyna_quat[:, -2:],
                self.hand_offsets.unsqueeze(0).expand(dyna_quat.shape[0], -1, -1),
            ) + dyna_pos[:, -2:]
            R = self._num_robot_bodies
            self._dyna_body_pos[:, :R] = dyna_pos
            self._dyna_body_quat[:, :R, :3] = dyna_quat[..., 1:]
            self._dyna_body_quat[:, :R, 3] = dyna_quat[..., 0]

        for collection, obj_idx_tensor, start, count in self._scene_coll_updates:
            pos_w = collection.data.object_pos_w[:, obj_idx_tensor]
            quat_w = collection.data.object_quat_w[:, obj_idx_tensor]
            end = start + count
            self._dyna_body_pos[:, start:end] = pos_w
            # Convert Isaac Lab (w,x,y,z) -> warp (x,y,z,w)
            self._dyna_body_quat[:, start:end, :3] = quat_w[..., 1:]
            self._dyna_body_quat[:, start:end, 3] = quat_w[..., 0]

    def reset(self, env_ids: torch.Tensor) -> None:
        """Resample per-episode DR parameters."""
        n = len(env_ids)
        self._episodic_noise[env_ids] = torch.empty(
            n, device=self.device,
        ).uniform_(*self.episodic_noise_range)

        if self._has_delay:
            self._delay[env_ids] = torch.randint(
                low=self.delay_range[0],
                high=self.delay_range[1] + 1,
                size=(n,),
                dtype=torch.long,
                device=self.device,
            )

        if self._randomize_fov:
            self._resample_fov(env_ids)

        if self._randomize_pose:
            self._resample_pose(env_ids)

        for shape_name, offset_tensor in self._shape_episodic_offset.items():
            noise_cfg = dict(self._shape_noise_cfg[shape_name])
            lo, hi = noise_cfg["episodic_offset_range"]
            offset_tensor[env_ids] = torch.empty(
                n, device=self.device,
            ).uniform_(lo, hi)

    def compute(self) -> torch.Tensor:
        """Ray-cast depth images and return the buffer.

        Returns
        -------
        torch.Tensor
            Depth buffer of shape ``(num_envs, num_cameras, height, width)``.
        """
        if self._graph is None:
            self._build_render_graph()

        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._launch_kernel()

        depth = self.depth_buffer

        # apply specific shape noise (here we do rails)
        if self._shape_noise_cfg and self.shape_labels:
            depth = depth.clone()
            seg = self.seg_buffer
            for shape_name, noise_cfg in self._shape_noise_cfg.items():
                label = self.shape_labels.get(shape_name)
                if label is None:
                    continue
                noise_cfg = dict(noise_cfg)
                mask = seg == label
                if noise_cfg.get("reverse_mask", False):
                    # ~(shape | robot): noise everywhere except labeled shape and dynamic robot (-1)
                    mask = ~(mask | seg.eq(-1))
                sn_std = noise_cfg.get("noise_std", 0.0)
                if noise_cfg.get("replace_noise", False):
                    uniform_depth = torch.empty_like(depth).uniform_(
                        self.min_depth, self.max_depth,
                    )
                    depth = torch.where(mask, uniform_depth, depth)
                else:
                    if sn_std > 0:
                        depth = torch.where(mask, depth + torch.randn_like(depth) * sn_std, depth)
                    if shape_name in self._shape_episodic_offset:
                        offset = self._shape_episodic_offset[shape_name]
                        depth = torch.where(mask, depth + offset.view(-1, 1, 1, 1), depth)

        if self._has_delay:
            self._delay_buffer = self._delay_buffer.roll(1, dims=1)
            self._delay_buffer[:, 0] = depth
            depth = self._delay_buffer[self._all_env_ids, self._delay]

        if self.noise_std > 0 or self.additional_noise_std > 0:
            sigma = torch.full_like(depth, self.noise_std)
            if self.additional_noise_std > 0:
                sigma = sigma + self.additional_noise_std * depth.abs()
            depth = depth + torch.randn_like(depth) * sigma

        depth = depth + self._episodic_noise.view(-1, 1, 1, 1)

        depth = depth.nan_to_num(nan=self.max_depth, posinf=self.max_depth, neginf=self.min_depth)
        depth = depth.clamp(min=self.min_depth, max=self.max_depth)

        if self.pixel_dropout_prob > 0.0:
            fill = self.pixel_dropout_fill
            if fill is None:
                fill = self.max_depth
            drop = torch.rand_like(depth) < self.pixel_dropout_prob
            depth = torch.where(drop, torch.full_like(depth, fill), depth)

        if self._should_apply_peripheral_occlusion():
            depth = self._apply_peripheral_depth_occlusion(depth)

        if self._depth_nn_y_idx is not None:
            depth = depth[..., self._depth_nn_y_idx[:, None], self._depth_nn_x_idx[None, :]]

        self._last_obs_depth = depth.detach()

        if self._show:
            vis = depth if self._visualize_noised_depth else None
            self._show_depth(vis)

        return depth  # (N, S, target_height, target_width)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_apply_peripheral_occlusion(self) -> bool:
        if not self._periph_occ_enable:
            return False
        return True

    def _apply_peripheral_depth_occlusion(self, depth: torch.Tensor) -> torch.Tensor:
        """Mask left/right column bands with i.i.d. uniform depth in [d_near, d_far]."""
        if self._periph_mask_w_max <= 0:
            return depth

        N, S, H, W = depth.shape
        dev = depth.device
        m = torch.randint(
            low=self._periph_mask_w_min,
            high=self._periph_mask_w_max + 1,
            size=(N,),
            device=dev,
            dtype=torch.long,
        )
        m = m.clamp(min=0, max=W // 2)

        cols = torch.arange(W, device=dev, dtype=torch.long)
        left = cols.view(1, W) < m.view(N, 1)
        right = cols.view(1, W) >= (W - m.view(N, 1))
        col_occ = left | right
        occ = col_occ.view(N, 1, 1, W).expand(N, S, H, W)

        if self._periph_apply_prob < 1.0:
            use = torch.rand(N, device=dev) < self._periph_apply_prob
            occ = occ & use.view(N, 1, 1, 1)

        fill = torch.empty_like(depth)
        lo, hi = self._periph_d_near, self._periph_d_far
        if hi <= lo:
            fill.fill_(lo)
        else:
            fill.uniform_(lo, hi)
        return torch.where(occ, fill, depth)

    def _launch_kernel(self) -> None:
        dims = (self.num_envs, self._num_sensors, self.ray_width, self.ray_height)
        fov_per_env = self._randomize_fov and self._wp_K_inv_array is not None
        dyna = self._num_dyna_bodies > 0

        if dyna and fov_per_env:
            wp.launch(
                kernel=depth_range_dynamic_per_env_fov_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_robot_mesh_ids,
                    self._wp_dyna_body_pos,
                    self._wp_dyna_body_quat,
                    self._wp_pos,
                    self._wp_quat,
                    self._wp_K_inv_array,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                    self._num_dyna_bodies,
                    self._wp_face_labels,
                    self._wp_seg,
                    self._wp_dyna_mesh_labels,
                ],
                device=str(self.device),
            )
        elif dyna:
            wp.launch(
                kernel=depth_range_dynamic_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_robot_mesh_ids,
                    self._wp_dyna_body_pos,
                    self._wp_dyna_body_quat,
                    self._wp_pos,
                    self._wp_quat,
                    self._K_inv,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                    self._num_dyna_bodies,
                    self._wp_face_labels,
                    self._wp_seg,
                    self._wp_dyna_mesh_labels,
                ],
                device=str(self.device),
            )
        elif fov_per_env:
            wp.launch(
                kernel=depth_range_per_env_fov_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_pos,
                    self._wp_quat,
                    self._wp_K_inv_array,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                    self._wp_face_labels,
                    self._wp_seg,
                ],
                device=str(self.device),
            )
        else:
            wp.launch(
                kernel=depth_range_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_pos,
                    self._wp_quat,
                    self._K_inv,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                    self._wp_face_labels,
                    self._wp_seg,
                ],
                device=str(self.device),
            )

    def _build_render_graph(self) -> None:
        try:
            wp.capture_begin(device=str(self.device))
            self._launch_kernel()
            self._graph = wp.capture_end(device=str(self.device))
        except Exception:
            self._graph = None

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _show_depth(self, depth_override: torch.Tensor | None = None) -> None:
        """Pop up OpenCV windows for env 0 depth (called from ``compute`` when ``show=True``).

        Parameters
        ----------
        depth_override
            If set, visualize this tensor ``(N, S, H, W)`` (e.g. post-noise
            observation).  Otherwise use raw ``depth_buffer``.
        """
        try:
            import cv2
        except ImportError:
            return

        buf = depth_override if depth_override is not None else self.depth_buffer
        if buf.is_cuda:
            torch.cuda.synchronize()

        for s, cam_name in enumerate(self.camera_names):
            frame = buf[0, s].detach().cpu().numpy()
            frame_norm = np.clip(frame / max(self.far_plane, 1e-3), 0.0, 1.0)
            frame_u8 = (frame_norm * 255).astype(np.uint8)
            cv2.imshow(f"warp_depth/{cam_name}", frame_u8)
        cv2.waitKey(1)

    def debug_draw(self) -> None:
        """Show depth images from env 0 in OpenCV windows (one per camera).

        Called by the environment when the GUI is active.  Also works
        independently of the ``show`` flag.
        """
        vis = self._last_obs_depth if self._visualize_noised_depth else None
        self._show_depth(vis)

    # def get_visualization(self, env_id: int = 0) -> np.ndarray:
    #     """Return a grayscale depth image for the given environment.

    #     Concatenates all cameras horizontally into a single image.
    #     When ``visualize_noised_depth`` is True and ``compute()`` has run at
    #     least once, uses the same depth as observations; otherwise the raw
    #     ray-cast buffer.

    #     Returns
    #     -------
    #     np.ndarray
    #         Grayscale uint8 image of shape ``(H, W_total, 3)``.
    #     """
    #     if self._visualize_noised_depth and self._last_obs_depth is not None:
    #         buf = self._last_obs_depth
    #     else:
    #         buf = self.depth_buffer
    #     if buf.is_cuda:
    #         torch.cuda.synchronize()

    #     panels = []
    #     for s in range(self._num_sensors):
    #         frame = buf[env_id, s].detach().cpu().numpy()
    #         frame_norm = np.clip(frame / max(self.far_plane, 1e-3), 0.0, 1.0)
    #         frame_u8 = (frame_norm * 255).astype(np.uint8)
    #         panels.append(np.stack([frame_u8] * 3, axis=-1))

    #     return np.concatenate(panels, axis=1)  # (H, S*W, 3)
