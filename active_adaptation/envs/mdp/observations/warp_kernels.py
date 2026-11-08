"""Warp GPU kernels for depth camera ray casting.

Ported from FAR-Holosoma. Uses NVIDIA Warp to perform massively parallel ray
casting against terrain meshes, producing depth or range images via a pinhole
camera model.

Each kernel has two variants:
* **shared K_inv**: a single ``wp.mat44`` intrinsic matrix used by every environment.
* **per-env FOV**: a flattened ``[num_envs, num_sensors, 16]`` array so each
  environment can have its own randomised FOV / intrinsic matrix.
"""

import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)


# =====================================================================
# Shared K_inv  (single intrinsic matrix for all envs)
# =====================================================================

@wp.kernel
def depth_range_kernel(
    terrain_mesh_id: wp.uint64,
    cam_poss: wp.array(dtype=wp.vec3, ndim=2),
    cam_quats: wp.array(dtype=wp.quat, ndim=2),
    K_inv: wp.mat44,
    far_plane: float,
    pixels: wp.array(dtype=float, ndim=4),
    c_x: int,
    c_y: int,
    calculate_depth: bool,
    face_labels: wp.array(dtype=wp.int32),
    seg: wp.array(dtype=wp.int32, ndim=4),
):
    env_id, cam_id, x, y = wp.tid()
    mesh = terrain_mesh_id
    cam_pos = cam_poss[env_id, cam_id]
    cam_quat = cam_quats[env_id, cam_id]

    cam_coords = wp.vec3(float(x), float(y), 1.0)
    cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)

    uv = wp.transform_vector(K_inv, cam_coords)
    uv_principal = wp.transform_vector(K_inv, cam_coords_principal)

    ro = cam_pos
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
    rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    multiplier = 1.0
    if calculate_depth:
        multiplier = wp.dot(rd, rd_principal)

    dist = NO_HIT_RAY_VAL
    seg_val = wp.int32(0)
    if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
        dist = multiplier * t
        seg_val = face_labels[f]

    pixels[env_id, cam_id, y, x] = dist
    seg[env_id, cam_id, y, x] = seg_val


@wp.kernel
def depth_range_dynamic_kernel(
    terrain_id: wp.uint64,
    robot_ids: wp.array(dtype=wp.uint64),
    body_poss: wp.array(dtype=wp.vec3, ndim=2),
    body_quats: wp.array(dtype=wp.quat, ndim=2),
    cam_poss: wp.array(dtype=wp.vec3, ndim=2),
    cam_quats: wp.array(dtype=wp.quat, ndim=2),
    K_inv: wp.mat44,
    far_plane: float,
    pixels: wp.array(dtype=wp.float32, ndim=4),
    c_x: int,
    c_y: int,
    calculate_depth: bool,
    num_bodies: int,
    face_labels: wp.array(dtype=wp.int32),
    seg: wp.array(dtype=wp.int32, ndim=4),
    mesh_labels: wp.array(dtype=wp.int32),
):
    env_id, cam_id, x, y = wp.tid()

    cam_pos = cam_poss[env_id, cam_id]
    cam_quat = cam_quats[env_id, cam_id]

    uv = wp.transform_vector(K_inv, wp.vec3(float(x), float(y), 1.0))
    uv_c = wp.transform_vector(K_inv, wp.vec3(float(c_x), float(c_y), 1.0))

    ro = cam_pos
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
    rd_c = wp.normalize(wp.quat_rotate(cam_quat, uv_c))

    mul = wp.float32(1.0)
    if calculate_depth:
        d = wp.dot(rd, rd_c)
        eps = wp.float32(1.0e-6)
        d = wp.max(d, eps)
        d = wp.min(d, wp.float32(1.0))
        mul = d

    best = wp.float32(NO_HIT_RAY_VAL)
    far_bound_world = wp.float32(far_plane)
    seg_val = wp.int32(0)

    for b in range(num_bodies):
        qb = body_quats[env_id, b]
        tb = body_poss[env_id, b]
        ro_l = wp.quat_rotate_inv(qb, ro - tb)
        rd_l = wp.quat_rotate_inv(qb, rd)

        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)

        if wp.mesh_query_ray(robot_ids[b], ro_l, rd_l, far_bound_world / mul, t, u, v, sign, n, f):
            d = mul * t
            if (best == NO_HIT_RAY_VAL) or (d < best):
                best = d
                far_bound_world = best
                seg_val = mesh_labels[b]

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    if wp.mesh_query_ray(terrain_id, ro, rd, far_bound_world / mul, t, u, v, sign, n, f):
        best = mul * t
        seg_val = face_labels[f]

    pixels[env_id, cam_id, y, x] = best
    seg[env_id, cam_id, y, x] = seg_val


# =====================================================================
# Per-env FOV  (each env gets its own K_inv from a flattened array)
# =====================================================================

@wp.kernel
def depth_range_per_env_fov_kernel(
    terrain_mesh_id: wp.uint64,
    cam_poss: wp.array(dtype=wp.vec3, ndim=2),
    cam_quats: wp.array(dtype=wp.quat, ndim=2),
    K_inv_array: wp.array(dtype=wp.float32, ndim=3),  # [num_envs, num_sensors, 16]
    far_plane: float,
    pixels: wp.array(dtype=float, ndim=4),
    c_x: int,
    c_y: int,
    calculate_depth: bool,
    face_labels: wp.array(dtype=wp.int32),
    seg: wp.array(dtype=wp.int32, ndim=4),
):
    env_id, cam_id, x, y = wp.tid()
    mesh = terrain_mesh_id
    cam_pos = cam_poss[env_id, cam_id]
    cam_quat = cam_quats[env_id, cam_id]

    K_inv = wp.mat44(
        K_inv_array[env_id, cam_id, 0],  K_inv_array[env_id, cam_id, 1],  K_inv_array[env_id, cam_id, 2],  K_inv_array[env_id, cam_id, 3],
        K_inv_array[env_id, cam_id, 4],  K_inv_array[env_id, cam_id, 5],  K_inv_array[env_id, cam_id, 6],  K_inv_array[env_id, cam_id, 7],
        K_inv_array[env_id, cam_id, 8],  K_inv_array[env_id, cam_id, 9],  K_inv_array[env_id, cam_id, 10], K_inv_array[env_id, cam_id, 11],
        K_inv_array[env_id, cam_id, 12], K_inv_array[env_id, cam_id, 13], K_inv_array[env_id, cam_id, 14], K_inv_array[env_id, cam_id, 15],
    )

    cam_coords = wp.vec3(float(x), float(y), 1.0)
    cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)

    uv = wp.transform_vector(K_inv, cam_coords)
    uv_principal = wp.transform_vector(K_inv, cam_coords_principal)

    ro = cam_pos
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
    rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    multiplier = 1.0
    if calculate_depth:
        multiplier = wp.dot(rd, rd_principal)

    dist = NO_HIT_RAY_VAL
    seg_val = wp.int32(0)
    if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
        dist = multiplier * t
        seg_val = face_labels[f]

    pixels[env_id, cam_id, y, x] = dist
    seg[env_id, cam_id, y, x] = seg_val


@wp.kernel
def depth_range_dynamic_per_env_fov_kernel(
    terrain_id: wp.uint64,
    robot_ids: wp.array(dtype=wp.uint64),
    body_poss: wp.array(dtype=wp.vec3, ndim=2),
    body_quats: wp.array(dtype=wp.quat, ndim=2),
    cam_poss: wp.array(dtype=wp.vec3, ndim=2),
    cam_quats: wp.array(dtype=wp.quat, ndim=2),
    K_inv_array: wp.array(dtype=wp.float32, ndim=3),  # [num_envs, num_sensors, 16]
    far_plane: float,
    pixels: wp.array(dtype=wp.float32, ndim=4),
    c_x: int,
    c_y: int,
    calculate_depth: bool,
    num_bodies: int,
    face_labels: wp.array(dtype=wp.int32),
    seg: wp.array(dtype=wp.int32, ndim=4),
    mesh_labels: wp.array(dtype=wp.int32),
):
    env_id, cam_id, x, y = wp.tid()

    cam_pos = cam_poss[env_id, cam_id]
    cam_quat = cam_quats[env_id, cam_id]

    K_inv = wp.mat44(
        K_inv_array[env_id, cam_id, 0],  K_inv_array[env_id, cam_id, 1],  K_inv_array[env_id, cam_id, 2],  K_inv_array[env_id, cam_id, 3],
        K_inv_array[env_id, cam_id, 4],  K_inv_array[env_id, cam_id, 5],  K_inv_array[env_id, cam_id, 6],  K_inv_array[env_id, cam_id, 7],
        K_inv_array[env_id, cam_id, 8],  K_inv_array[env_id, cam_id, 9],  K_inv_array[env_id, cam_id, 10], K_inv_array[env_id, cam_id, 11],
        K_inv_array[env_id, cam_id, 12], K_inv_array[env_id, cam_id, 13], K_inv_array[env_id, cam_id, 14], K_inv_array[env_id, cam_id, 15],
    )

    uv = wp.transform_vector(K_inv, wp.vec3(float(x), float(y), 1.0))
    uv_c = wp.transform_vector(K_inv, wp.vec3(float(c_x), float(c_y), 1.0))

    ro = cam_pos
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
    rd_c = wp.normalize(wp.quat_rotate(cam_quat, uv_c))

    mul = wp.float32(1.0)
    if calculate_depth:
        d = wp.dot(rd, rd_c)
        eps = wp.float32(1.0e-6)
        d = wp.max(d, eps)
        d = wp.min(d, wp.float32(1.0))
        mul = d

    best = wp.float32(NO_HIT_RAY_VAL)
    far_bound_world = wp.float32(far_plane)
    seg_val = wp.int32(0)

    for b in range(num_bodies):
        qb = body_quats[env_id, b]
        tb = body_poss[env_id, b]
        ro_l = wp.quat_rotate_inv(qb, ro - tb)
        rd_l = wp.quat_rotate_inv(qb, rd)

        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)

        if wp.mesh_query_ray(robot_ids[b], ro_l, rd_l, far_bound_world / mul, t, u, v, sign, n, f):
            d = mul * t
            if (best == NO_HIT_RAY_VAL) or (d < best):
                best = d
                far_bound_world = best
                seg_val = mesh_labels[b]

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    if wp.mesh_query_ray(terrain_id, ro, rd, far_bound_world / mul, t, u, v, sign, n, f):
        best = mul * t
        seg_val = face_labels[f]

    pixels[env_id, cam_id, y, x] = best
    seg[env_id, cam_id, y, x] = seg_val
