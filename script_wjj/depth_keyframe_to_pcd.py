import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to rotation matrix (3, 3)."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_rotmat_batch(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion batch (N, 4) in xyzw to rotation matrices (N, 3, 3)."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {q.shape}")

    n = np.linalg.norm(q, axis=1, keepdims=True)
    safe = n > 1e-12
    qn = np.zeros_like(q, dtype=np.float64)
    qn[safe[:, 0]] = q[safe[:, 0]] / n[safe]

    x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    # For near-zero quaternions, return identity
    if not np.all(safe):
        R[~safe[:, 0]] = np.eye(3, dtype=np.float64)

    return R


def depth_to_points_world(depth: np.ndarray, K: np.ndarray, cam_pos: np.ndarray, cam_quat_xyzw: np.ndarray):
    """
    Back-project depth (H, W) to world-space point cloud (N, 3), using the same
    chain as safe_playback3 / depth_to_pcd:
        1) pixel-center intrinsics (cx-0.5, cy-0.5)
        2) p_camera = z * K^{-1} [u, v, 1]^T
        3) camera-axis fix: Rx(pi)
        4) camera->world transform from (cam_pos, cam_quat)
    """
    H, W = depth.shape
    # Use pixel-center rays, same as safe_playback.py (principal point shifted by -0.5)
    K_center = np.asarray(K, dtype=np.float32).copy()
    K_center[0, 2] = K_center[0, 2] - 0.5
    K_center[1, 2] = K_center[1, 2] - 0.5

    v, u = np.indices((H, W), dtype=np.float32)
    z = depth.astype(np.float32)

    valid = np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    uv = np.stack([u[valid], v[valid], np.ones_like(z[valid], dtype=np.float32)], axis=-1).astype(np.float64)
    z_valid = z[valid].astype(np.float64)

    Kinv = np.linalg.inv(K_center.astype(np.float64))
    pts_cam = z_valid[:, None] * (uv @ Kinv.T)

    # Same extra camera-coordinate adjustment as depth_to_pcd(): Rx(pi)
    rot_add = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    rel_rot = quat_xyzw_to_rotmat(cam_quat_xyzw)
    rel_rot_matrix = rel_rot @ rot_add
    rel_pos = np.asarray(cam_pos, dtype=np.float64).reshape(1, 3)

    pts_world = pts_cam @ rel_rot_matrix.T + rel_pos
    return pts_world.astype(np.float32), valid


def is_valid_intrinsic(K: np.ndarray) -> bool:
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        return False
    if not np.isfinite(K).all():
        return False
    return float(K[0, 0]) > 1e-8 and float(K[1, 1]) > 1e-8


def select_intrinsic_matrix(traj: h5py.Group) -> np.ndarray:
    # 1) Prefer static zed_camera_intrinsic if valid
    if "zed_camera_intrinsic" in traj:
        K = np.asarray(traj["zed_camera_intrinsic"][:], dtype=np.float32)
        if is_valid_intrinsic(K):
            return K

    # 2) Fallback to per-step intrinsics and pick first valid one
    if "zed_camera_intrinsic_step" in traj:
        Ks = np.asarray(traj["zed_camera_intrinsic_step"][:], dtype=np.float32)
        if Ks.ndim == 2 and Ks.shape == (3, 3):
            if is_valid_intrinsic(Ks):
                return Ks
        elif Ks.ndim == 3 and Ks.shape[1:] == (3, 3):
            for K in Ks:
                if is_valid_intrinsic(K):
                    return K

    raise RuntimeError(
        "No valid camera intrinsics found in H5. "
        "Need zed_camera_intrinsic (valid fx/fy) or zed_camera_intrinsic_step."
    )


def build_prim_pose_table(traj: h5py.Group):
    """Build prim_path -> (position[T,3], orientation[T,4]) lookup from traj/prims."""
    table = {}
    if "prims" not in traj:
        return table

    for grp_name in traj["prims"].keys():
        g = traj["prims"][grp_name]
        prim_path = str(g.attrs.get("prim_path", ""))
        if len(prim_path) == 0:
            continue
        if "position" not in g or "orientation" not in g:
            continue
        table[prim_path] = (np.asarray(g["position"][:], dtype=np.float32), np.asarray(g["orientation"][:], dtype=np.float32))

    return table


def parse_seg_mapping_step(traj: h5py.Group, step_idx: int):
    """Return seg_id(int) -> prim_path(str) for a given step."""
    if "seg_instance_id_prim_mapping" not in traj:
        return {}

    raw = traj["seg_instance_id_prim_mapping"][step_idx]
    s = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    obj = json.loads(s)

    mapping = {}
    for k, v in obj.items():
        try:
            seg_id = int(k)
        except Exception:
            continue
        prim_path = v.get("prim_path", None) if isinstance(v, dict) else None
        if isinstance(prim_path, str) and len(prim_path) > 0:
            mapping[seg_id] = prim_path
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Export 5 depth keyframes to 5 tracked point cloud .npy files")
    parser.add_argument("--input_h5", type=str, required=True, help="Input replay H5 path")
    parser.add_argument("--demo", type=str, default="demo_0", help="Demo group name, e.g. demo_0")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .npy point clouds")
    parser.add_argument("--output_frame",type=str,default="local",choices=["local", "world"],help="Output coordinate frame: local (object frame) or world")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as f:
        traj = f["data"][args.demo]
        obs_grp = traj["obs"]

        depth_key = None
        seg_key = None
        for k in obs_grp.keys():
            if k.endswith("::depth_linear"):
                depth_key = k
            if k.endswith("::seg_instance_id"):
                seg_key = k
                break
        if depth_key is None:
            raise RuntimeError("Cannot find depth dataset ending with '::depth_linear' in obs group.")
        if seg_key is None:
            raise RuntimeError("Cannot find seg dataset ending with '::seg_instance_id' in obs group.")

        depth_all = obs_grp[depth_key][:]  # (T, H, W)
        seg_all = obs_grp[seg_key][:]  # (T, H, W)
        cam_pos_all = traj["zed_camera_position"][:]  # (T, 3)
        cam_quat_all = traj["zed_camera_orientation"][:]  # (T, 4), xyzw
        K = select_intrinsic_matrix(traj)
        prim_pose_table = build_prim_pose_table(traj)
        seg_maps_all = [parse_seg_mapping_step(traj, i) for i in range(depth_all.shape[0])]

    T = int(depth_all.shape[0])
    keyframe_indices = [int(round(i * (T - 1) / 4.0)) for i in range(5)]

    print(f"[info] depth key: {depth_key}")
    print(f"[info] total frames: {T}")
    print(f"[info] keyframe indices: {keyframe_indices}")

    H, W = depth_all.shape[1], depth_all.shape[2]

    rot_series_cache = {}

    def get_rot_series(prim_path: str, quat_arr: np.ndarray) -> np.ndarray:
        if prim_path not in rot_series_cache:
            rot_series_cache[prim_path] = quat_xyzw_to_rotmat_batch(quat_arr).astype(np.float32)
        return rot_series_cache[prim_path]

    # -------- for each reference keyframe, build one full T x H x W x 3 sequence --------
    for rank, ref_idx in enumerate(keyframe_indices):
        ref_points_world, ref_valid_mask = depth_to_points_world(
            depth=depth_all[ref_idx],
            K=K,
            cam_pos=cam_pos_all[ref_idx],
            cam_quat_xyzw=cam_quat_all[ref_idx],
        )

        ref_seg_flat = np.asarray(seg_all[ref_idx]).reshape(-1)
        ref_valid_flat = ref_valid_mask.reshape(-1)
        ref_valid_pixel_indices = np.nonzero(ref_valid_flat)[0]
        ref_seg_valid = ref_seg_flat[ref_valid_flat]
        ref_seg_map = seg_maps_all[ref_idx]

        seg_to_indices = {}
        for i, sid in enumerate(ref_seg_valid):
            seg_to_indices.setdefault(int(sid), []).append(i)

        prim_groups = []
        for seg_id, local_indices in seg_to_indices.items():
            prim_path = ref_seg_map.get(seg_id, None)
            if prim_path is None:
                continue

            pose = prim_pose_table.get(prim_path, None)
            if pose is None:
                continue
            pos_arr, quat_arr = pose
            if ref_idx >= len(pos_arr):
                continue

            idx = np.asarray(local_indices, dtype=np.int64)
            p_world = ref_points_world[idx].astype(np.float32)
            pix_idx = ref_valid_pixel_indices[idx].astype(np.int64)

            prim_pos = pos_arr[ref_idx].astype(np.float32)
            prim_rot = quat_xyzw_to_rotmat(quat_arr[ref_idx]).astype(np.float32)
            p_local = (p_world - prim_pos) @ prim_rot

            prim_groups.append((prim_path, p_local, pix_idx))

        n_tracked = sum(g[1].shape[0] for g in prim_groups)
        if n_tracked == 0:
            raise RuntimeError(f"Tracking init failed at ref frame {ref_idx}: no valid local points.")

        print(f"[tracking] ref={ref_idx} initialized pixels: {n_tracked}")

        out_suffix = "world" if args.output_frame == "world" else "local"
        out_path = out_dir / f"track_from_keyframe_{rank:02d}_frame_{ref_idx:06d}_points_{out_suffix}_TxHxWx3.npy"

        seq = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=(T, H, W, 3))
        seq[:] = np.nan
        seq_flat = seq.reshape(T, H * W, 3)

        if args.output_frame == "local":
            # local points are constant over time, but only for frames where the prim pose exists
            time_chunk = 128
            for prim_path, p_local, pix_idx in prim_groups:
                pos_arr, _ = prim_pose_table[prim_path]
                n_valid = min(T, len(pos_arr))
                if n_valid <= 0:
                    continue
                for t0 in range(0, n_valid, time_chunk):
                    t1 = min(n_valid, t0 + time_chunk)
                    seq_flat[t0:t1, pix_idx, :] = p_local[None, :, :]
        else:
            time_chunk = 16
            for prim_path, p_local, pix_idx in prim_groups:
                pos_arr, quat_arr = prim_pose_table[prim_path]
                n_valid = min(T, len(pos_arr))
                if n_valid <= 0:
                    continue

                rot_series = get_rot_series(prim_path, quat_arr)[:n_valid]  # (t, 3, 3)
                pos_series = pos_arr[:n_valid].astype(np.float32)  # (t, 3)

                for t0 in range(0, n_valid, time_chunk):
                    t1 = min(n_valid, t0 + time_chunk)
                    # row-vector form: p_world = p_local @ R^T + t
                    world_points = np.einsum("mj,tij->tmi", p_local, rot_series[t0:t1], optimize=True)
                    world_points = world_points + pos_series[t0:t1, None, :]
                    seq_flat[t0:t1, pix_idx, :] = world_points.astype(np.float32)

        seq.flush()
        print(f"[saved] {out_path} | shape={seq.shape}")


if __name__ == "__main__":
    main()
