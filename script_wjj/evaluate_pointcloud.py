import omnigibson.lazy as lazy
import omnigibson as og
import numpy as np
from pathlib import Path

cfg = dict()

# Define scene
cfg["scene"] = {
    "type": "InteractiveTraversableScene",
    "scene_model": "house_double_floor_lower",
    "scene_instance": "house_double_floor_lower_task_turning_on_radio_0_0_template",
    "load_room_types": ["corridor", "kitchen", "living_room"],
    "include_robots": False,
}

# Define robots
cfg["robots"] = [
    {
        "type": "R1Pro",
        "name": "baby_robot",
        # Example initial pose (world frame)
        "position": [4.33, 5.15, 0.0],
        # Quaternion in xyzw, here yaw=90deg around z axis
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "include_sensor_names": [],
    },
]

# Define task
cfg["task"] = {
    "type": "DummyTask",
    "termination_config": dict(),
    "reward_config": dict(),
}

# Create the environment
env = og.Environment(cfg)

# Step!
# Use Isaac Sim debug draw for lightweight point cloud visualization.
draw = lazy.isaacsim.util.debug_draw._debug_draw.acquire_debug_draw_interface()

# Load full tracked sequence (T, H, W, 3) once
track_npy_path = Path(
    "/home/ps/BEHAVIOR-1K/keyframe_pointclouds2/track_from_keyframe_02_frame_000978_points_world_TxHxWx3.npy"
)
if not track_npy_path.exists():
    raise FileNotFoundError(f"Tracking sequence file not found: {track_npy_path}")

tracked_seq = np.load(track_npy_path)  # (T, H, W, 3)
if tracked_seq.ndim != 4 or tracked_seq.shape[-1] != 3:
    raise ValueError(f"Expected tracking npy shape (T, H, W, 3), got {tracked_seq.shape}")

track_T = tracked_seq.shape[0]
track_frame_idx = 0
print(f"[tracking replay] loaded {track_npy_path.name}, shape={tracked_seq.shape}")

while True:
    action = {k: v * 0 for k, v in env.action_space.sample().items()}
    env.step(action)
    if hasattr(draw, "clear_points"):
        draw.clear_points()
    
    # Replay full tracking sequence, one frame per sim step
    tracked_frame = tracked_seq[track_frame_idx % track_T].reshape(-1, 3)
    tracked_valid = np.isfinite(tracked_frame).all(axis=1)
    exported_pts_list = [(float(x), float(y), float(z)) for x, y, z in tracked_frame[tracked_valid]]
    track_frame_idx += 1
    
    if hasattr(draw, "draw_points"):
        if exported_pts_list:
            draw.draw_points(exported_pts_list, [(0.0, 0.0, 1.0, 1.0)] * len(exported_pts_list), [10.0] * len(exported_pts_list))