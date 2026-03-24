from omnigibson.envs import RBSDataPlaybackWrapper
import omnigibson as og
from omnigibson.macros import gm
import os
import h5py

# Disable transition rules for DataPlaybackWrapper
gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = False
INPUT_H5 = "/home/ps/BEHAVIOR-1K/episode_00000010.h5"
OUTPUT_H5 = "/home/ps/BEHAVIOR-1K/episode_00000010_new3.h5"
OUTPUT_RGB_VIDEO = "/home/ps/BEHAVIOR-1K/episode_00000010_rgb.mp4"
FLUSH_EVERY_N_STEPS = 2000
ENABLE_RGB_RECORDING =True
OBS_MODALITIES = ["depth_linear", "seg_instance_id"]

# Record all frames
os.environ["OG_RECORD_GRIPPER_CONTACTS"] = "1"

# Full-frame mode: keep every frame depth + seg
with h5py.File(INPUT_H5, "r") as _f:
    _first_demo = _f["data"]["demo_0"]
    _n_steps = int(_first_demo["action"].shape[0])
print(f"[OBS export] full-frame mode, total steps: {_n_steps}")

runtime_flush_every_n_steps = FLUSH_EVERY_N_STEPS
print(f"[record] full-frame mode: flush_every_n_steps={runtime_flush_every_n_steps}")
# Create a playback environment
playback_env = RBSDataPlaybackWrapper.create_from_hdf5(
    input_path=INPUT_H5,
    output_path=OUTPUT_H5,
    include_task_obs=False,
    robot_obs_modalities=OBS_MODALITIES + (["rgb"] if ENABLE_RGB_RECORDING else []),
    robot_sensor_config={
        "VisionSensor": {
            "modalities": OBS_MODALITIES + (["rgb"] if ENABLE_RGB_RECORDING else []),
            "sensor_kwargs": {
                "image_height": 480,
                "image_width": 832,
            },
        }
    },
    external_sensors_config=None,
    include_sensor_names=["zed_link"],
    n_render_iterations=1,
    only_successes=False,
    flush_every_n_traj=1,
    flush_every_n_steps=runtime_flush_every_n_steps,
    enable_rgb_recording=ENABLE_RGB_RECORDING,
    output_rgb_video=OUTPUT_RGB_VIDEO,
    rgb_video_resolution=(480, 832),
    rgb_video_rate=16.0,
)
print(f"[record] flush every {runtime_flush_every_n_steps} steps")

# Playback the entire dataset and record observations + rgb videos
n_episodes = playback_env.input_hdf5["data"].attrs["n_episodes"]

for episode_id in range(n_episodes):
    playback_env.playback_episode(
        episode_id=episode_id,
        record_data=True,
        video_writers=None,
    )

# Save the recorded playback data
playback_env.save_data()

# Properly shutdown OmniGibson
og.shutdown()