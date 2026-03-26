import omnigibson as og
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm

# 必须关闭 transition rules
gm.ENABLE_TRANSITION_RULES = False
# 最快模式：无界面
gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = False
INPUT_H5 = "/home/ps/BEHAVIOR-1K/episode_00000010.h5"
OUTPUT_H5 = "/home/ps/BEHAVIOR-1K/episode_00000010_datawrapper_replay.h5"
CAM_H = 480
CAM_W = 832

ROBOT_SENSOR_CONFIG = {
    "VisionSensor": {
        "modalities": ["depth_linear", "seg_instance_id", "rgb"],
        "sensor_kwargs": {
            "image_height": CAM_H,
            "image_width": CAM_W,
        },
    }
}
ROBOT_SENSOR_CONFIG2 = []

# 最简单：只回放并记录 depth / seg
env = DataPlaybackWrapper.create_from_hdf5(
    input_path=INPUT_H5,
    output_path=OUTPUT_H5,
    robot_obs_modalities=["depth_linear", "seg_instance_id", "rgb"],
    robot_sensor_config=ROBOT_SENSOR_CONFIG,
    include_sensor_names=["zed_link"],
    n_render_iterations=0,
    # include_task=False,
    # include_task_obs=False,
    # include_robot_control=False,
    # include_contacts=False,
    external_sensors_config=[],
    only_successes=False,
    flush_every_n_traj=2000,
)

# Playback the entire dataset and record observations
env.playback_dataset(record_data=True)
# 关闭 OG
og.shutdown()
