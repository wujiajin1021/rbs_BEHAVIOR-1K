import json
import os
import re
from collections import defaultdict
from fractions import Fraction
import logging

import h5py
import torch as th
import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.controllers.controller_base import ControlType
from omnigibson.envs.data_wrapper import DataWrapper
from omnigibson.envs.env_wrapper import create_wrapper
from omnigibson.macros import gm, macros
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
import omnigibson.utils.transform_utils as T
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import PoseAPI, mesh_prim_to_trimesh_mesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes


log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)


try:
    from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
except ImportError:
    create_video_writer = None
    write_video = None


class RBSDataPlaybackWrapper(DataWrapper):
    """
    An OmniGibson environment wrapper for playing back data and collecting observations.

    NOTE: This assumes a DataCollectionWrapper environment has been used to collect data!
    """

    @classmethod
    def create_from_hdf5(
        cls,
        input_path,
        output_path,
        compression=dict(),
        robot_obs_modalities=tuple(),
        robot_proprio_keys=None,
        robot_sensor_config=None,
        external_sensors_config=None,
        include_sensor_names=None,
        exclude_sensor_names=None,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        include_env_wrapper=False,
        additional_wrapper_configs=None,
        full_scene_file=None,
        include_task=True,
        include_task_obs=True,
        include_robot_control=True,
        include_contacts=True,
        load_room_instances=None,
        enable_rgb_recording=False,
        output_rgb_video=None,
        rgb_video_resolution=(480, 832),
        rgb_video_rate=16.0,
    ):
        """
        Create a DataPlaybackWrapper environment instance form the recorded demonstration info
        from @hdf5_path, and aggregate observation_modalities @obs during playback

        Args:
            input_path (str): Absolute path to the input hdf5 file containing the relevant collected data to playback
            output_path (str): Absolute path to the output hdf5 file that will contain the recorded observations from
                the replayed data
            compression (dict): If specified, the compression arguments to use for the hdf5 file.
            robot_obs_modalities (list): Robot observation modalities to use. This list is directly passed into
                the robot_cfg (`obs_modalities` kwarg) when spawning the robot
            robot_proprio_keys (None or list of str): If specified, a list of proprioception keys to use for the robot.
            robot_sensor_config (None or dict): If specified, the sensor configuration to use for the robot. See the
                example sensor_config in fetch_behavior.yaml env config. This can be used to specify relevant sensor
                params, such as image_height and image_width
            external_sensors_config (None or list): If specified, external sensor(s) to use. This will override the
                external_sensors kwarg in the env config when the environment is loaded. Each entry should be a
                dictionary specifying an individual external sensor's relevant parameters. See the example
                external_sensors key in fetch_behavior.yaml env config. This can be used to specify additional sensors
                to collect observations during playback.
            include_sensor_names (None or list of str): If specified, substring(s) to check for in all raw sensor prim
                paths found on the robot. A sensor must include one of the specified substrings in order to be included
                in this robot's set of sensors during playback
            exclude_sensor_names (None or list of str): If specified, substring(s) to check against in all raw sensor
                prim paths found on the robot. A sensor must not include any of the specified substrings in order to
                be included in this robot's set of sensors during playback
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data. This is needed because the omniverse real-time raytracing always lags behind the
                underlying physical state by a few frames, and additionally produces transient visual artifacts when
                the physical state changes. Increasing this number will improve the rendered quality at the expense of
                speed.
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file
            flush_every_n_steps (int): How often to flush (write) current data to file within an episode.
                If this is greater than 0, flush_every_n_traj must be set to 1.
            include_env_wrapper (bool): Whether to include environment wrapper stored in the underlying env config
            additional_wrapper_configs (None or list of dict): If specified, list of wrapper config(s) specifying
                environment wrappers to wrap the internal environment class in
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection
                the scene file stored may be partial, and will be used to fill in the missing scene objects from the
                full scene file.
            include_task (bool): Whether to include the original task or not. If False, will use a DummyTask instead
            include_task_obs (bool): Whether to include task observations or not. If False, will not include task obs
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all
                objects to be visual_only
            load_room_instances (None or list of str): If specified, list of room instance names to load during
                playback

        Returns:
            RBSDataPlaybackWrapper: Generated playback environment
        """
        # check flush parameters
        if flush_every_n_steps > 0:
            assert flush_every_n_traj == 1, "flush_every_n_traj must be 1 if flush_every_n_steps is greater than 0"
        # Read from the HDF5 file
        f = h5py.File(input_path, "r")
        config = json.loads(f["data"].attrs["config"])

        # Hot swap in additional info for playing back data

        if include_contacts:
            # Minimize physics leakage during playback (we need to take an env step when loading state)
            config["env"]["action_frequency"] = 1000.0
            config["env"]["rendering_frequency"] = 1000.0
            config["env"]["physics_frequency"] = 1000.0
        else:
            # Since we are setting all objects to be visual-only, physics will not be propogating
            config["env"]["action_frequency"] = 30.0
            config["env"]["rendering_frequency"] = 30.0
            config["env"]["physics_frequency"] = 120.0
            # Simulator-level visual-only set to True
            gm.VISUAL_ONLY = True

        # Make sure obs space is flattened for recording
        config["env"]["flatten_obs_space"] = True

        # Set the scene file either to the one stored in the hdf5 or the hot swap scene file
        config["scene"]["scene_file"] = json.loads(f["data"].attrs["scene_file"])
        if full_scene_file:
            with open(full_scene_file, "r") as json_file:
                full_scene_json = json.load(json_file)
            config["scene"]["scene_file"] = merge_scene_files(
                scene_a=full_scene_json, scene_b=config["scene"]["scene_file"], keep_robot_from="b"
            )
            # Overwrite rooms type to avoid loading room types from the hdf5 file
            config["scene"]["load_room_types"] = None
            config["scene"]["load_room_instances"] = load_room_instances
        else:
            config["scene"]["scene_file"] = json.loads(f["data"].attrs["scene_file"])

        # Use dummy task if not loading task
        if not include_task:
            config["task"] = {"type": "DummyTask"}

        # Maybe include task observations
        config["task"]["include_obs"] = include_task_obs

        # Set scene file and disable online object sampling if BehaviorTask is being used
        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False
            # Don't use presampled robot pose
            config["task"]["use_presampled_robot_pose"] = False

        # Because we're loading directly from the cached scene file, we need to disable any additional objects that are being added since
        # they will already be cached in the original scene file
        config["objects"] = []

        # Set observation modalities and update sensor config
        for robot_cfg in config["robots"]:
            robot_cfg["obs_modalities"] = list(robot_obs_modalities)
            robot_cfg["include_sensor_names"] = include_sensor_names
            robot_cfg["exclude_sensor_names"] = exclude_sensor_names
            if robot_proprio_keys is not None:
                robot_cfg["proprio_obs"] = robot_proprio_keys
            if robot_sensor_config is not None:
                robot_cfg["sensor_config"] = robot_sensor_config
        if external_sensors_config is not None:
            config["env"]["external_sensors"] = external_sensors_config

        # Load env
        env = og.Environment(configs=config)

        # Optionally include the desired environment wrapper specified in the config
        if include_env_wrapper:
            env = create_wrapper(env=env)

        if additional_wrapper_configs is not None:
            for wrapper_cfg in additional_wrapper_configs:
                env = create_wrapper(env=env, wrapper_cfg=wrapper_cfg)

        # Wrap and return env
        return cls(
            env=env,
            input_path=input_path,
            output_path=output_path,
            compression=compression,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            enable_rgb_recording=enable_rgb_recording,
            output_rgb_video=output_rgb_video,
            rgb_video_resolution=rgb_video_resolution,
            rgb_video_rate=rgb_video_rate,
        )

    def __init__(
        self,
        env,
        input_path,
        output_path,
        compression=dict(),
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        full_scene_file=None,
        load_room_instances=None,
        include_robot_control=True,
        include_contacts=True,
        enable_rgb_recording=False,
        output_rgb_video=None,
        rgb_video_resolution=(480, 832),
        rgb_video_rate=16.0,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to input hdf5 collected data file
            output_path (str): path to store output hdf5 data file
            compression (dict): If specified, the compression arguments to use for the hdf5 file.
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file across episodes
            flush_every_n_steps (int): How often to flush (write) current data to file within an episode.
                If this is greater than 0, flush_every_n_traj must be set to 1.
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection,
                the scene file stored may be partial, and this will be used to fill in the missing scene objects from the
                full scene file.
            load_room_instances (None or str): If specified, the room instances to load for playback.
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all objects to be visual_only
        """
        # Make sure transition rules are DISABLED for playback since we manually propagate transitions
        assert not gm.ENABLE_TRANSITION_RULES, "Transition rules must be disabled for DataPlaybackWrapper env!"

        # Stabilize skipped objects
        # we can do this here because we know that whatever's skipped during load state must have been asleep during data collection
        # which means they're not moving and we can safely keep them still
        with macros.unlocked():
            macros.utils.registry_utils.STABILIZE_SKIPPED_OBJECTS = True

        # Store scene file so we can restore the data upon each episode reset
        self.input_hdf5 = h5py.File(input_path, "r")
        self.scene_file = json.loads(self.input_hdf5["data"].attrs["scene_file"])
        assert not (
            load_room_instances and not full_scene_file
        ), "Full scene file must be specified in order to load room instances"
        if full_scene_file:
            with open(full_scene_file, "r") as json_file:
                full_scene_json = json.load(json_file)
            self.scene_file = merge_scene_files(scene_a=full_scene_json, scene_b=self.scene_file, keep_robot_from="b")
            if load_room_instances is not None and full_scene_file is not None:
                # we loaded more room than the stored scene file, but still not the full scene
                # we need to save the current scene file here to avoid errors
                self.scene_file = env.scene.save(as_dict=True)

        # Store additional variables
        self.n_render_iterations = n_render_iterations
        if flush_every_n_steps > 0:
            assert flush_every_n_traj == 1, "flush_every_n_traj must be 1 if flush_every_n_steps is greater than 0"
        self.flush_every_n_steps = flush_every_n_steps

        self.current_traj_grp = None
        self.current_episode_step_count = 0
        self.traj_dsets = dict()
        self.include_robot_control = include_robot_control
        self.include_contacts = include_contacts
        self.enable_rgb_recording = bool(enable_rgb_recording)
        self.output_rgb_video = (
            output_rgb_video
            if output_rgb_video is not None
            else str(output_path).replace(".h5", "_rgb.mp4").replace(".hdf5", "_rgb.mp4")
        )
        self.rgb_video_resolution = tuple(rgb_video_resolution)
        self.rgb_video_rate = rgb_video_rate
        self._default_video_writers = None

        # Run super
        super().__init__(
            env=env,
            output_path=output_path,
            compression=compression,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
        )

    def _create_default_video_writers(self):
        if not self.enable_rgb_recording:
            return {}

        assert create_video_writer is not None, (
            "RGB recording requested but create_video_writer is unavailable. "
            "Please make sure omnigibson eval dependencies are installed."
        )

        rgb_obs_keys = [
            k
            for k in self.observation_space.keys()
            if str(k).endswith("::rgb") and "zed_link:Camera:0::rgb" in str(k)
        ]

        rate = self.rgb_video_rate
        if isinstance(rate, float):
            rate = int(rate) if rate.is_integer() else Fraction(str(rate))

        writers = {}
        for i, rgb_key in enumerate(rgb_obs_keys):
            suffix = "" if i == 0 else f"_{i}"
            video_path = self.output_rgb_video if i == 0 else self.output_rgb_video.replace(".mp4", f"{suffix}.mp4")
            writers[rgb_key] = create_video_writer(
                fpath=video_path,
                resolution=self.rgb_video_resolution,
                codec_name="libx264",
                pix_fmt="yuv420p",
                rate=rate,
            )
        return writers

    def _get_default_video_writers(self):
        if self._default_video_writers is None:
            self._default_video_writers = self._create_default_video_writers()
        return self._default_video_writers

    def _close_default_video_writers(self):
        if self._default_video_writers is None:
            return
        for container, stream in self._default_video_writers.values():
            try:
                # Flush delayed packets (e.g. from B-frames / encoder lookahead)
                # so the tail frames are not dropped.
                for packet in stream.encode(None):
                    container.mux(packet)
            finally:
                container.close()
        self._default_video_writers = None

    def _safe_hdf5_group_name(self, name):
        """Convert an object name into an HDF5-safe group name."""
        safe = str(name).replace("/", "__")
        return safe if len(safe) > 0 else "obj"

    def _get_object_usd_path(self, obj):
        """Best-effort retrieval of an object's usd path as a string."""
        usd_path = getattr(obj, "usd_path", None)
        if usd_path:
            return str(usd_path)

        get_usd_path = getattr(obj, "get_usd_path", None)
        if callable(get_usd_path):
            try:
                usd_path = get_usd_path()
                if usd_path:
                    return str(usd_path)
            except Exception:
                pass

        return ""

    def _get_scene_visual_mesh_prim_entries(self):
        """Returns deterministic metadata for rigid visual mesh prims in the scene."""
        prim_entries = []
        if hasattr(self, "scene") and self.scene is not None:
            for obj in self.scene.objects:
                obj_name = str(getattr(obj, "name", ""))
                obj_usd_path = self._get_object_usd_path(obj)
                for link in obj.links.values():
                    link_name = str(getattr(link, "name", ""))
                    for vm in link.visual_meshes.values():
                        prim_path = str(getattr(vm, "prim_path", ""))
                        if len(prim_path) == 0:
                            continue
                        prim_entries.append(
                            {
                                "prim_path": prim_path,
                                "prim_parent_path": prim_path.rsplit("/", 1)[0] if "/" in prim_path else prim_path,
                                "object_name": obj_name,
                                "link_name": link_name,
                                "usd_path": obj_usd_path,
                                "prim_name": str(getattr(vm, "name", prim_path.split("/")[-1])),
                            }
                        )

        prim_entries.sort(key=lambda x: x["prim_path"])
        return prim_entries

    def _get_recorded_prim_entries(self, n_prim, traj_grp=None):
        """Best-effort retrieval of prim metadata aligned with the recorded prim index axis."""
        if traj_grp is not None and "prim_path_list" in traj_grp:
            prim_paths = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in traj_grp["prim_path_list"][:]]
            object_names = (
                [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in traj_grp["prim_object_name_list"][:]]
                if "prim_object_name_list" in traj_grp
                else [""] * len(prim_paths)
            )
            link_names = (
                [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in traj_grp["prim_link_name_list"][:]]
                if "prim_link_name_list" in traj_grp
                else [""] * len(prim_paths)
            )
            usd_paths = (
                [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in traj_grp["prim_usd_path_list"][:]]
                if "prim_usd_path_list" in traj_grp
                else [""] * len(prim_paths)
            )
            prim_names = [path.split("/")[-1] if len(path) > 0 else "" for path in prim_paths]
            entries = []
            for idx, prim_path in enumerate(prim_paths[:n_prim]):
                entries.append(
                    {
                        "prim_path": prim_path,
                        "prim_parent_path": prim_path.rsplit("/", 1)[0] if "/" in prim_path else prim_path,
                        "object_name": object_names[idx] if idx < len(object_names) else "",
                        "link_name": link_names[idx] if idx < len(link_names) else "",
                        "usd_path": usd_paths[idx] if idx < len(usd_paths) else "",
                        "prim_name": prim_names[idx],
                    }
                )
            if len(entries) >= n_prim:
                return entries[:n_prim]

        prim_entries = self._get_scene_visual_mesh_prim_entries()
        if len(prim_entries) < n_prim:
            prim_entries += [
                {
                    "prim_path": f"/World/unresolved_prim_{i}",
                    "prim_parent_path": f"/World/unresolved_prim_{i}",
                    "object_name": "",
                    "link_name": "",
                    "usd_path": "",
                    "prim_name": f"unresolved_prim_{i}",
                }
                for i in range(len(prim_entries), n_prim)
            ]
        return prim_entries[:n_prim]

    def _postprocess_prims_group(self, traj_grp):
        """
        Convert flat prim tensors (T, N, ...) into:
            traj_grp/prims/<prim_path>/{position, orientation, ...}
        and drop old flat prim_* datasets.
        """
        required = (
            "prim_positions",
            "prim_orientations",
            "prim_positions_in_zed_camera",
            "prim_orientations_in_zed_camera",
        )
        if any(k not in traj_grp for k in required):
            return

        n_prim = int(traj_grp["prim_positions"].shape[1])
        prim_entries = self._get_recorded_prim_entries(n_prim=n_prim, traj_grp=traj_grp)

        for meta_key, values in (
            ("prim_path_list", [entry["prim_path"] for entry in prim_entries]),
            ("prim_object_name_list", [entry["object_name"] for entry in prim_entries]),
            ("prim_link_name_list", [entry["link_name"] for entry in prim_entries]),
            ("prim_usd_path_list", [entry.get("usd_path", "") for entry in prim_entries]),
        ):
            if meta_key in traj_grp:
                del traj_grp[meta_key]
            traj_grp.create_dataset(meta_key, data=values, dtype=h5py.string_dtype(encoding="utf-8"))

        if "prims" in traj_grp:
            del traj_grp["prims"]
        prims_grp = traj_grp.create_group("prims")
        prims_grp.attrs["n_prims"] = n_prim

        source_to_target = {
            "prim_positions": "position",
            "prim_orientations": "orientation",
            "prim_positions_in_zed_camera": "position_in_zed_camera",
            "prim_orientations_in_zed_camera": "orientation_in_zed_camera",
        }

        used_group_names = set()
        for idx, entry in enumerate(prim_entries):
            base_name = self._safe_hdf5_group_name(entry["prim_path"])
            grp_name = base_name
            suffix = 1
            while grp_name in used_group_names:
                grp_name = f"{base_name}_{suffix}"
                suffix += 1
            used_group_names.add(grp_name)

            prim_grp = prims_grp.create_group(grp_name)
            prim_grp.attrs["prim_path"] = entry["prim_path"]
            prim_grp.attrs["prim_parent_path"] = entry["prim_parent_path"]
            prim_grp.attrs["prim_name"] = entry["prim_name"]
            prim_grp.attrs["object_name"] = entry["object_name"]
            prim_grp.attrs["link_name"] = entry["link_name"]
            prim_grp.attrs["usd_path"] = entry.get("usd_path", "")
            prim_grp.attrs["prim_index"] = idx

            for src_key, dst_key in source_to_target.items():
                if src_key not in traj_grp:
                    continue
                src_dset = traj_grp[src_key]
                prim_grp.create_dataset(dst_key, data=src_dset[:, idx, ...], **self.compression)

        # Drop flat prim datasets to keep only prim-centric representation
        for key in (
            "prim_indices",
            "prim_positions",
            "prim_orientations",
            "prim_positions_in_zed_camera",
            "prim_orientations_in_zed_camera",
        ):
            if key in traj_grp:
                del traj_grp[key]

    def _prune_recorded_prims_by_seg_mapping(self, traj_grp):
        """Keep only prims that appear in retained seg_instance_id frames and remap prim indices."""
        if "seg_instance_id_prim_mapping" not in traj_grp or "prim_positions" not in traj_grp:
            return

        mapping_ds = traj_grp["seg_instance_id_prim_mapping"]
        keep_indices = set()
        remapped_mappings = []

        for i in range(mapping_ds.shape[0]):
            raw = mapping_ds[i]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                mapping = json.loads(raw) if isinstance(raw, str) else {}
            except Exception:
                mapping = {}

            if not isinstance(mapping, dict):
                mapping = {}

            for meta in mapping.values():
                if isinstance(meta, dict):
                    prim_index = int(meta.get("prim_index", -1))
                    if prim_index >= 0:
                        keep_indices.add(prim_index)

            remapped_mappings.append(mapping)

        if len(keep_indices) == 0:
            return

        keep_indices = sorted(keep_indices)
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        prim_entries = self._get_recorded_prim_entries(n_prim=int(traj_grp["prim_positions"].shape[1]), traj_grp=None)
        kept_entries = [prim_entries[idx] for idx in keep_indices if idx < len(prim_entries)]

        for key in (
            "prim_indices",
            "prim_positions",
            "prim_orientations",
            "prim_positions_in_zed_camera",
            "prim_orientations_in_zed_camera",
        ):
            if key not in traj_grp:
                continue
            dset = traj_grp[key]
            trimmed = dset[:, keep_indices, ...] if len(dset.shape) >= 2 else dset[keep_indices, ...]
            del traj_grp[key]
            traj_grp.create_dataset(key, data=trimmed, **self.compression)

        serialized = []
        for mapping in remapped_mappings:
            new_mapping = {}
            for seg_id, meta in mapping.items():
                if not isinstance(meta, dict):
                    continue
                old_idx = int(meta.get("prim_index", -1))
                if old_idx not in index_map:
                    continue
                meta = dict(meta)
                meta["prim_index"] = index_map[old_idx]
                new_mapping[seg_id] = meta
            serialized.append(json.dumps(new_mapping, cls=TorchEncoder))

        del traj_grp["seg_instance_id_prim_mapping"]
        traj_grp.create_dataset(
            "seg_instance_id_prim_mapping",
            data=serialized,
            dtype=h5py.string_dtype(encoding="utf-8"),
            **self.compression,
        )

        for meta_key, values in (
            ("prim_path_list", [entry["prim_path"] for entry in kept_entries]),
            ("prim_object_name_list", [entry["object_name"] for entry in kept_entries]),
            ("prim_link_name_list", [entry["link_name"] for entry in kept_entries]),
            ("prim_usd_path_list", [entry.get("usd_path", "") for entry in kept_entries]),
        ):
            if meta_key in traj_grp:
                del traj_grp[meta_key]
            traj_grp.create_dataset(meta_key, data=values, dtype=h5py.string_dtype(encoding="utf-8"))

    def postprocess_traj_group(self, traj_grp):
        # Store camera intrinsics once per trajectory (intrinsics are static)
        self._write_zed_camera_intrinsic_once(traj_grp=traj_grp)

        self._prune_recorded_prims_by_seg_mapping(traj_grp=traj_grp)

        super().postprocess_traj_group(traj_grp=traj_grp)
        self._postprocess_prims_group(traj_grp=traj_grp)

    def _write_zed_camera_intrinsic_once(self, traj_grp):
        """Writes a single 3x3 zed camera intrinsic matrix into @traj_grp if available."""
        key = "zed_camera_intrinsic"
        if key in traj_grp:
            return

        # Prefer per-step recorded intrinsic matrices and pick the first valid one
        if "zed_camera_intrinsic_step" in traj_grp:
            try:
                Ks = np.asarray(traj_grp["zed_camera_intrinsic_step"][:], dtype=np.float32)
                if Ks.ndim == 2 and Ks.shape == (3, 3):
                    Ks = Ks.reshape(1, 3, 3)
                if Ks.ndim == 3 and Ks.shape[1:] == (3, 3):
                    for K_candidate in Ks:
                        if self._is_valid_intrinsic_matrix(K_candidate):
                            traj_grp.create_dataset(key, data=K_candidate.astype(np.float32), **self.compression)
                            return
            except Exception:
                pass

        robot = self.env.robots[0] if len(self.env.robots) > 0 else None
        if robot is None or not hasattr(robot, "sensors"):
            return

        zed_sensor = None
        for sensor in robot.sensors.values():
            prim_path = getattr(sensor, "prim_path", "")
            if "zed_link" in str(prim_path) and "Camera" in str(prim_path) and hasattr(sensor, "intrinsic_matrix"):
                zed_sensor = sensor
                break

        if zed_sensor is None:
            return

        try:
            K = np.asarray(zed_sensor.intrinsic_matrix, dtype=np.float32)
        except Exception:
            return

        if K.shape != (3, 3):
            try:
                K = K.reshape(3, 3)
            except Exception:
                return

        if not self._is_valid_intrinsic_matrix(K):
            return

        traj_grp.create_dataset(key, data=K, **self.compression)

    def _is_valid_intrinsic_matrix(self, K):
        """Checks whether a camera intrinsic matrix is valid for depth backprojection."""
        try:
            K = np.asarray(K, dtype=np.float32)
        except Exception:
            return False
        if K.shape != (3, 3):
            return False
        if not np.isfinite(K).all():
            return False
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        return fx > 1e-8 and fy > 1e-8

    def _process_obs(self, obs, info):
        """
        Modifies @obs inplace for any relevant post-processing

        Args:
            obs (dict): Keyword-mapped relevant observations from the immediate env step
            info (dict): Keyword-mapped relevant information from the immediate env step
        """
        # Track step index within the current playback episode
        if not hasattr(self, "_obs_step_idx"):
            self._obs_step_idx = 0
        else:
            self._obs_step_idx += 1

        # Keep only requested modalities in recorded obs
        keep_suffixes = (
            "::depth_linear",
            "::seg_instance_id",
            "::rgb",
        )
        for k in list(obs.keys()):
            if not k.endswith(keep_suffixes):
                obs.pop(k, None)

        return obs

    def _sample_single_visual_mesh_points_pytorch3d(self, visual_mesh, n_points=128):
        """
        Sample @n_points from a single visual mesh prim in world frame using PyTorch3D.

        Returns:
            th.Tensor or None: (n_points, 3) sampled points in world frame
        """
        mesh_prim = visual_mesh.prim
        tm = mesh_prim_to_trimesh_mesh(
            mesh_prim=mesh_prim,
            include_normals=False,
            include_texcoord=False,
            world_frame=True,
        )
        if tm.vertices is None or tm.faces is None or len(tm.vertices) == 0 or len(tm.faces) == 0:
            return None

        verts = th.as_tensor(tm.vertices, dtype=th.float32)
        faces = th.as_tensor(tm.faces, dtype=th.int64)
        mesh = Meshes(verts=[verts], faces=[faces])
        sampled = sample_points_from_meshes(meshes=mesh, num_samples=n_points)[0].detach()
        return sampled

    def _sanitize_prim_name(self, name):
        """Convert arbitrary object name into a USD-safe prim name."""
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
        if len(safe) == 0:
            safe = "obj"
        if safe[0].isdigit():
            safe = f"obj_{safe}"
        return safe

    def _maybe_create_mesh_points_prim_once(self, obj_name, points_world):
        """Create sampled mesh points USD prim once per object; do not update afterwards."""
        if not self._mesh_points_visualization_enabled:
            return

        if obj_name in self._mesh_points_created:
            return

        if points_world is None or points_world.numel() == 0:
            return

        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        parent_path = "/World/mesh_sampled_points"
        if stage.GetPrimAtPath(parent_path) is None:
            lazy.pxr.UsdGeom.Xform.Define(stage, parent_path)

        prim_name = self._sanitize_prim_name(obj_name)
        prim_path = f"{parent_path}/{prim_name}"
        points_prim = lazy.pxr.UsdGeom.Points.Define(stage, prim_path)

        pts_np = points_world.detach().cpu().numpy()
        vt_points = lazy.pxr.Vt.Vec3fArray(
            [lazy.pxr.Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in pts_np]
        )
        points_prim.GetPointsAttr().Set(vt_points)

        primvars_api = lazy.pxr.UsdGeom.PrimvarsAPI(points_prim.GetPrim())
        color_primvar = primvars_api.CreatePrimvar(
            "displayColor", lazy.pxr.Sdf.ValueTypeNames.Color3fArray, lazy.pxr.UsdGeom.Tokens.vertex
        )
        width_primvar = primvars_api.CreatePrimvar(
            "widths", lazy.pxr.Sdf.ValueTypeNames.FloatArray, lazy.pxr.UsdGeom.Tokens.vertex
        )
        color_primvar.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(0.0, 1.0, 0.0)] * int(points_world.shape[0])))
        width_primvar.Set(lazy.pxr.Vt.FloatArray([0.03] * int(points_world.shape[0])))

        self._mesh_points_created.add(obj_name)

    def _get_gripper_collision_recorded_prim_paths_by_arm(self, robot, prim_entries):
        """
        Returns recorded visual-prim paths currently contacted by robot grippers, split by arm.
        Contact paths are mapped to the nearest recorded prim path from @prim_entries.

        Returns:
            dict[str, list[str]]: per-arm sorted contacted recorded prim paths
        """
        if robot is None:
            return {}

        per_arm_contact_prims = {}
        arms_to_check = list(robot.arm_names) if hasattr(robot, "arm_names") and len(robot.arm_names) > 0 else ["default"]

        for arm_name in arms_to_check:
            arm_contact_prims = set()

            # Path 1: internal gripper contact API
            if hasattr(robot, "_find_gripper_contacts"):
                try:
                    contact_prims, _ = robot._find_gripper_contacts(arm=arm_name)
                    arm_contact_prims.update(contact_prims)
                except Exception:
                    pass

            # Path 2: fallback from finger links contact_list
            try:
                if hasattr(robot, "finger_links") and arm_name in robot.finger_links:
                    for finger_link in robot.finger_links[arm_name]:
                        for c in finger_link.contact_list():
                            if isinstance(c.body0, str) and len(c.body0) > 0:
                                arm_contact_prims.add(c.body0)
                            if isinstance(c.body1, str) and len(c.body1) > 0:
                                arm_contact_prims.add(c.body1)
            except Exception:
                pass

            # Filter robot self-collision prims
            robot_prim_path = getattr(robot, "prim_path", "")
            filtered_arm = sorted(
                [
                    p
                    for p in arm_contact_prims
                    if isinstance(p, str) and len(p) > 0 and (not robot_prim_path or not p.startswith(robot_prim_path))
                ]
            )
            per_arm_contact_prims[arm_name] = filtered_arm

        recorded_prim_paths = sorted(
            [
                str(entry.get("prim_path", ""))
                for entry in prim_entries
                if isinstance(entry, dict) and len(str(entry.get("prim_path", ""))) > 0
            ],
            key=lambda x: len(x),
            reverse=True,
        )

        # Build link-prefix -> recorded prim candidates (e.g. /.../link -> /.../link/visuals/...)
        link_prefix_to_recorded = defaultdict(list)
        for rp in recorded_prim_paths:
            visual_idx = rp.find("/visuals")
            if visual_idx != -1:
                link_prefix = rp[:visual_idx]
                if len(link_prefix) > 0:
                    link_prefix_to_recorded[link_prefix].append(rp)

        for link_prefix in link_prefix_to_recorded:
            link_prefix_to_recorded[link_prefix] = sorted(set(link_prefix_to_recorded[link_prefix]))

        def _map_contact_to_recorded(contact_prim_path):
            # Direct match to recorded visual prim path subtree
            for rp in recorded_prim_paths:
                if contact_prim_path.startswith(rp):
                    return rp

            # Fallback: match by link prefix, then pick canonical first recorded prim under that link
            best_link_prefix = None
            for link_prefix in link_prefix_to_recorded.keys():
                if contact_prim_path.startswith(link_prefix):
                    if best_link_prefix is None or len(link_prefix) > len(best_link_prefix):
                        best_link_prefix = link_prefix
            if best_link_prefix is not None:
                candidates = link_prefix_to_recorded[best_link_prefix]
                if len(candidates) > 0:
                    return candidates[0]

            return None

        mapped = {}
        for arm_name, prims in per_arm_contact_prims.items():
            arm_recorded_prim_paths = []
            if prims is None:
                mapped[arm_name] = arm_recorded_prim_paths
                continue

            for prim_path in prims:
                if not isinstance(prim_path, str) or len(prim_path) == 0:
                    continue
                match_prim_path = _map_contact_to_recorded(prim_path)
                if match_prim_path is not None and match_prim_path not in arm_recorded_prim_paths:
                    arm_recorded_prim_paths.append(match_prim_path)

            mapped[arm_name] = sorted(arm_recorded_prim_paths)

        return mapped

    def _get_scene_visual_mesh_prim_alias_to_index(self, prim_entries):
        """Returns alias lookup for seg prim paths to recorded rigid visual mesh prim indices."""
        alias_to_index = {}
        parent_counts = defaultdict(int)
        for entry in prim_entries:
            parent_counts[entry["prim_parent_path"]] += 1

        for idx, entry in enumerate(prim_entries):
            alias_to_index[entry["prim_path"]] = idx
            if parent_counts[entry["prim_parent_path"]] == 1:
                alias_to_index.setdefault(entry["prim_parent_path"], idx)

        return alias_to_index

    def _extract_seg_instance_id_info(self, info):
        """Best-effort retrieval of seg_instance_id info dict from nested observation info."""

        def _search(node):
            if not isinstance(node, dict):
                return None

            direct = node.get("seg_instance_id", None)
            if isinstance(direct, dict):
                return direct

            for value in node.values():
                if isinstance(value, dict):
                    found = _search(value)
                    if found is not None:
                        return found

            return None

        return _search(info.get("obs_info", {})) if isinstance(info, dict) else None

    def _resolve_instance_label_to_prim_path(self, label):
        if not isinstance(label, str) or len(label) == 0:
            return None

        if "/" in label:
            return label

        if label == "groundPlane":
            floor_plane = getattr(og.sim, "floor_plane", None)
            return getattr(floor_plane, "prim_path", None) if floor_plane is not None else None

        scene = getattr(self, "scene", None)
        if scene is not None and hasattr(scene, "object_registry"):
            obj = scene.object_registry("name", value=label)
            if obj is not None:
                return getattr(obj, "prim_path", None)

        return None

    def _build_seg_instance_id_prim_mapping(self, info, prim_entries):
        """
        Build a mapping from seg_instance_id to prim path / matched rigid visual prim metadata.

        Returns:
            dict[str, dict]: Mapping like
                {
                    "47": {
                        "seg_prim_path": "/World/.../visuals",
                        "prim_path": "/World/.../visuals/mesh_0",
                        "prim_index": 0,
                    },
                    ...
                }
        """
        seg_instance_id_info = self._extract_seg_instance_id_info(info)
        if not isinstance(seg_instance_id_info, dict) or len(seg_instance_id_info) == 0:
            return {}

        alias_to_index = self._get_scene_visual_mesh_prim_alias_to_index(prim_entries)
        mapping = {}
        for raw_id, raw_prim_path in seg_instance_id_info.items():
            try:
                inst_id = int(raw_id)
            except Exception:
                continue

            seg_prim_path = str(raw_prim_path)
            prim_index = alias_to_index.get(seg_prim_path, -1)
            matched_entry = prim_entries[prim_index] if prim_index >= 0 and prim_index < len(prim_entries) else None

            mapping[str(inst_id)] = {
                "seg_prim_path": seg_prim_path,
                "prim_path": matched_entry["prim_path"] if matched_entry is not None else None,
                "prim_index": prim_index,
                "object_name": matched_entry["object_name"] if matched_entry is not None else None,
                "link_name": matched_entry["link_name"] if matched_entry is not None else None,
            }

        return mapping

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store action, obs, reward, terminated, truncated, info
        step_data = dict()
        step_data["obs"] = self._process_obs(obs=obs, info=info)
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated

        seg_instance_id_keys = [k for k in step_data["obs"].keys() if str(k).endswith("::seg_instance_id")]
        prim_entries = self._get_scene_visual_mesh_prim_entries() if len(seg_instance_id_keys) > 0 else []
        step_data["seg_instance_id_prim_mapping"] = (
            self._build_seg_instance_id_prim_mapping(info=info, prim_entries=prim_entries)
            if len(seg_instance_id_keys) > 0
            else {}
        )

        record_gripper_contacts = os.getenv("OG_RECORD_GRIPPER_CONTACTS", "1") == "1"

        # Get zed camera world pose (for expressing object pose in segmentation-camera frame)
        zed_cam_pos = None
        zed_cam_quat = None
        zed_sensor = None
        robot = self.env.robots[0] if len(self.env.robots) > 0 else None
        if robot is not None and hasattr(robot, "sensors"):
            for sensor in robot.sensors.values():
                prim_path = getattr(sensor, "prim_path", "")
                if "zed_link" in str(prim_path) and "Camera" in str(prim_path) and hasattr(sensor, "get_position_orientation"):
                    zed_sensor = sensor
                    zed_cam_pos, zed_cam_quat = sensor.get_position_orientation()
                    break
        if zed_cam_pos is None or zed_cam_quat is None:
            if robot is not None:
                zed_cam_pos, zed_cam_quat = robot.get_position_orientation()
            else:
                zed_cam_pos, zed_cam_quat = th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

        zed_cam_pos = th.as_tensor(zed_cam_pos, dtype=th.float32)
        zed_cam_quat = th.as_tensor(zed_cam_quat, dtype=th.float32)
        # Save camera pose into step data (world-frame position + quaternion)
        step_cam_pos = zed_cam_pos
        step_cam_quat = zed_cam_quat

        # Attach camera pose to step data so it's saved per-frame
        step_data["zed_camera_position"] = step_cam_pos
        step_data["zed_camera_orientation"] = step_cam_quat

        # Save per-step camera intrinsics (safe_playback uses sensor.intrinsic_matrix during stepping)
        if zed_sensor is not None and hasattr(zed_sensor, "intrinsic_matrix"):
            try:
                K_step = np.asarray(zed_sensor.intrinsic_matrix, dtype=np.float32)
                if K_step.shape != (3, 3):
                    K_step = K_step.reshape(3, 3)
                if self._is_valid_intrinsic_matrix(K_step):
                    step_data["zed_camera_intrinsic_step"] = th.as_tensor(K_step, dtype=th.float32)
            except Exception:
                pass

        # Record rigid visual prim poses
        if len(prim_entries) == 0:
            prim_entries = self._get_scene_visual_mesh_prim_entries()
        prim_positions = []
        prim_orientations = []
        prim_positions_in_zed_camera = []
        prim_orientations_in_zed_camera = []
        for entry in prim_entries:
            pos, quat = PoseAPI.get_world_pose(entry["prim_path"])

            # Express prim pose in zed camera frame: T_cam_prim = inv(T_world_cam) @ T_world_prim
            rel_pos, rel_quat = T.relative_pose_transform(
                pos1=pos,
                quat1=quat,
                pos0=zed_cam_pos,
                quat0=zed_cam_quat,
            )

            prim_positions.append(pos)
            prim_orientations.append(quat)
            prim_positions_in_zed_camera.append(rel_pos)
            prim_orientations_in_zed_camera.append(rel_quat)

        # Keep this field tensor-typed so flush / hdf5 allocation paths work
        # (string lists do not have .shape / .numpy())
        step_data["prim_indices"] = th.arange(len(prim_entries), dtype=th.int32)
        step_data["prim_positions"] = th.stack(prim_positions) if prim_positions else th.zeros((0, 3))
        step_data["prim_orientations"] = th.stack(prim_orientations) if prim_orientations else th.zeros((0, 4))
        step_data["prim_positions_in_zed_camera"] = (
            th.stack(prim_positions_in_zed_camera) if prim_positions_in_zed_camera else th.zeros((0, 3))
        )
        step_data["prim_orientations_in_zed_camera"] = (
            th.stack(prim_orientations_in_zed_camera) if prim_orientations_in_zed_camera else th.zeros((0, 4))
        )

        # Record gripper contacts in fixed [left, right] format (T x 2 over time), value is recorded prim_path
        if record_gripper_contacts:
            gripper_contact_prim_paths_by_arm = self._get_gripper_collision_recorded_prim_paths_by_arm(
                robot=robot,
                prim_entries=prim_entries,
            )

            arm_names = list(getattr(robot, "arm_names", [])) if robot is not None else []
            if "left" in arm_names and "right" in arm_names:
                arm_order = ["left", "right"]
            elif len(arm_names) >= 2:
                arm_order = arm_names[:2]
            elif len(arm_names) == 1:
                arm_order = [arm_names[0], arm_names[0]]
            else:
                arm_order = ["left", "right"]

            left_or_first = gripper_contact_prim_paths_by_arm.get(arm_order[0], [])
            right_or_second = gripper_contact_prim_paths_by_arm.get(arm_order[1], [])

            step_data["gripper_contact_prim_paths_lr"] = [
                left_or_first[0] if len(left_or_first) > 0 else "",
                right_or_second[0] if len(right_or_second) > 0 else "",
            ]

        return step_data

    def playback_episode(self, episode_id, record_data=True, video_writers=None):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
            video_writers (Any): Optional video writers to record the playback
        """
        if video_writers is None and self.enable_rgb_recording:
            video_writers = self._get_default_video_writers()

        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Grab episode data
        # Skip early if found malformed data
        try:
            transitions = json.loads(traj_grp.attrs["transitions"])
            traj_grp = h5py_group_to_torch(traj_grp)
            init_metadata = traj_grp["init_metadata"]
            action = traj_grp["action"]
            state = traj_grp["state"]
            state_size = traj_grp["state_size"]
            reward = traj_grp["reward"]
            terminated = traj_grp["terminated"]
            truncated = traj_grp["truncated"]
        except KeyError as e:
            print(f"Got error when trying to load episode {episode_id}:")
            print(f"Error: {str(e)}")
            return

        # Reset environment and update this to be the new initial state
        self.scene.restore(self.scene_file, update_initial_file=True)

        # Reset object attributes from the stored metadata
        with og.sim.stopped():
            for attr, vals in init_metadata.items():
                assert len(vals) == self.scene.n_objects
            for i, obj in enumerate(self.scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)
        self.reset()

        # If not controlling robots, disable for all robots
        if not self.include_robot_control:
            for robot in self.robots:
                robot.control_enabled = False
                # Set all controllers to effort mode with zero gain, this keeps the robot still
                for controller in robot.controllers.values():
                    for i, dof in enumerate(controller.dof_idx):
                        dof_joint = robot.joints[robot.dof_names_ordered[dof]]
                        dof_joint.set_control_type(
                            control_type=ControlType.EFFORT,
                            kp=None,
                            kd=None,
                        )

        # Restore to initial state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)

        # If record, record initial observations
        if record_data:
            # We need to step the environment to get the initial observations propagated
            first_time_load_n_iteration = 10
            self.current_obs, _, _, _, init_info = self.env.step(
                action=action[0], n_render_iterations=self.n_render_iterations + first_time_load_n_iteration
            )
            init_obs = self._process_obs(obs=self.current_obs, info=init_info)
            step_data = {"obs": init_obs}

            init_seg_info = None
            if isinstance(init_info, dict):
                init_seg_info = self._build_seg_instance_id_prim_mapping(
                    info=init_info,
                    prim_entries=self._get_scene_visual_mesh_prim_entries(),
                )

            if isinstance(init_seg_info, dict):
                step_data["seg_instance_id_prim_mapping"] = init_seg_info

            self.current_traj_history.append(step_data)

        for i, (a, s, ss, r, te, tr) in enumerate(
            zip(action, state[1:], state_size[1:], reward, terminated, truncated)
        ):
            # Execute any transitions that should occur at this current step
            if str(i) in transitions:
                cur_transitions = transitions[str(i)]
                scene = og.sim.scenes[0]
                for add_sys_name in cur_transitions["systems"]["add"]:
                    scene.get_system(add_sys_name, force_init=True)
                for remove_sys_name in cur_transitions["systems"]["remove"]:
                    scene.clear_system(remove_sys_name)
                for remove_obj_name in cur_transitions["objects"]["remove"]:
                    obj = scene.object_registry("name", remove_obj_name)
                    scene.remove_object(obj)
                for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                    obj = create_object_from_init_info(add_obj_info)
                    scene.add_object(obj)
                    obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                # Step physics to initialize any new objects
                og.sim.step()

            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            og.sim.load_state(s[: int(ss)], serialized=True)
            if not self.include_contacts:
                # When all objects/systems are visual-only, keep them still on every step
                for obj in self.scene.objects:
                    obj.keep_still()
                for system in self.scene.systems:
                    # TODO: Implement keep_still for other systems
                    if isinstance(system, MacroPhysicalParticleSystem):
                        system.set_particles_velocities(
                            lin_vels=th.zeros((system.n_particles, 3)), ang_vels=th.zeros((system.n_particles, 3))
                        )
            self.current_obs, _, _, _, info = self.env.step(action=a, n_render_iterations=self.n_render_iterations)

            # If recording, record data
            if record_data:
                step_data = self._parse_step_data(
                    action=a,
                    obs=self.current_obs,
                    reward=r,
                    terminated=te,
                    truncated=tr,
                    info=info,
                )
                if self.flush_every_n_steps > 0:
                    if i == 0:
                        alloc_step_data = dict(step_data)
                        if len(self.current_traj_history) > 0:
                            init_step_data = self.current_traj_history[0]
                            # Ensure obs schema comes from initial propagated frame,
                            # while keeping other fields from parsed step_data
                            if "obs" in init_step_data:
                                alloc_step_data["obs"] = init_step_data["obs"]
                        self.current_traj_grp, self.traj_dsets = self.allocate_traj_to_hdf5(
                            alloc_step_data, f"demo_{episode_id}", num_samples=len(action), video_writers=video_writers
                        )
                    if i % self.flush_every_n_steps == 0:
                        self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
                # append to current trajectory history
                self.current_traj_history.append(step_data)

            self.current_episode_step_count += 1
            self.step_count += 1

        if record_data:
            if self.flush_every_n_steps > 0:
                self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
            self.flush_current_traj()

    def playback_dataset(self, record_data=False):
        """
        Playback all episodes from the input HDF5 file, and optionally record observation data if @record is True

        Args:
            record_data (bool): Whether to record data during playback or not
        """
        for episode_id in range(self.input_hdf5["data"].attrs["n_episodes"]):
            self.playback_episode(
                episode_id=episode_id,
                record_data=record_data,
            )

    def allocate_traj_to_hdf5(
        self, step_data, traj_grp_name, num_samples: int, nested_keys=("obs",), data_grp=None, video_writers=None
    ):
        """
        Allocate trajectory data space from @step_data given the number of samples @num_samples.

        Args:
            step_data (dict): Keyword-mapped set of data for a single sim step
            traj_grp_name (str): Name of the trajectory group to store
            num_samples (int): Number of samples in the trajectory
            nested_keys (list of str): Name of key(s) corresponding to nested data in @step_data. This specific data
                is assumed to be its own keyword-mapped dictionary of numpy array values, and will be parsed
                differently from the rest of the data.
            data_grp (None or h5py.Group): If specified, the h5py Group under which a new group wtih name
                @traj_grp_name will be created. If None, will default to "data" group
            video_writers (None or dict): If specified, a dictionary mapping observation keys to video writers
                for saving video frames during replay

        Returns:
            Tuple[h5py.Group, dict(str, hdf5.Dataset)]: Generated hdf5 group and datasets to store the trajectory data in the future
        """
        traj_dsets = dict()
        nested_keys = set(nested_keys)
        for k in nested_keys:
            traj_dsets[k] = dict()
        data_grp = self.hdf5_file.require_group("data") if data_grp is None else data_grp
        traj_grp = data_grp.create_group(traj_grp_name)
        log.info(f"Number of samples: {num_samples}")
        traj_grp.attrs["num_samples"] = num_samples

        for k, dat in step_data.items():
            if k in nested_keys:
                obs_grp = traj_grp.create_group(k)
                for mod, step_mod_data in dat.items():
                    if video_writers is None or mod not in video_writers.keys():
                        traj_dsets[k][mod] = obs_grp.create_dataset(
                            mod,
                            shape=(num_samples, *step_mod_data.shape),
                            dtype=step_mod_data.numpy().dtype,
                            **self.compression,
                            chunks=(1, *step_mod_data.shape),
                            shuffle=True,
                        )
                    else:
                        log.info(f"Skipping storing {mod} in h5, writing to video instead.")
            else:
                if k == "gripper_contact_prim_paths_lr":
                    traj_dsets[k] = traj_grp.create_dataset(
                        k,
                        shape=(num_samples, 2),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                    continue
                if isinstance(dat, th.Tensor):
                    traj_dsets[k] = traj_grp.create_dataset(
                        k, shape=(num_samples, *dat.shape), dtype=dat.numpy().dtype, **self.compression, shuffle=True
                    )
                else:
                    try:
                        dat_tensor = th.as_tensor(dat)
                        traj_dsets[k] = traj_grp.create_dataset(
                            k,
                            shape=(num_samples, *dat_tensor.shape),
                            dtype=dat_tensor.numpy().dtype,
                            **self.compression,
                            shuffle=True,
                        )
                    except Exception:
                        traj_dsets[k] = traj_grp.create_dataset(
                            k,
                            shape=(num_samples,),
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        )

        return traj_grp, traj_dsets

    def flush_partial_traj(self, num_samples: int, video_writers=None):
        """
        Flush the current trajectory data to file.
        If flush_every_n_steps is greater than 0, flush the current trajectory data to file every n steps.
        Args:
            num_samples: (int): The number of samples to flush.
            video_writers: (None or dict): If specified, a dictionary mapping observation keys to video writers
                for saving video frames during replay
        """
        log.info(f"Storing partial trajectory at step {self.current_episode_step_count}...")
        assert self.flush_every_n_steps > 0, "flush_every_n_steps must be greater than 0 to flush partial trajectory"
        data_length_to_flush = len(self.current_traj_history)
        # At step 0, we only have observation data, so observation data will only have one more offset than others
        if self.current_episode_step_count == 0:
            assert data_length_to_flush == 1
            for key, dat in self.current_traj_history[0].items():
                if not isinstance(dat, dict):
                    continue
                for mod in dat.keys():
                    if video_writers is not None and mod in video_writers.keys():
                        assert (
                            write_video is not None
                        ), "video_writers not imported! Please make sure you have omnigibson setup with eval dependencies!"
                        # write to video
                        write_video(
                            self.current_traj_history[0][key][mod].unsqueeze(0).numpy(),
                            video_writer=video_writers[mod],
                            batch_size=None,
                            mode=mod.split("::")[-1],
                        )
                    else:
                        if (
                            key in self.traj_dsets
                            and isinstance(self.traj_dsets[key], dict)
                            and mod in self.traj_dsets[key]
                        ):
                            self.traj_dsets[key][mod][0] = self.current_traj_history[0][key][mod]
        else:
            for key, dat in self.current_traj_history[0].items():
                if (
                    isinstance(dat, dict)
                    and key in self.traj_dsets
                    and isinstance(self.traj_dsets[key], dict)
                ):
                    for mod in dat.keys():
                        obs_data_length = (
                            data_length_to_flush
                            if self.current_episode_step_count < num_samples
                            else data_length_to_flush - 1
                        )
                        if obs_data_length > 0:
                            if not all(mod in self.current_traj_history[i][key] for i in range(obs_data_length)):
                                continue
                            mod_values = [self.current_traj_history[i][key][mod] for i in range(obs_data_length)]
                            if not all(v is not None for v in mod_values):
                                continue
                            data_to_write = th.stack(
                                [v if isinstance(v, th.Tensor) else th.as_tensor(v) for v in mod_values], dim=0
                            )
                            if video_writers is not None and mod in video_writers.keys():
                                assert (
                                    write_video is not None
                                ), "video_writers not imported! Please make sure you have omnigibson setup with eval dependencies!"
                                # write to video
                                write_video(
                                    data_to_write.numpy(),
                                    video_writer=video_writers[mod],
                                    batch_size=None,
                                    mode=mod.split("::")[-1],
                                )
                            else:
                                if (
                                    key in self.traj_dsets
                                    and isinstance(self.traj_dsets[key], dict)
                                    and mod in self.traj_dsets[key]
                                ):
                                    self.traj_dsets[key][mod][
                                        self.current_episode_step_count
                                        - data_length_to_flush
                                        + 1 : self.current_episode_step_count + 1
                                    ] = data_to_write
                else:
                    if key in self.traj_dsets:
                        values = [self.current_traj_history[i][key] for i in range(data_length_to_flush)]
                        write_slice = slice(
                            self.current_episode_step_count - data_length_to_flush,
                            self.current_episode_step_count,
                        )
                        dset = self.traj_dsets[key]

                        if key == "gripper_contact_prim_paths_lr":
                            rows = np.asarray(
                                [["" if v is None else str(v) for v in row] for row in values],
                                dtype=object,
                            )
                            dset[write_slice, :] = rows
                            continue

                        if h5py.check_string_dtype(dset.dtype) is not None:
                            serialized = [
                                v if isinstance(v, str) else json.dumps(v, cls=TorchEncoder)
                                for v in values
                            ]
                            dset[write_slice] = serialized
                        else:
                            tensor_values = [v if isinstance(v, th.Tensor) else th.as_tensor(v) for v in values]
                            dset[write_slice] = th.stack(tensor_values, dim=0)
        # Reset the current trajectory history
        self.current_traj_history = []

    def flush_current_traj(self):
        """
        Flush current trajectory data
        For playback, we assume that all data needs to be stored.
        """
        if self.flush_every_n_steps == 0:
            super().flush_current_traj()
        else:
            self.postprocess_traj_group(self.current_traj_grp)
            self.flush_current_file()
            # Clear trajectory and transition buffers
            self.traj_count += 1
            self.current_episode_step_count = 0
            self.current_traj_history = []

    def save_data(self):
        try:
            super().save_data()
        finally:
            self._close_default_video_writers()
