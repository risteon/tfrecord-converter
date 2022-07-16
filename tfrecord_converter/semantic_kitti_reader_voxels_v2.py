#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import pathlib
import typing
import logging

import tensorflow as tf
import numpy as np
from ruamel.yaml import YAML

from nuscenes.utils.splits import train as nuscenes_split_train
from nuscenes.utils.splits import val as nuscenes_split_val

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class SemanticKittiReaderVoxelsV2:
    """
    Convert SemanticKITTI data:
    * point clouds (KITTI odometry version: ego-motion corrected)
    * semantic labels
    * voxel occupied/invalid/label
    * accumulated point clouds


    kitti_odometry_root:

    """

    def __init__(
        self,
        kitti_odometry_sequences: str,
        semantic_kitti_sequences: str,
        semantic_kitti_voxels_sequences: str,
        input_data_version: str = "nuscenes",
    ):
        if input_data_version not in ["kitti", "nuscenes"]:
            raise ValueError(f"Unknown input format '{input_data_version}.")

        self.kitti_odometry_sequences = pathlib.Path(kitti_odometry_sequences)
        self.semantic_kitti_sequences = pathlib.Path(semantic_kitti_sequences)

        self.semantic_kitti_voxels_sequences = pathlib.Path(
            semantic_kitti_voxels_sequences
        )

        if input_data_version == "kitti":
            self.config_semantic: pathlib.Path = (
                pathlib.Path(__file__).parent.parent / "config" / "semantic-kitti.yaml"
            )
        elif input_data_version == "nuscenes":
            self.config_semantic: pathlib.Path = (
                pathlib.Path(__file__).parent.parent
                / "config"
                / "nuscenes-lidarseg.yaml"
            )
        else:
            assert False

        if not self.kitti_odometry_sequences.is_dir():
            raise RuntimeError(
                "Could not find semantic kitti odometry root folder '{}'".format(
                    str(self.kitti_odometry_sequences)
                )
            )
        if not self.semantic_kitti_sequences.is_dir():
            raise RuntimeError(
                "Could not find semantic kitti root folder '{}'".format(
                    str(self.semantic_kitti_sequences)
                )
            )
        if not self.semantic_kitti_voxels_sequences.is_dir():
            raise RuntimeError(
                "Could not find semantic kitti voxels root folder '{}'".format(
                    str(self.semantic_kitti_voxels_sequences)
                )
            )

        if not self.config_semantic.is_file():
            raise RuntimeError(
                "Could not find kitti semantic config file '{}'".format(
                    str(self.config_semantic)
                )
            )

        self.testset_flag = False

        self._split = None
        self._config_data = None
        self._samples_to_generate = None
        self._label_mapping = None
        self._voxel_data_cache = {}
        # these points will be kept for geometry training. Unlabeled but useful points.
        self._unlabeled_index = None
        # these points will be removed also for geometry training
        self._noise_index = None

        self._make_split(input_data_version)

    def __iter__(self):
        return self._samples_to_generate.__iter__()

    @property
    def split(self):
        return self._split

    def _make_split(self, split_version):
        """
        Use generated <self.voxel_version> output to build split.
        split_version: define how to make train/val
        """

        self.sample_id_template = f"{split_version}_kitti_{{seq:02d}}_{{frame:04d}}"

        if split_version == "kitti":
            self._seq_format = lambda x: "{:02d}".format(x)
            self._label_format = lambda x: "{:06d}".format(x)
            self._frame_format = lambda x: "{:06d}".format(x)
        elif split_version == "nuscenes":
            self._seq_format = lambda x: "{:04d}".format(x)
            # TODO(risteon) Fix input. 6 digits would be more consistent
            self._label_format = lambda x: "{:05d}".format(x)
            self._frame_format = lambda x: "{:05d}".format(x)
        else:
            assert False

        # format for files from the voxelizer
        self._voxel_format = lambda x: "{:06d}".format(x)

        # Todo: no test split option for now
        assert self.testset_flag is False
        # read config
        with open(str(self.config_semantic), "r") as file_conf_sem:
            yaml = YAML()
            data = yaml.load(file_conf_sem)
            self._config_data = {k: dict(v) for k, v in data.items()}

        # figure out if there is an unlabeled class
        label_names = self._config_data["labels"]
        try:
            self._unlabeled_index = next(k for k, v in label_names.items() if v == "unlabeled")
        except StopIteration:
            raise ValueError("No unlabeled label found. Check if this is intended.")
        try:
            self._noise_index = next(k for k, v in label_names.items() if v == "noise")
        except StopIteration:
            pass

        if split_version == "kitti":
            valid_splits = ["train", "valid"]
            map_split_names = {"train": "train", "valid": "val", "test": "test"}
            # config file holds kitti split
            data_splits = {
                map_split_names[k]: v
                for k, v in self._config_data["split"].items()
                if k in valid_splits
            }

        elif split_version == "nuscenes":
            # sequence folders are 4-digits on disk.
            def parse_nuscenes_scene_name(x):
                if len(x) != 10:
                    raise ValueError(f"Unexpected nuscenes scene {x}.")
                return int(x[-4:])

            # Translate nuscenes scene names 'scene-XXXX' from split into these integers
            data_splits = {"train": nuscenes_split_train, "val": nuscenes_split_val}
            data_splits = {
                k: sorted([parse_nuscenes_scene_name(x) for x in v])
                for k, v in data_splits.items()
            }
        else:
            raise NotImplementedError(f"Unknown split {split_version}.")

        self._split = {
            "name": "semantic_kitti_voxels_{}".format(
                split_version if not self.testset_flag else f"test_{split_version}"
            ),
            "data": {k: [] for k in data_splits.keys()},
        }

        self._samples_to_generate = []

        def parse_sequence_folder_name(x):
            try:
                return int(x)
            except ValueError:
                # will not be in sequence split and therefore skipped
                return -1

        voxel_sequences = {
            parse_sequence_folder_name(x.name): x
            for x in self.semantic_kitti_voxels_sequences.iterdir()
        }
        kitti_input_sequences = {
            parse_sequence_folder_name(x.name): x
            for x in self.semantic_kitti_sequences.iterdir()
        }

        for split_name, sequences in data_splits.items():
            split_data = self._split["data"][split_name]
            for sequence_index in sequences:
                if not self.testset_flag:

                    # for completeness, also check if sequence available in KITTI input folder.
                    # This can be removed when this folder is not necessary at all.
                    if sequence_index not in kitti_input_sequences:
                        logger.warning(
                            "Sequence {:02d} not available in KITTI folder. Skipping.".format(
                                sequence_index
                            )
                        )
                        continue

                    if sequence_index not in voxel_sequences:
                        logger.warning(
                            "Sequence {:02d} not available. Skipping.".format(
                                sequence_index
                            )
                        )
                        continue

                    voxel_dir = voxel_sequences[sequence_index]
                    if not voxel_dir.is_dir():
                        logger.warning(
                            "Voxels not available in sequence {:02d}. Skipping.".format(
                                sequence_index
                            )
                        )
                        continue

                    self._voxel_data_cache[sequence_index] = {
                        int(x.stem[:6]): x
                        for x in (voxel_dir).iterdir()
                        if x.suffix == ".tfrecord"
                    }

                    split_data.extend(
                        [
                            self.sample_id_template.format(seq=sequence_index, frame=x)
                            for x in sorted(
                                list(self._voxel_data_cache[sequence_index].keys())
                            )
                        ]
                    )
                    self._samples_to_generate.extend(
                        [
                            (sequence_index, x)
                            for x in sorted(
                                list(self._voxel_data_cache[sequence_index].keys())
                            )
                        ]
                    )
                else:
                    raise NotImplementedError()

        self._label_mapping: dict = self._config_data["learning_map"]
        # make 255 the 'unlabeled' label and shift all others down (-1) accordingly
        self._label_mapping = {
            k: v - 1 if v != 0 else 255 for k, v in self._label_mapping.items()
        }
        self._label_mapping_voxels = self._label_mapping.copy()
        # map unlabeled to extra entry 254 when voxelizing
        # Todo(risteon): Is this better?
        # -> Map noise to 254, this will get removed when parsing
        # -> Map unlabeled to 255 to keep for geometry training
        self._label_mapping[self._unlabeled_index] = 255
        self._label_mapping_voxels[self._unlabeled_index] = 255
        if self._noise_index is not None:
            self._label_mapping_voxels[self._noise_index] = 254

        assert all(x <= 255 for x in self._label_mapping.values())
        assert all(x <= 255 for x in self._label_mapping_voxels.values())

        self._label_mapping = np.vectorize(self._label_mapping.get, otypes=[np.int64])
        self._label_mapping_voxels = np.vectorize(
            self._label_mapping_voxels.get, otypes=[np.int64]
        )

    def make_sample_id(self, sample: typing.Tuple[int, int]):
        if not self.testset_flag:
            return self.sample_id_template.format(seq=sample[0], frame=sample[1])
        else:
            raise NotImplementedError()

    def read(self, sample: typing.Tuple[int, int], sample_id: str) -> {str: typing.Any}:

        r = {"sample_id": sample_id.encode("utf-8")}

        if not self.testset_flag:

            sequence_str = self._seq_format(sample[0])
            voxel_str = self._voxel_format(sample[1])

            voxel_base = self.semantic_kitti_voxels_sequences / sequence_str
            voxel_data = self.read_semantic_kitti_voxel_data(
                voxel_base / voxel_str,
                unpack_compressed=False,
            )

            # contains accumulated point cloud and labels
            proto_file_accumulated = voxel_base / (voxel_str + "_points.tfrecord")
            proto_data_voxel = self.read_proto_file_voxel_data(proto_file_accumulated)

            # contains input frame and labels (e.g. single frame for KITTI)
            proto_file_point_input = voxel_base / (voxel_str + "_input.tfrecord")
            proto_data_input = self.read_proto_file_input_data(proto_file_point_input)

        else:
            raise NotImplementedError()

        # #####
        # the dynamic occlusion mask contains the actual input frame of the object
        # -> remove input voxels from dynamic occlusion mask
        # ## Accumulated points ("points") contain occupied voxels. But include dynamic objects
        # ## also only for first (aka "input") frame(s). So all statically occupied
        # ## voxels do not need to be considered for the dyn occlusion map.
        # ## This map only filters out free space points for later frames.
        dynamic_occlusion = np.bitwise_and(
            voxel_data["dynamic_occlusion"],
            np.bitwise_not(self.packbits_kitti(voxel_data["points"] > 0)),
        )

        # We actually do not need the voxelized accumulated point cloud.
        # r["voxel_points"] = voxel_data["points"].tobytes()
        r["voxel_free"] = voxel_data["free"].tobytes()
        r["voxel_dynamic_occlusion"] = dynamic_occlusion.tobytes()
        r.update(**proto_data_voxel)
        r.update(**proto_data_input)
        return r

    @staticmethod
    def read_pointcloud(filepath):
        scan = np.fromfile(str(filepath), dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    @staticmethod
    def read_label(filepath):
        label = np.fromfile(str(filepath), dtype=np.uint32)
        label = label.reshape((-1))

        # only fill in attribute if the right size
        label_sem = label & 0xFFFF  # semantic label in lower half
        label_inst = label >> 16  # instance id in upper half

        # sanity check
        assert (label_sem + (label_inst << 16) == label).all()
        return label_sem, label_inst

    @staticmethod
    def unpack(compressed: np.ndarray):
        assert compressed.ndim == 1
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.bool)
        for b in range(8):
            uncompressed[b::8] = compressed >> (7 - b) & 1
        return uncompressed

    @staticmethod
    def packbits_kitti(uncompressed: np.ndarray):
        """Semantic KITTI uses 'little' bitorder.
        Array entries [A, B, C, D, ...] are writtin in this order into bits:
        0bABCD...

        :param uncompressed:
        :return:
        """
        return np.packbits(uncompressed, bitorder="little")

    @staticmethod
    def read_semantic_kitti_voxel_data(
        semantic_kitti_sample: pathlib.Path,
        unpack_compressed: bool = False,
    ) -> {str: np.ndarray}:
        # compressed/uncompressed (compressed means 8 booleans are packed into one byte)
        d = {
            "dynamic_occlusion": True,
            "free": False,  # < number of free space points in a voxel [0-255]
            "points": False,  # < number of lidar points in a voxel [0-255]
        }

        data = {}
        for k, compressed in d.items():
            filepath = semantic_kitti_sample.parent / (
                semantic_kitti_sample.stem + "." + k
            )
            if not filepath.is_file():
                raise FileNotFoundError("Cannot find voxel label file '{}'.".format(k))

            if compressed:
                x = np.fromfile(str(filepath), dtype=np.uint8)
                if unpack_compressed:
                    x = SemanticKittiReaderVoxelsV2.unpack(x)

            else:
                x = np.fromfile(str(filepath), dtype=np.uint8)

            data[k] = x
        return data

    def read_proto_file_voxel_data(self, proto_file: pathlib.Path):
        proto_bytes = open(proto_file, "rb").read()
        example = tf.train.Example()
        example.ParseFromString(proto_bytes)

        points = np.array(
            example.features.feature["points"].float_list.value, np.float32
        ).reshape((-1, 3))
        transforms = np.array(
            example.features.feature["transforms"].float_list.value, np.float32
        ).reshape((-1, 4, 4))
        splits = np.array(example.features.feature["splits"].int64_list.value, np.int64)
        if splits[-1] != points.shape[0] or np.any(np.ediff1d(splits) < 0):
            raise ValueError(
                "Corrupted data in {}: Invalid splits.".format(str(proto_file))
            )

        scan_idx_min_max = np.array(
            example.features.feature["scan_idx_min_max"].int64_list.value, np.int64
        )

        if transforms.shape[0] != scan_idx_min_max[1] - scan_idx_min_max[0]:
            raise ValueError(
                "Corrupted data in {}: Number of frames.".format(str(proto_file))
            )

        label_bytes = example.features.feature["labels"].bytes_list.value[0]
        labels = tf.io.decode_raw(
            label_bytes, out_type=tf.dtypes.uint16, little_endian=True
        )
        try:
            labels = self._label_mapping_voxels(labels)
        except TypeError:
            raise RuntimeError(
                "Invalid label entry in accumulated label data '{}'.".format(
                    str(proto_file)
                )
            )

        return {
            "lidar_accumulated": points.reshape((-1,)),
            "transforms": transforms.reshape((-1,)),
            "splits": splits,
            "scan_idx_min_max": scan_idx_min_max,
            "labels_accumulated": labels.astype(np.uint8).tobytes(),
        }

    def read_proto_file_input_data(self, proto_file: pathlib.Path):
        """The expected keys in the input proto file are:
        * lidar: Input points with remission. 4 floats per point
        * splits: Ints to split point list into frames
        * transforms: tf between reference frame and point. Similar to voxelized tfs

        :param proto_file:
        :return:
        """
        proto_bytes = open(proto_file, "rb").read()
        example = tf.train.Example()
        example.ParseFromString(proto_bytes)

        points = np.array(
            example.features.feature["lidar"].float_list.value, np.float32
        ).reshape((-1, 4))
        transforms = np.array(
            example.features.feature["transforms"].float_list.value, np.float32
        ).reshape((-1, 4, 4))
        splits = np.array(example.features.feature["splits"].int64_list.value, np.int64)
        if splits[-1] != points.shape[0] or np.any(np.ediff1d(splits) < 0):
            raise ValueError(
                "Corrupted data in {}: Invalid splits.".format(str(proto_file))
            )

        # Pointwise semantic labels
        label_bytes = example.features.feature["labels"].bytes_list.value[0]
        labels = tf.io.decode_raw(
            label_bytes, out_type=tf.dtypes.uint16, little_endian=True
        )
        try:
            labels = self._label_mapping(labels)
        except TypeError:
            raise RuntimeError(
                "Invalid label entry in accumulated label data '{}'.".format(
                    str(proto_file)
                )
            )

        if len(points) != len(labels):
            raise ValueError("Invalid labels. Length mismatch.")

        return {
            "point_cloud": points.reshape((-1,)),
            "point_cloud_transforms": transforms.reshape((-1,)),
            "point_cloud_splits": splits,
            "semantic_labels": labels.astype(np.uint8).tobytes(),
        }
