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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class SemanticKittiReaderVoxels:
    """
    Convert SemanticKITTI data:
    * point clouds (KITTI odometry version: ego-motion corrected)
    * semantic labels
    * voxel occupied/invalid/label
    * accumulated point clouds

    """

    sample_id_template = "semantic_kitti_{seq:02d}_{frame:04d}"

    def __init__(
        self,
        kitti_odometry_root: str,
        semantic_kitti_root: str,
        semantic_kitti_voxels_root: str,
        voxel_version: str = "voxels_v2",
    ):
        self.kitti_odometry_root = (
            pathlib.Path(kitti_odometry_root) / "dataset" / "sequences"
        )
        self.semantic_kitti_root = (
            pathlib.Path(semantic_kitti_root) / "dataset" / "sequences"
        )
        self.semantic_kitti_voxels_root = (
            pathlib.Path(semantic_kitti_voxels_root) / "dataset" / "sequences"
        )

        self.config_semantic: pathlib.Path = pathlib.Path(
            __file__
        ).parent.parent / "config" / "semantic-kitti.yaml"

        if not self.kitti_odometry_root.is_dir():
            raise RuntimeError(
                "Could not find semantic kitti odometry root folder '{}'".format(
                    str(self.kitti_odometry_root)
                )
            )
        if not self.semantic_kitti_root.is_dir():
            raise RuntimeError(
                "Could not find semantic kitti root folder '{}'".format(
                    str(self.semantic_kitti_root)
                )
            )
        if not self.semantic_kitti_voxels_root.is_dir():
            raise RuntimeError(
                "Could not find semantic kitti voxels root folder '{}'".format(
                    str(self.semantic_kitti_voxels_root)
                )
            )

        if not self.config_semantic.is_file():
            raise RuntimeError(
                "Could not find kitti semantic config file '{}'".format(
                    str(self.config_semantic)
                )
            )

        self.voxel_version = voxel_version

        self.testset_flag = False

        self._split = None
        self._config_data = None
        self._samples_to_generate = None
        self._label_mapping = None
        self._voxel_data_cache = {}

        self._make_split()

    def __iter__(self):
        return self._samples_to_generate.__iter__()

    @property
    def split(self):
        return self._split

    def _make_split(self):
        """
        Use generated <self.voxel_version> output to build split.
        """

        # Todo: no test split option for now
        assert self.testset_flag is False
        valid_splits = ["train", "valid"]
        map_split_names = {"train": "train", "valid": "val", "test": "test"}
        # read config
        with open(str(self.config_semantic), "r") as file_conf_sem:
            yaml = YAML()
            data = yaml.load(file_conf_sem)
            self._config_data = {k: dict(v) for k, v in data.items()}

        data_splits = {
            map_split_names[k]: v
            for k, v in self._config_data["split"].items()
            if k in valid_splits
        }
        self._split = {
            "name": "semantic_kitti_voxels_{}".format(
                "default" if not self.testset_flag else "test"
            ),
            "data": {k: [] for k in data_splits.keys()},
        }

        self._samples_to_generate = []

        def parse_sequence_folder_name(x):
            try:
                return int(x)
            except ValueError:
                return -1

        voxel_sequences = {
            parse_sequence_folder_name(x.name): x
            for x in self.semantic_kitti_voxels_root.iterdir()
        }

        for split_name, sequences in data_splits.items():
            split_data = self._split["data"][split_name]
            for sequence_index in sequences:
                if not self.testset_flag:

                    if sequence_index not in voxel_sequences:
                        logger.warning(
                            "Sequence {:02d} not available. Skipping.".format(
                                sequence_index
                            )
                        )
                        continue

                    voxel_dir = voxel_sequences[sequence_index] / self.voxel_version
                    if not voxel_dir.is_dir():
                        logger.warning(
                            "Voxels not available in sequence {:02d}. Skipping.".format(
                                sequence_index
                            )
                        )
                        continue

                    self._voxel_data_cache[sequence_index] = {
                        int(x.stem[:6]): x
                        for x in (
                            voxel_sequences[sequence_index] / self.voxel_version
                        ).iterdir()
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
        # map unlabeled to extra entry 254
        self._label_mapping_voxels[0] = 254
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

            sequence_str = "{:02d}".format(sample[0])
            frame_str = "{:06d}".format(sample[1])

            point_cloud_file = (
                self.kitti_odometry_root
                / sequence_str
                / "velodyne"
                / "{}.bin".format(frame_str)
            )

            label_file = (
                self.semantic_kitti_root
                / sequence_str
                / "labels"
                / "{}.label".format(frame_str)
            )
            label_sem, _ = self.read_label(label_file)

            voxel_base = (
                self.semantic_kitti_voxels_root / sequence_str / self.voxel_version
            )
            voxel_data = self.read_semantic_kitti_voxel_label(
                voxel_base / frame_str, unpack_compressed=False,
            )

            proto_file = voxel_base / (frame_str + "_points.tfrecord")
            proto_data = self.read_proto_file(proto_file)

        else:
            raise NotImplementedError()

        point_cloud = self.read_pointcloud(point_cloud_file)

        if label_sem.shape[0] != point_cloud.shape[0]:
            raise RuntimeError(
                "Length of labels and point cloud does not match"
                "({} and {})".format(str(point_cloud_file), str(label_file))
            )
        try:
            label_sem = self._label_mapping(label_sem)
        except TypeError:
            raise RuntimeError(
                "Invalid label entry in label data '{}'.".format(str(label_file))
            )

        r["point_cloud"] = point_cloud.flatten()
        assert np.all(label_sem <= 255)
        r["semantic_labels"] = label_sem.astype(np.uint8).tobytes()

        # voxelized scene completion data
        try:
            voxel_label = self._label_mapping_voxels(voxel_data["label"])
        except TypeError:
            raise RuntimeError(
                "Invalid label entry in voxel label data '{}'.".format(str(voxel_base))
            )
        r["voxel_label"] = voxel_label.astype(np.uint8).tobytes()
        r["voxel_invalid"] = voxel_data["invalid"].tobytes()
        r["voxel_occluded"] = voxel_data["occluded"].tobytes()
        r["voxel_dynamic_occlusion"] = voxel_data["dynamic_occlusion"].tobytes()
        r.update(**proto_data)
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
    def read_semantic_kitti_voxel_label(
        semantic_kitti_sample: pathlib.Path, unpack_compressed: bool = False,
    ) -> {str: np.ndarray}:
        # compressed/uncompressed
        d = {
            "bin": True,
            "invalid": True,
            "label": False,
            "occluded": True,
            "dynamic_occlusion": True,
        }
        voxel_dims = (256, 256, 32)

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
                    x = SemanticKittiReaderVoxels.unpack(x)
                    x = x.reshape(voxel_dims)

            else:
                x = np.fromfile(str(filepath), dtype=np.uint16)
                x = x.reshape(voxel_dims)

            data[k] = x
        return data

    def read_proto_file(self, proto_file: pathlib.Path):
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
