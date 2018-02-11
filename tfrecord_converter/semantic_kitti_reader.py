#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import pathlib
import typing
import itertools

import numpy as np
from ruamel.yaml import YAML


class SemanticKittiReader:

    sample_id_template = "kitti_raw_{day}_{seq:04d}_{frame:04d}"
    sample_id_test_template = "kitti_odometry_test_{seq:02d}_{frame:04d}"
    kitti_raw_seq_template = "{day}/{day}_drive_{seq:04d}_sync"

    def __init__(
        self,
        kitti_raw_root: str,
        kitti_odometry_root: str,
        semantic_kitti_root: str,
        testset: bool = False,
    ):
        self.kitti_raw_root = pathlib.Path(kitti_raw_root)
        self.kitti_odometry_root = (
            pathlib.Path(kitti_odometry_root) / "dataset" / "sequences"
        )
        self.semantic_kitti_root = (
            pathlib.Path(semantic_kitti_root) / "dataset" / "sequences"
        )
        self.config_semantic: pathlib.Path = pathlib.Path(
            __file__
        ).parent.parent / "config" / "semantic-kitti.yaml"
        self.odometry_mapping: pathlib.Path = pathlib.Path(
            __file__
        ).parent.parent / "config" / "kitti_odometry_mapping.yaml"

        self.testset = testset

        if not self.kitti_raw_root.is_dir():
            raise RuntimeError(
                "Could not find kitti raw root folder '{}'".format(
                    str(self.kitti_raw_root)
                )
            )
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
        if not self.config_semantic.is_file():
            raise RuntimeError(
                "Could not find kitti semantic config file '{}'".format(
                    str(self.config_semantic)
                )
            )
        if not self.odometry_mapping.is_file():
            raise RuntimeError(
                "Could not find odometry mapping file '{}'".format(
                    str(self.odometry_mapping)
                )
            )

        self._split = None
        self._config_data = None
        self._samples_to_generate = None
        self._label_mapping = None
        self._data_cache = {}
        self._make_split()

    def __iter__(self):
        return self._samples_to_generate.__iter__()

    @property
    def split(self):
        return self._split

    def _make_split(self):
        # not processing test
        valid_splits = ["train", "valid"] if not self.testset else ["test"]
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
            "name": "semantic_kitti_{}".format(
                "default" if not self.testset else "test"
            ),
            "data": {k: [] for k in data_splits.keys()},
        }

        # read kitti odometry to raw mapping
        with open(str(self.odometry_mapping), "r") as file_odometry_mapping:
            yaml = YAML()
            odometry_mapping = yaml.load(file_odometry_mapping)
            odometry_mapping = [tuple(x) for x in list(odometry_mapping)]

        self._samples_to_generate = []
        for split_name, sequences in data_splits.items():
            split_data = self._split["data"][split_name]
            for sequence_index in sequences:
                if not self.testset:
                    k_day, k_seq, index_start, index_end = odometry_mapping[
                        sequence_index
                    ]
                    # index end is inclusive (KITTI Odometry Readme)
                    split_data.extend(
                        [
                            self.sample_id_template.format(
                                day=k_day, seq=k_seq, frame=index
                            )
                            for index in range(index_start, index_end + 1)
                        ]
                    )
                    self._samples_to_generate.extend(
                        [
                            (k_day, k_seq, raw_index, sequence_index, sem_index)
                            for sem_index, raw_index in enumerate(
                                range(index_start, index_end + 1)
                            )
                        ]
                    )
                else:
                    sequence_data_path = (
                        self.kitti_odometry_root
                        / "{:02d}".format(sequence_index)
                        / "velodyne"
                    )
                    if not sequence_data_path.is_dir():
                        raise RuntimeError(
                            "Data directory '{}' is not available.".format(
                                str(sequence_data_path)
                            )
                        )
                    data_files = sorted([x for x in sequence_data_path.iterdir()])
                    # sanity check
                    if not all(
                        data_files[i].stem == "{:06d}".format(i)
                        for i in range(len(data_files))
                    ):
                        raise RuntimeError(
                            "Invalid files in data directory '{}'".format(
                                str(sequence_data_path)
                            )
                        )
                    self._data_cache[sequence_index] = data_files
                    split_data.extend(
                        [
                            self.sample_id_test_template.format(
                                seq=sequence_index, frame=i
                            )
                            for i in range(len(data_files))
                        ]
                    )
                    self._samples_to_generate.extend(
                        list(
                            zip(
                                itertools.repeat(sequence_index), range(len(data_files))
                            )
                        )
                    )

        self._label_mapping: dict = self._config_data["learning_map"]
        # make 255 the 'unlabeled' label and shift all others down (-1) accordingly
        self._label_mapping = {
            k: v - 1 if v != 0 else 255 for k, v in self._label_mapping.items()
        }
        self._label_mapping = np.vectorize(self._label_mapping.get, otypes=[np.int64])

    def make_sample_id(
        self,
        sample: typing.Union[
            typing.Tuple[str, int, int, int, int], typing.Tuple[int, int]
        ],
    ):
        if not self.testset:
            return self.sample_id_template.format(
                day=sample[0], seq=sample[1], frame=sample[2]
            )
        else:
            return self.sample_id_test_template.format(seq=sample[0], frame=sample[1])

    def read(
        self,
        sample: typing.Union[
            typing.Tuple[str, int, int, int, int], typing.Tuple[int, int]
        ],
        sample_id: str,
    ):

        if not self.testset:
            kitti_sequence = self.kitti_raw_seq_template.format(
                day=sample[0], seq=sample[1]
            )
            kitti_raw_seq_folder = self.kitti_raw_root / kitti_sequence
            if not kitti_raw_seq_folder.is_dir():
                # use from backup KITTI Odometry location
                # (one sequence is missing in KITTI raw)
                point_cloud_file = (
                    self.kitti_odometry_root
                    / "{:02d}".format(sample[3])
                    / "velodyne"
                    / "{:06d}.bin".format(sample[4])
                )
            else:
                point_cloud_file = (
                    self.kitti_raw_root
                    / kitti_sequence
                    / "velodyne_points"
                    / "data"
                    / "{:010d}.bin".format(sample[2])
                )
        else:
            point_cloud_file = self._data_cache[sample[0]][sample[1]]

        point_cloud = self.read_pointcloud(point_cloud_file)

        r = {
            "sample_id": sample_id.encode("utf-8"),
            "point_cloud": point_cloud.flatten(),
        }

        if not self.testset:
            label_file = (
                self.semantic_kitti_root
                / "{:02d}".format(sample[3])
                / "labels"
                / "{:06d}.label".format(sample[4])
            )
            label_sem, _ = self.read_label(label_file)
            if label_sem.shape[0] != point_cloud.shape[0]:
                raise RuntimeError(
                    "Lenght of labels and point cloud does not match"
                    "({} and {})".format(str(point_cloud_file), str(label_file))
                )
            try:
                label_sem = self._label_mapping(label_sem)
            except TypeError:
                raise RuntimeError(
                    "Invalid label entry in label data '{}'.".format(label_file)
                )
            r["semantic_labels"] = label_sem
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
