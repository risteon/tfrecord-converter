#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import itertools
import typing
import pathlib
import re

import numpy as np

from .kitti_semantics_reader import KittiSemanticsIterator
from .kitti_accumulated_reader import KittiAccumulatedReader
from .cli_helper import get_files_flat


class KittiRawReader:
    """Read all samples specified by split from KITTI Raw
    and KITTI Raw processed folders.

    """

    sample_id_template = "kitti_raw_{day}_{seq:04d}_{frame:04d}"
    kitti_raw_seq_template = "{day}/{day}_drive_{seq:04d}_sync"
    sample_id_regex = re.compile(
        r"kitti_raw_(\d\d\d\d_\d\d_\d\d)_(\d\d\d\d)_(\d\d\d\d)"
    )

    def __init__(
        self,
        kitti_raw_root,
        kitti_processed_root,
        split: typing.Dict[str, typing.Union[str, dict]],
        sequence_data_generator="kitti_semantics",
        generator_kwargs=None,
    ):
        self.kitti_raw_root = pathlib.Path(kitti_raw_root)
        self.kitti_processed_root = pathlib.Path(kitti_processed_root)
        self.samples_to_generate = itertools.chain.from_iterable(split["data"].values())
        self._sem_prob_cache_old_seqs = {}
        self._sem_prob_cache = {}

        if sequence_data_generator == "kitti_semantics":
            self._get_generator = self.sequence_generator_kitti_semantics
        elif sequence_data_generator == "kitti_accumulated":
            self._get_generator = self.sequence_generator_kitti_accumulated
        else:
            raise RuntimeError(
                "Unknown sequence data generator '{}'.".format(sequence_data_generator)
            )
        self._generator_kwargs = generator_kwargs

    def __iter__(self):
        return self.samples_to_generate.__iter__()

    def sequence_generator_kitti_semantics(self, day, seq):
        raw_seq_path = KittiRawReader.kitti_raw_seq_template.format(day=day, seq=seq)

        velodyne_input_files = get_files_flat(
            self.kitti_raw_root / raw_seq_path / "velodyne_points" / "data",
            ending=".bin",
        )
        velodyne_input_files = sorted(velodyne_input_files)
        semantic_input_files = get_files_flat(
            self.kitti_processed_root
            / raw_seq_path
            / "lidar_semantics_deeplab_v3+71_90000",
            ending=".tfrecords",
        )
        semantic_input_files = sorted(semantic_input_files)

        kwargs = self._generator_kwargs if self._generator_kwargs is not None else {}
        return KittiSemanticsIterator(
            velodyne_input_files,
            semantic_input_files,
            sequence_id="kitti_raw_{day}_{seq:04d}".format(day=day, seq=seq),
            **kwargs
        )

    def sequence_generator_kitti_accumulated(self, day, seq):
        raw_seq_path = KittiRawReader.kitti_raw_seq_template.format(day=day, seq=seq)

        files_point_cloud = get_files_flat(
            self.kitti_raw_root / raw_seq_path / "velodyne_points" / "data",
            ending=".bin",
        )
        files_point_cloud = sorted(files_point_cloud)
        files_point_cloud_corrected = get_files_flat(
            self.kitti_processed_root
            / raw_seq_path
            / "velodyne_points_corrected"
            / "data",
            ending=".bin",
        )
        files_point_cloud_corrected = sorted(files_point_cloud_corrected)

        poses_file = (
            self.kitti_processed_root
            / raw_seq_path
            / "poses_sensor"
            / "sensor_poses.npy"
        )
        sensor_poses = np.load(str(poses_file))

        kwargs = self._generator_kwargs if self._generator_kwargs is not None else {}
        return KittiAccumulatedReader(
            sensor_poses,
            files_point_cloud,
            files_point_cloud_corrected,
            sequence_id="kitti_raw_{day}_{seq:04d}".format(day=day, seq=seq),
            **kwargs
        )

    @staticmethod
    def make_sample_id(sample: str) -> str:
        return sample

    def read(self, sample: str, _: str):
        try:
            match = self.sample_id_regex.fullmatch(sample)
            day, seq, frame = match[1], int(match[2]), int(match[3])
        except TypeError:
            raise RuntimeError(
                "Invalid KITTI Raw sample ID in split: {}".format(sample)
            )

        if (
            sample not in self._sem_prob_cache
            and sample not in self._sem_prob_cache_old_seqs
        ):
            # move current cached sequence to old sequences
            self._sem_prob_cache_old_seqs.update(self._sem_prob_cache)
            self._sem_prob_cache.clear()
            # read new sequence into cache
            gen = self._get_generator(day, seq)
            self._read_kitti_sequence_into_cache(gen)

        # now sample is either in new or old cache
        try:
            x, gen = self._sem_prob_cache.pop(sample)
        except KeyError:
            x, gen = self._sem_prob_cache_old_seqs.pop(sample)

        # trying to save some RAM
        if len(self._sem_prob_cache_old_seqs) > 1000:
            self._sem_prob_cache_old_seqs.clear()
        return gen.reader(x, sample)

    def _read_kitti_sequence_into_cache(self, gen):
        for x in gen:
            sample_id = gen.make_id_from_sample(x)
            self._sem_prob_cache[sample_id] = (x, gen)
