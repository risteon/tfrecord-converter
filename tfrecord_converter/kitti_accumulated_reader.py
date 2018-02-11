#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
import pathlib
import collections
import itertools
import typing
import numpy as np


class KittiAccumulatedReader:
    def __init__(
        self,
        sensor_poses: np.ndarray,
        files_point_clouds: typing.List[pathlib.Path],
        files_point_clouds_corrected: typing.List[str],
        sequence_id: str,
        cutoff_radius: float = np.inf,
        cuboid: np.ndarray = np.asarray([[0.0, -25.0, -3.0], [50.0, 25.0, 1.0]]),
    ):
        """

        :param sensor_poses:
        :param files_point_clouds:
        :param files_point_clouds_corrected:
        :param cutoff_radius:
        """

        if not (
            len(sensor_poses)
            == len(files_point_clouds)
            == len(files_point_clouds_corrected)
        ):
            raise RuntimeError("Different input data lenghts.")
        if not self.is_sorted(files_point_clouds):
            raise RuntimeError("Velodyne filenames are not in sorted order")
        if not self.is_sorted(files_point_clouds_corrected):
            raise RuntimeError("Semantic tfrecords filenames are not in sorted order")

        self.window_length_future = 35
        self.window_length_past = 8

        self.files_point_clouds = files_point_clouds
        self.files_point_clouds_corrected = files_point_clouds_corrected
        self.sensor_poses = sensor_poses
        self.sample_indices = [int(x.stem) for x in self.files_point_clouds]

        self.sequence_id = sequence_id
        self.cutoff_radius = cutoff_radius
        self.cuboid = cuboid

        self.pcs = None
        self.attributes = None
        self.read_sequence()

    def __len__(self):
        return len(self.files_point_clouds)

    def __iter__(self):
        return range(len(self)).__iter__()

    def make_id_from_sample(self, sample):
        return "{}_{:04d}".format(self.sequence_id, self.sample_indices[sample])

    def reader(self, sample, sample_id):
        window_begin = max(sample - self.window_length_past, 0)
        window_end = min(sample + self.window_length_future, len(self))

        pc_slice = list(itertools.islice(self.pcs, window_begin, window_end))
        attr_slice = {
            k: list(itertools.islice(v, window_begin, window_end))
            for k, v in self.attributes.items()
        }
        accumulated_pc, attr = self.make_cutout(
            pc_slice, self.sensor_poses[sample], attr_slice
        )
        return {
            "sample_id": sample_id.encode("utf-8"),
            "point_cloud": KittiAccumulatedReader.read_pointcloud(
                self.files_point_clouds[sample]
            ).flatten(),
            "point_cloud_ego_motion": KittiAccumulatedReader.read_pointcloud(
                self.files_point_clouds_corrected[sample]
            ).flatten(),
            "acc_point_cloud": accumulated_pc.flatten(),
            **{"acc_" + k: v for k, v in attr.items()},
        }

    @staticmethod
    def read_pointcloud(filepath):
        scan = np.fromfile(str(filepath), dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    @staticmethod
    def is_sorted(filelist):
        return all(filelist[i] <= filelist[i + 1] for i in range(len(filelist) - 1))

    def read_sequence(self):

        self.pcs = collections.deque()
        self.attributes = {
            "reflectivity": collections.deque(),
        }

        for i, (sensor_pose, file_pc_corrected) in enumerate(
            zip(self.sensor_poses, self.files_point_clouds_corrected)
        ):

            pc = self.read_pointcloud(file_pc_corrected)
            pc, _ = self.filter_kitti_roof_artefacts(pc, ())
            pc, _ = self.filter_distance(pc, self.cutoff_radius, ())
            homogeneous = np.concatenate(
                (pc[:, :3], np.ones(dtype=pc.dtype, shape=[pc.shape[0], 1])), axis=1
            )
            pc_world = np.squeeze(
                np.matmul(sensor_pose, np.expand_dims(homogeneous, axis=-1)), axis=-1
            ).astype(np.float32)

            self.pcs.append(pc_world)
            self.attributes["reflectivity"].append(pc[:, 3])

    def make_cutout(self, point_clouds, sensor_pose: np.ndarray, attributes):
        pc = np.concatenate(point_clouds, axis=0)
        attr = {k: np.concatenate(v, axis=0) for k, v in attributes.items()}
        # transform all points into sensor pose coordinates, discard homogeneous '1s'
        transformed = np.squeeze(
            np.matmul(np.linalg.inv(sensor_pose), np.expand_dims(pc, axis=-1)), axis=-1
        )[:, :3]
        # get all points within selected cuboid around sensor
        valid = np.all(
            np.concatenate(
                (transformed >= self.cuboid[0], transformed <= self.cuboid[1]), axis=-1
            ),
            axis=1,
        )
        return pc[valid][:, :3], {k: v[valid] for k, v in attr.items()}

    @staticmethod
    def filter_kitti_roof_artefacts(pc: np.ndarray, extra=()):
        x_margin = np.logical_and(pc[:, 0] < 1.6, pc[:, 0] > -1.8)
        y_margin = np.logical_and(pc[:, 1] < 1.5, pc[:, 1] > -1.5)
        z_margin = np.logical_and(pc[:, 2] < -0.4, pc[:, 2] > -0.8)
        mask = ~np.all(np.stack((x_margin, y_margin, z_margin)), axis=0)
        return pc[mask], tuple(x[mask] for x in extra)

    @staticmethod
    def filter_distance(pc: np.ndarray, cut_off_radius, extra=()):
        distances = np.linalg.norm(pc[:, :3], axis=-1)
        mask = distances < cut_off_radius
        return pc[mask], tuple(x[mask] for x in extra)
