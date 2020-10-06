#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import itertools
import typing
import pathlib

import numpy as np
import tensorflow as tf

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.splits import train as split_train
from nuscenes.utils.splits import val as split_val
from nuscenes.utils.geometry_utils import transform_matrix

from pyquaternion import Quaternion

from .common import ObjectClass
from .algorithm import assign_point_cloud_to_bounding_boxes

# Full mapping of all NuScene classes for pseudo semseg
NUSCENES_SEM_CLASSES = {
    "animal": 0,
    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 2,
    "human.pedestrian.construction_worker": 3,
    "human.pedestrian.personal_mobility": 4,
    "human.pedestrian.police_officer": 5,
    "human.pedestrian.stroller": 6,
    "human.pedestrian.wheelchair": 7,
    "movable_object.barrier": 8,
    "movable_object.debris": 9,
    "movable_object.pushable_pullable": 10,
    "movable_object.trafficcone": 11,
    "vehicle.bicycle": 12,
    "vehicle.bus.bendy": 13,
    "vehicle.bus.rigid": 14,
    "vehicle.car": 15,
    "vehicle.construction": 16,
    "vehicle.emergency.ambulance": 17,
    "vehicle.emergency.police": 18,
    "vehicle.motorcycle": 19,
    "vehicle.trailer": 20,
    "vehicle.truck": 21,
    "static_object.bicycle_rack": 22,
}

# Reduced mapping for object detection/classification
NUSCENES_OBJECT_CLASSES = {
    "vehicle.car": ObjectClass.PASSENGER_CAR,
    "vehicle.truck": ObjectClass.TRUCK,
    "vehicle.bus": ObjectClass.TRUCK,
    "vehicle.bicycle": ObjectClass.CYCLIST,
    "human.pedestrian": ObjectClass.PEDESTRIAN,
}


def nuscenes_category_to_semantic_class(cat: str) -> int:
    return NUSCENES_SEM_CLASSES.get(cat, -1)


def nuscenes_category_reduced(cat: str) -> str:
    # use only first two levels (for now...)
    # e.g. human.pedestrian.construction_worker -> human.pedestrian
    return ".".join(cat.split(".")[:2])


def nuscenes_category_to_object_class(cat: str) -> int:
    # use only first two levels (for now...)
    # e.g. human.pedestrian.construction_worker -> human.pedestrian
    return NUSCENES_OBJECT_CLASSES.get(
        nuscenes_category_reduced(cat), ObjectClass.UNKNOWN
    ).value


class NuscenesReader:

    scene_split_lists = {"train": set(split_train), "val": set(split_val)}
    sample_id_template = "ns_{}_{:02d}"
    np_str_to_cat = np.vectorize(nuscenes_category_to_object_class, otypes=[np.int64])
    np_str_to_sem = np.vectorize(nuscenes_category_to_semantic_class, otypes=[np.int64])

    # Nuscenes LiDAR has x-axis to vehicle right and y-axis to front.
    # Turn this 90 degrees to have x-axis facing the front
    turn_matrix = np.linalg.inv(
        np.asarray(
            [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
    )
    turn_quaternion = Quaternion(axis=(0.0, 0.0, 1.0), radians=-np.pi / 2.0)

    def __init__(
        self,
        nuscenes_root,
        version="v1.0-trainval",
        max_scenes=None,
        *,
        read_radar: bool = True,
        read_camera: bool = True,
        read_semantics: bool = True,
        read_bounding_boxes: bool = True
    ):
        self.nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
        self.root = pathlib.Path(nuscenes_root)

        # global counter to sanity-check if we calculate the same number of points
        # within boxes as the dataset authors
        self.ns_lidar_pts_to_calculated_diff = 0

        # flags defining the data entries to return from 'read'
        self.read_radar = read_radar
        self.read_camera = read_camera
        self.read_semantics = read_semantics
        self.read_bounding_boxes = read_bounding_boxes

        if self.read_semantics and not hasattr(self.nusc, "lidarseg"):
            raise RuntimeError("Error: nuScenes-lidarseg not installed!")

        # assert that the training targets range from 0 - (|mapping| - 1)
        assert len(set(NUSCENES_SEM_CLASSES.values())) == len(NUSCENES_SEM_CLASSES)
        assert all(
            a == b
            for a, b in zip(
                sorted(NUSCENES_SEM_CLASSES.values()), range(len(NUSCENES_SEM_CLASSES))
            )
        )

        split_name = {
            "v1.0-trainval": "nuscenes_default",
            "v1.0-mini": "nuscenes_mini",
        }.get(version, "nuscenes_{}".format(version))

        # create split dict
        self.split = {
            "name": split_name,
            "data": {k: [] for k in self.scene_split_lists.keys()},
        }
        for i, scene in enumerate(self.nusc.scene):
            if max_scenes is not None and i > max_scenes:
                break
            name = scene["name"]
            for k, v in self.scene_split_lists.items():
                if name in v:
                    split_list = self.split["data"][k]
                    split_list.extend(
                        [
                            self.sample_id_template.format(name, i)
                            for i in range(0, scene["nbr_samples"])
                        ]
                    )
                    break
            else:
                raise RuntimeError(
                    "Found scene that is not in a split: {}".format(name)
                )

    def _make_path(self, filename: str):
        return str(self.root / filename)

    def __iter__(self):
        self._scene_iter = self.nusc.scene.__iter__()
        self._next_scene()
        return self

    def __next__(self) -> typing.Tuple[dict, int]:
        try:
            return self._sample_iter.__next__()
        except StopIteration:
            self._next_scene()
            return self._sample_iter.__next__()

    def make_sample_id(self, sample: typing.Tuple[dict, int]) -> str:
        scene = self.nusc.get("scene", sample[0]["scene_token"])
        return self.sample_id_template.format(scene["name"], sample[1])

    def _next_scene(self):
        self._current_scene = self._scene_iter.__next__()

        class SampleIteration:
            """Iterate over nuscenes samples of a scene.
            Add an additional sample index.
            """

            def __init__(self, scene, nusc):
                self.nusc = nusc
                self._pos = scene["first_sample_token"]
                self._index = itertools.count().__iter__()

            def __next__(self):
                if not self._pos:
                    raise StopIteration()
                sample = self.nusc.get("sample", self._pos)
                self._pos = sample["next"]
                return sample, self._index.__next__()

        self._sample_iter = SampleIteration(self._current_scene, self.nusc)

    def read(self, sample: typing.Tuple[dict, int], _sample_id: str):
        d = self._read_sample(sample)

        def convert(entry):
            """Convert data types to be compatible with tfrecords features."""
            if isinstance(entry, str):
                return entry.encode("utf-8")
            if isinstance(entry, np.ndarray):
                entry = entry.flatten()
                if not np.issubdtype(entry.dtype, np.number):
                    entry = tf.io.serialize_tensor(tf.convert_to_tensor(entry))
                elif entry.dtype == np.float64:
                    entry = entry.astype(np.float32)
                return entry
            return entry

        return {k: convert(v) for k, v in d.items()}

    def _read_sample(self, sample: typing.Tuple[dict, int]):

        radars = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
        cameras = ["CAM_FRONT"]

        sample_id = self.make_sample_id(sample)
        sample, sample_index = sample
        sample_data_token: str = sample["data"]["LIDAR_TOP"]

        lidar_sample_data = self.nusc.get("sample_data", sample_data_token)
        assert lidar_sample_data["is_key_frame"]

        data_path, box_list, cam_intrinsic = self.nusc.get_sample_data(
            sample_data_token
        )

        # POINT CLOUD [(x y z I) x P]. Intensity from 0.0 to 255.0
        point_cloud_orig = LidarPointCloud.from_file(data_path)
        if point_cloud_orig.points.dtype != np.float32:
            raise RuntimeError("Point cloud has wrong data type.")
        # -> [P x (x y z I)]
        point_cloud_orig = point_cloud_orig.points.transpose()

        camera_data = {}
        radar_data = {}
        box_data = {}
        lidarseg_data = {}

        if self.read_semantics:
            lidarseg_data = self._read_lidarseg_data(sample_data_token, point_cloud_orig)

        if self.read_bounding_boxes:
            box_data = self._read_bounding_boxes(box_list, point_cloud_orig, sample)

        # rotate point cloud 90 degrees around z-axis,
        # so that x-axis faces front of vehicle
        x = point_cloud_orig[:, 0:1]
        y = point_cloud_orig[:, 1:2]
        point_cloud = np.concatenate((y, -x, point_cloud_orig[:, 2:4]), axis=-1)

        # LiDAR extrinsic calibration
        calibration_lidar = self.nusc.get(
            "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
        )
        tf_lidar = transform_matrix(
            calibration_lidar["translation"], Quaternion(calibration_lidar["rotation"])
        )
        # vehicle -> point cloud coords (turned by 90 degrees)
        tf_vehicle_pc = np.linalg.inv(np.dot(tf_lidar, NuscenesReader.turn_matrix))

        if self.read_camera:
            # CAMERAS
            camera_data: [{str: typing.Any}] = [
                self._read_sensor_data_and_extrinsics(
                    sample, cam_name, tf_vehicle_pc, self._read_camera_data
                )
                for cam_name in cameras
            ]
            camera_data = {k: v for d in camera_data for k, v in d.items()}

        if self.read_radar:
            # RADARS
            radar_data: [{str: typing.Any}] = [
                self._read_sensor_data_and_extrinsics(
                    sample, radar_name, tf_vehicle_pc, self._read_radar_data
                )
                for radar_name in radars
            ]
            radar_data = {k: v for d in radar_data for k, v in d.items()}

        assert box_data.keys().isdisjoint(camera_data.keys())
        assert box_data.keys().isdisjoint(radar_data.keys())
        assert box_data.keys().isdisjoint(lidarseg_data.keys())
        # return feature array. Add sample ID.
        d = {
            "sample_id": sample_id,
            "point_cloud": point_cloud,
            **box_data,
            **camera_data,
            **radar_data,
            **lidarseg_data,
        }
        return d

    def _read_lidarseg_data(self, sample_data_token: str, point_cloud_orig: np.ndarray):
        # self.nusc.lidarseg_idx2name_mapping

        lidarseg_labels_filename = (
            pathlib.Path(self.nusc.dataroot)
            / self.nusc.get("lidarseg", sample_data_token)["filename"]
        )
        # Load labels from .bin file.
        points_label = np.fromfile(
            str(lidarseg_labels_filename), dtype=np.uint8
        )  # [num_points]

        if points_label.shape[0] != point_cloud_orig.shape[0]:
            raise ValueError("Semantic labels do not match point cloud.")

        return {"semantic_labels": tf.io.serialize_tensor(points_label)}

    def _read_bounding_boxes(self, box_list, point_cloud_orig, sample):
        # create transform matrices for all boxes
        dt = np.float32
        if box_list:
            r = np.asarray([x.rotation_matrix for x in box_list])
            c = np.asarray([x.center for x in box_list])
            center_pos = np.asarray([x.center for x in box_list], dtype=dt)
            # NuScenes format: width, length, height. Reorder to l w h
            box_dims_lwh = np.asarray([x.wlh for x in box_list], dtype=np.float64)[
                :, [1, 0, 2]
            ]
            box_rot = np.asarray(
                [(self.turn_quaternion * x.orientation).q for x in box_list], dtype=dt
            )
        else:
            r = np.zeros(shape=[0, 3, 3], dtype=np.float64)
            c = np.zeros(shape=[0, 3], dtype=np.float64)
            center_pos = np.zeros(shape=[0, 3], dtype=np.float64)
            box_dims_lwh = np.zeros(shape=[0, 3], dtype=np.float64)
            box_rot = np.zeros(shape=[0, 4], dtype=np.float64)

        rc = np.concatenate((r, c[:, :, None]), axis=-1)
        tfs = np.concatenate(
            (
                rc,
                np.broadcast_to(
                    tf.constant([0, 0, 0, 1], dtype=rc.dtype)[None, None, :],
                    [rc.shape[0], 1, 4],
                ),
            ),
            axis=1,
        )

        total_points_per_box, mapping = assign_point_cloud_to_bounding_boxes(
            point_cloud_orig, bounding_boxes_tfs=tfs, bounding_boxes_dims=box_dims_lwh,
        )

        # 3D BOXES IN LIDAR COORDS [(x y z) (w x y z) (l w h)]
        bboxes_spatial = np.empty(shape=(len(box_list), 10), dtype=dt)
        bboxes_spatial[:, 0] = center_pos[:, 1]
        bboxes_spatial[:, 1] = -center_pos[:, 0]
        bboxes_spatial[:, 2] = center_pos[:, 2]
        bboxes_spatial[:, 3:7] = box_rot
        bboxes_spatial[:, 7:10] = box_dims_lwh.astype(dt)

        object_str = np.array([x.name for x in box_list], dtype=np.unicode)
        object_category = NuscenesReader.np_str_to_cat(object_str).astype(np.int64)
        class_value = NuscenesReader.np_str_to_sem(object_str).astype(np.int64)
        lidar_pts = np.asarray(
            [
                x["num_lidar_pts"]
                for x in [
                    self.nusc.get("sample_annotation", ann_token)
                    for ann_token in sample["anns"]
                ]
            ]
        )
        # rotate box transforms (same as bboxes_spatial)
        tfs = np.matmul(np.linalg.inv(self.turn_matrix)[None, ...], tfs)

        # track to see if box calculation gives the same results as the
        # lidar points counter from the nuscenes dataset
        diff = np.sum(np.abs(lidar_pts - total_points_per_box))
        self.ns_lidar_pts_to_calculated_diff += diff

        return {
            "bounding_boxes_3d_spatial": bboxes_spatial,
            "bounding_boxes_3d_transforms": tfs,
            "bounding_boxes_class": class_value,
            "bounding_boxes_category": object_category,
            "bounding_boxes_class_str": object_str,
            "bounding_boxes_point_counter": total_points_per_box.astype(np.int64),
            "bounding_boxes_point_mapping": mapping,
        }

    @staticmethod
    def _read_sensor_data_and_extrinsics(
        sample, sensor_name: str, coord_transform: np.ndarray, reader_func
    ):
        data, extrinsics = reader_func(sample["data"][sensor_name])
        extrinsics = np.matmul(coord_transform, extrinsics)
        return {
            **{k.format(sensor_name.lower()): v for k, v in data.items()},
            "{}_extrinsics".format(sensor_name.lower()): extrinsics,
        }

    def _read_camera_data(self, sample_data_token):
        sd_record = self.nusc.get("sample_data", sample_data_token)
        cs_record = self.nusc.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )
        sensor_record = self.nusc.get("sensor", cs_record["sensor_token"])
        assert sensor_record["modality"] == "camera"

        # currently using only keyframes
        assert sd_record["is_key_frame"]

        data_path = self.nusc.get_sample_data_path(sample_data_token)
        imsize = (sd_record["width"], sd_record["height"])

        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        cam_extrinsics = transform_matrix(
            cs_record["translation"], Quaternion(cs_record["rotation"])
        )

        with open(data_path, "rb") as in_file:
            img_bytes_jpg = in_file.read()

        return (
            {
                "{}_jpg": img_bytes_jpg,
                "{}_size": np.asarray(imsize, dtype=np.int64),
                "{}_intrinsics": cam_intrinsic.astype(np.float32),
            },
            cam_extrinsics,
        )

    def _read_radar_data(self, sample_data_token):
        sd_record = self.nusc.get("sample_data", sample_data_token)
        cs_record = self.nusc.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )
        sensor_record = self.nusc.get("sensor", cs_record["sensor_token"])
        assert sensor_record["modality"] == "radar"

        # currently using only keyframes
        assert sd_record["is_key_frame"]

        data_path = self.nusc.get_sample_data_path(sample_data_token)
        radar_point_cloud = RadarPointCloud.from_file(data_path)
        points = tf.convert_to_tensor(radar_point_cloud.points.transpose())

        radar_extrinsics = transform_matrix(
            cs_record["translation"], Quaternion(cs_record["rotation"])
        )

        return {"{}_points": tf.io.serialize_tensor(points)}, radar_extrinsics


class NuscenesObjectsReader:
    """Return individual objects."""

    SAMPLE_ID_PATTERN = "{sample}_#{obj_id:03d}_{category}"
    check_valid = np.vectorize(lambda x: x in NUSCENES_OBJECT_CLASSES, otypes=[np.bool])

    def __init__(self, nuscenes_root, version="v1.0-trainval", max_scenes=None):
        self.nusc_reader = NuscenesReader(nuscenes_root, version, max_scenes)
        self._build_split()
        self._sample_iterator = iter(self.nusc_reader)

    def __iter__(self):
        self._next_sample()
        return self

    def __next__(self):
        """Returned sample is:
        ((obj_index, (annotation_token, category str)), frame_id)
        """
        try:
            return next(self.data)
        except StopIteration:
            self._next_sample()
            return self.__next__()

    @staticmethod
    def read(sample, sample_id):
        return {
            "sample_id": sample_id.encode("utf-8"),
            "transform": sample["bounding_boxes_3d_transforms"]
            .flatten()
            .astype(np.float32),
            "bounding_box_3d": sample["bounding_boxes_3d_spatial"],
            "point_count": sample["bounding_boxes_point_counter"],
            "points": (
                sample["point_cloud"][sample["mapping"] == sample["box_id"]]
            ).flatten(),
            "object_cls": sample["bounding_boxes_3d_class"],
            "object_category": sample["bounding_boxes_3d_category"].encode("utf-8"),
        }

    def make_sample_id(self, sample):
        return self.SAMPLE_ID_PATTERN.format(
            sample=sample["frame_id"],
            obj_id=sample["box_id"],
            category=sample["category_reduced"],
        )

    def _next_sample(self):
        sample = next(self._sample_iterator)
        # read all objects in this frame. Order is the same as annotation tokens
        data = self.nusc_reader._read_sample(sample)
        frame_id = data["sample_id"]
        per_box = {
            k: v
            for k, v in data.items()
            if k
            in {
                "bounding_boxes_3d_spatial",
                "bounding_boxes_3d_class",
                "bounding_boxes_point_counter",
                "bounding_boxes_3d_category",
                "bounding_boxes_3d_transforms",
            }
        }

        annotation_tokens = sample[0]["anns"]
        categories = np.asarray(
            [
                nuscenes_category_reduced(
                    self.nusc_reader.nusc.get("sample_annotation", x)["category_name"]
                )
                for x in annotation_tokens
            ]
        )
        box_ids = np.arange(len(annotation_tokens), dtype=np.int32)
        valid_ids = box_ids[self.check_valid(categories)]

        valid_objects = {k: v[valid_ids] for k, v in per_box.items()}
        valid_objects["box_id"] = valid_ids
        valid_objects["category_reduced"] = categories[valid_ids]

        class Getter:
            def __init__(
                self,
                obj: typing.Dict[str, typing.Any],
                fixed: typing.Dict[str, typing.Any],
            ):
                self._len = len(next(iter(obj.values())))
                for v in obj.values():
                    if len(v) != self._len:
                        raise RuntimeError("Unequal lengths.")
                self._index = None
                self._obj = obj
                self._fixed = fixed

            def __iter__(self):
                self._index = iter(range(self._len))
                return self

            def __next__(self):
                i = next(self._index)
                at_index = {k: v[i] for k, v in self._obj.items()}
                at_index.update(self._fixed)
                return at_index

        # only return samples of known categories
        self.data = iter(
            Getter(
                obj=valid_objects,
                fixed={
                    "frame_id": frame_id,
                    "mapping": data["bounding_boxes_point_mapping"],
                    "point_cloud": data["point_cloud"],
                },
            )
        )

    def _build_split(self):
        # run through data once to make split
        # from number of objects in each class and sample
        combinations = list(
            itertools.product(
                NUSCENES_OBJECT_CLASSES.keys(), self.nusc_reader.split["data"].keys()
            )
        )
        self._split = {
            "name": "objects_gt_{}".format(self.nusc_reader.split["name"]),
            "data": {self._make_split_name(x[0], x[1]): [] for x in combinations},
        }
        for sample in self.nusc_reader:
            # find split of objects from this sample
            frame_sample_id = self.nusc_reader.make_sample_id(sample)
            split_name = self._get_split(frame_sample_id)
            # all objects in this frame
            annotations = [
                self.nusc_reader.nusc.get("sample_annotation", ann_token)
                for ann_token in sample[0]["anns"]
            ]
            # simplified category strings
            cats_str = np.array(
                [
                    nuscenes_category_reduced(annotation["category_name"])
                    for annotation in annotations
                ]
            )
            box_ids = np.arange(len(cats_str), dtype=np.int32)

            for obj_cat in NUSCENES_OBJECT_CLASSES.keys():
                obj_split_name = self._make_split_name(obj_cat, split_name)
                current_ids = box_ids[cats_str == np.asarray(obj_cat)]
                if current_ids.size == 0:
                    continue
                self._split["data"][obj_split_name].extend(
                    [
                        self.SAMPLE_ID_PATTERN.format(
                            sample=frame_sample_id, obj_id=x, category=obj_cat
                        )
                        for x in current_ids
                    ]
                )

    @property
    def split(self):
        return self._split

    @staticmethod
    def _make_split_name(obj_category: str, base_split: str) -> str:
        return obj_category + "_" + base_split

    def _get_split(self, frame_sample_id: str):
        for split_name, samples in self.nusc_reader.split["data"].items():
            if frame_sample_id in samples:
                return split_name
        raise RuntimeError(
            "Invalid sample ID {} not in any split.".format(frame_sample_id)
        )
