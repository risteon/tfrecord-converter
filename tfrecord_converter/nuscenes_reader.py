# -*- coding: utf-8 -*-

"""

"""

import itertools
import typing
import pathlib

import numpy as np
import tensorflow as tf

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import train as split_train
from nuscenes.utils.splits import val as split_val
from nuscenes.utils.geometry_utils import transform_matrix

from pyquaternion import Quaternion

from .common import ObjectClass
from .algorithm import assign_point_cloud_to_bounding_boxes


NUSCENES_OBJECT_CLASSES = {
    "vehicle.car": ObjectClass.PASSENGER_CAR,
    "vehicle.truck": ObjectClass.TRUCK,
    "vehicle.bus": ObjectClass.TRUCK,
    "vehicle.bicycle": ObjectClass.CYCLIST,
    "human.pedestrian": ObjectClass.PEDESTRIAN,
}


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
    np_str_to_cls = np.vectorize(nuscenes_category_to_object_class, otypes=[np.int64])

    # Nuscenes LiDAR has x-axis to vehicle right and y-axis to front.
    # Turn this 90 degrees to have x-axis facing the front
    turn_matrix = np.asarray(
        [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )

    def __init__(self, nuscenes_root, version="v1.0-trainval", max_scenes=None):
        self.nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
        self.root = pathlib.Path(nuscenes_root)
        self.ns_lidar_pts_to_calculated_diff = 0

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

    def make_sample_id(self, sample: typing.Tuple[dict, int]):
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
        d = self.read_internal(sample)
        return {
            "sample_id": d["sample_id"].encode("utf-8"),
            "point_cloud": d["point_cloud"].flatten(),
            "bounding_boxes_3d_spatial": d["bounding_boxes_3d_spatial"].flatten(),
            "bounding_boxes_3d_class": d["bounding_boxes_3d_class"],
            "bounding_boxes_points": d["bounding_boxes_points"],
            "bounding_boxes_3d_category": tf.io.serialize_tensor(
                tf.convert_to_tensor(d["bounding_boxes_3d_category"])
            ),
        }

    def read_internal(self, sample: typing.Tuple[dict, int]):
        sample_id = self.make_sample_id(sample)
        sample, sample_index = sample
        lidar_sample_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        assert lidar_sample_data["is_key_frame"]
        ego_pose_lidar = self.nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        # [(x y z I) x P]. Intensity from 0.0 to 255.0
        point_cloud = LidarPointCloud.from_file(
            self._make_path(lidar_sample_data["filename"])
        )
        if point_cloud.points.dtype != np.float32:
            raise RuntimeError("Point cloud has wrong data type.")
        pc = point_cloud.points.transpose()
        calibration_lidar = self.nusc.get(
            "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
        )

        annotations = self._get_annotations_in_lidar_coords(
            ego_pose_lidar,
            calibration_lidar,
            [
                self.nusc.get("sample_annotation", ann_token)
                for ann_token in sample["anns"]
            ],
            pc,
        )

        self.ns_lidar_pts_to_calculated_diff += annotations["pts_diff"]
        del annotations["pts_diff"]

        d = {
            "sample_id": sample_id,
            **annotations,
        }
        return d

    @staticmethod
    def _get_annotations_in_lidar_coords(
        vehicle_pose,
        calibration_lidar,
        annotations: typing.List[typing.Dict],
        point_cloud: np.ndarray,
    ):
        tf_vehicle_pose = transform_matrix(
            vehicle_pose["translation"], Quaternion(vehicle_pose["rotation"])
        )
        tf_lidar = transform_matrix(
            calibration_lidar["translation"], Quaternion(calibration_lidar["rotation"])
        )
        transf = np.linalg.inv(
            np.dot(
                np.dot(tf_vehicle_pose, tf_lidar),
                np.linalg.inv(NuscenesReader.turn_matrix),
            )
        )

        # rotate point cloud 90 degrees around z-axis,
        # so that x-axis faces front of vehicle
        x = point_cloud[:, 0:1]
        y = point_cloud[:, 1:2]
        point_cloud = np.concatenate((y, -x, point_cloud[:, 2:4]), axis=-1)

        dt = np.float32
        # numpy stack does not handle empty lists (== no annotated objects)
        if annotations:
            tfs = np.stack(
                [
                    transform_matrix(
                        annotation["translation"], Quaternion(annotation["rotation"])
                    )
                    for annotation in annotations
                ]
            )
            tfs = np.matmul(transf, tfs)
            # [(x y z) (w x y z) (l w h)]
            bboxes_spatial = np.empty(shape=(len(tfs), 10), dtype=dt)
            bboxes_spatial[:, 0:3] = tfs[:, :3, 3].astype(dt)
            bboxes_spatial[:, 3:7] = np.stack(
                [Quaternion(matrix=x).elements.astype(dt) for x in tfs]
            )
            # NuScenes format: width, length, height. Reorder to l w h
            bboxes_spatial[:, 7:10] = (
                np.stack([np.array(x["size"]) for x in annotations])[:, [1, 0, 2]]
            ).astype(dt)
        else:
            # [(x y z) (w x y z) (l w h)]
            bboxes_spatial = np.empty(shape=(0, 10), dtype=dt)
            tfs = np.empty(shape=(0, 4, 4), dtype=np.float64)

        cats_str = np.array([annotation["category_name"] for annotation in annotations])
        cats = NuscenesReader.np_str_to_cls(cats_str).astype(np.int64)
        lidar_pts = np.asarray([x["num_lidar_pts"] for x in annotations])

        total_points_per_box, mapping = assign_point_cloud_to_bounding_boxes(
            point_cloud,
            bounding_boxes_tfs=tfs,
            bounding_boxes_dims=bboxes_spatial[:, 7:10],
        )

        diff = np.sum(np.abs(lidar_pts - total_points_per_box))

        return {
            "point_cloud": point_cloud,
            "bounding_boxes_3d_spatial": bboxes_spatial,
            "bounding_boxes_3d_class": cats,
            "bounding_boxes_points": total_points_per_box.astype(np.int64),
            "bounding_boxes_3d_category": cats_str,
            "bounding_boxes_points_mapping": mapping,
            "bounding_boxes_3d_transforms": tfs,
            "pts_diff": diff,
        }


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
            "point_count": sample["bounding_boxes_points"],
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
        data = self.nusc_reader.read_internal(sample)
        frame_id = data["sample_id"]
        per_box = {
            k: v
            for k, v in data.items()
            if k
            in {
                "bounding_boxes_3d_spatial",
                "bounding_boxes_3d_class",
                "bounding_boxes_points",
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
                    "mapping": data["bounding_boxes_points_mapping"],
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
