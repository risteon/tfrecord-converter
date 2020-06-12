#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import functools
import logging
import h5py
import pyquaternion


from .common import ObjectClass
from .algorithm import assign_point_cloud_to_bounding_boxes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


OBJ_CLASS_FROM_STRING = {
    "Unknown": ObjectClass.UNKNOWN,
    "DontCare": ObjectClass.DONTCARE,
    "PassengerCar": ObjectClass.PASSENGER_CAR,
    "Pedestrian": ObjectClass.PEDESTRIAN,
    "Van": ObjectClass.VAN,
    "Truck": ObjectClass.TRUCK,
    "Person_sitting": ObjectClass.PERSON_SITTING,
    "Cyclist": ObjectClass.CYCLIST,
    "Tram": ObjectClass.TRAM,
    "Misc": ObjectClass.MISC,
    "LargeVehicle": ObjectClass.LARGE_VEHICLE,
    "RidableVehicle": ObjectClass.CYCLIST,  # CMORE boxes. Mapping to cyclist.
    "Trailer": ObjectClass.TRAILER,  # CMORE boxes. Mapping to trailer.
}


class HDF5SubgroupIterableInBlocks:
    def __init__(self, filename):
        self.file = h5py.File(filename, "r")
        self.iterator_keys = iter(self.file.keys())
        self.max_num_samples = 0
        self.samples_generated = 0

    class Iterator:
        def __init__(self, keys_iter, num_examples, file_ref):
            self.keys_iter = keys_iter
            self.num_examples = num_examples
            self.counter = 0
            self.file_ref = file_ref

        def __next__(self):
            if self.counter < self.num_examples:
                self.counter += 1
                return self.file_ref[next(self.keys_iter)]
            else:
                raise StopIteration

    def set_range(self, num_samples):
        self.max_num_samples = num_samples

    def __iter__(self):
        r = HDF5SubgroupIterableInBlocks.Iterator(
            self.iterator_keys, self.max_num_samples - self.samples_generated, self.file
        )
        self.samples_generated = self.max_num_samples
        return r

    def __len__(self):
        return self.max_num_samples - self.samples_generated

    def __del__(self):
        self.file.close()


class HDF5SubgroupIterable:
    """Iterate through all subgroups of the given files. One file after another."""

    def __init__(self, files):
        self._current_file = None
        self._iter_files = iter(files)
        self._iter_keys = None

    def __next__(self):
        try:
            group = next(self._iter_keys)
            return self._current_file[group]
        except StopIteration:
            self._next_file()
            return self.__next__()

    def __iter__(self):
        self._next_file()
        return self

    def _next_file(self):
        self._current_file = h5py.File(next(self._iter_files), "r")
        self._iter_keys = iter(self._current_file.keys())


class HDF5ObjectIterator:
    def __init__(self, files, select_split):
        self.sample_iterator = iter(HDF5SubgroupIterable(files))
        self.select_split = select_split
        self.sample = None
        self.split_name = None

    def __iter__(self):
        self._next_sample()
        return self

    def __next__(self):
        """Returns obj, split_name"""
        try:
            return next(self.object_iter), self.split_name
        except StopIteration:
            self._next_sample()
            return self.__next__()

    def _next_sample(self):
        self.split_name = None
        while self.split_name is None:
            self.sample = next(self.sample_iterator)
            self.split_name = self.select_split(self.sample.name[1:])

        self.object_list = np.array(self.sample["bounding_boxes_3d"])
        self.object_iter = iter(self.object_list)


class HDF5ObjectIteratorWithSegmentation:

    HOMOGENEOUS_ARRAY = np.asarray([[[0, 0, 0, 1]]], dtype=np.float64)

    def __init__(self, files, select_split):
        self.sample_iterator = iter(HDF5SubgroupIterable(files))
        self.select_split = select_split
        self.sample = None
        self.split_name = None

    def __iter__(self):
        self._next_sample()
        return self

    def __next__(self):
        """Returns obj, split_name"""
        try:
            return next(self.data), self.split_name, self.sample_id
        except StopIteration:
            self._next_sample()
            return self.__next__()

    def _next_sample(self):
        self.split_name = None
        while self.split_name is None:
            self.sample = next(self.sample_iterator)
            self.split_name = self.select_split(self.sample.name[1:])

        self._process_sample()
        self.data = iter(
            zip(
                self.object_list,
                self.tfs,
                self.total_points_per_box,
                self.point_cloud_mapping,
            )
        )

    def _process_sample(self):
        self.object_list = np.array(self.sample["bounding_boxes_3d"])
        self.object_iter = iter(self.object_list)
        point_cloud = np.array(self.sample["point_cloud[velodyne]"])
        rot = np.stack(
            [
                pyquaternion.Quaternion(x["box_orientation"]).rotation_matrix
                for x in self.object_list
            ]
        )
        transl = np.stack([x["box_position"] for x in self.object_list])
        dims = np.stack([x["box_size"] for x in self.object_list])
        rot_transl = np.concatenate((rot, np.expand_dims(transl, axis=-1)), axis=-1)
        self.tfs = np.concatenate(
            (rot_transl, np.tile(self.HOMOGENEOUS_ARRAY, [transl.shape[0], 1, 1])),
            axis=1,
        )
        self.total_points_per_box, mapping = assign_point_cloud_to_bounding_boxes(
            point_cloud, self.tfs, dims
        )
        self.point_cloud_mapping = [
            point_cloud[mapping == i] for i in range(len(self.object_list))
        ]
        self.sample_id = self.sample.name[1:]


class HDF5Objects:

    SAMPLE_ID_PATTERN = "{sample}_#{obj_id:03d}_{category}"

    def __init__(
        self, files, select_split=lambda _: "", dataset_name="objects_gt_sampling"
    ):
        self.objects = HDF5ObjectIteratorWithSegmentation(files, select_split)
        self.select_split = select_split
        self.dataset_name = dataset_name
        self.split = self._build_split(HDF5ObjectIterator(files, select_split))

    def _build_split(self, object_iterator):
        split = {
            "name": self.dataset_name,
            "data": {},
        }
        d = split["data"]

        # single run through data to build the split
        for dataset_obj, base_split in object_iterator:
            obj_cls = dataset_obj["object_class"].decode("utf-8").lower()
            if base_split:
                split_name = obj_cls + "_" + base_split
            else:
                split_name = obj_cls
            box_id = dataset_obj["box_id"]
            if split_name not in d:
                d[split_name] = []
            d[split_name].append(
                HDF5Objects.SAMPLE_ID_PATTERN.format(
                    sample=object_iterator.sample.name[1:],
                    obj_id=box_id,
                    category=obj_cls,
                )
            )
        return split

    def __iter__(self):
        return self.objects.__iter__()

    @staticmethod
    def read(sample, _sample_id):
        data, base_split, sample_id = sample
        obj_cls = data[0]["object_class"].decode("utf-8")
        obj_cls_value = np.asarray(OBJ_CLASS_FROM_STRING[obj_cls].value, dtype=np.int64)
        obj_cls = obj_cls.lower()
        box_id = data[0]["box_id"]

        d = {
            "sample_id": HDF5Objects.SAMPLE_ID_PATTERN.format(
                sample=sample_id, obj_id=box_id, category=obj_cls
            ).encode("utf-8"),
            "transform": data[1].flatten().astype(np.float32),
            "bounding_box_3d": np.concatenate(
                (
                    data[0]["box_position"],
                    data[0]["box_orientation"],
                    data[0]["box_size"],
                ),
                axis=0,
            ).astype(np.float32),
            "point_count": data[2],
            "points": data[3].flatten(),
            "object_cls": obj_cls_value,
        }
        return d

    @staticmethod
    def map_to_id(sample):
        data, base_split, sample_id = sample
        obj_cls = data[0]["object_class"].decode("utf-8").lower()
        box_id = data[0]["box_id"]
        return HDF5Objects.SAMPLE_ID_PATTERN.format(
            sample=sample_id, obj_id=box_id, category=obj_cls
        )


def read_pointcloud(h5_obj, _sample_id, sensor="velodyne"):
    values = {
        "point_cloud": np.array(h5_obj["point_cloud[{}]".format(sensor)]).flatten(),
    }
    return values


def read_sample_id(h5_obj, _sample_id):
    return {
        "sample_id": h5_obj.name[1:].encode("utf-8"),
    }


def read_boxes_kitti(h5_obj, _sample_id, variant_points_meta="velodyne"):
    """Read 3D, 2D, and meta.

    Single reader function to make sure that the of each output array is identical.
    """
    h5_boxes_3d = h5_obj["bounding_boxes_3d"]
    h5_boxes_2d = h5_obj["bounding_boxes_2d"]
    h5_boxes_meta = h5_obj["bounding_boxes_meta"]
    if not variant_points_meta:
        key = "bounding_boxes_points"
    else:
        key = "bounding_boxes_points[{}]".format(variant_points_meta)
    h5_boxes_points = h5_obj[key]
    if not variant_points_meta:
        key = "point_cloud_box_mapping"
    else:
        key = "point_cloud_box_mapping[{}]".format(variant_points_meta)
    h5_point_cloud_mapping = h5_obj[key]

    n_boxes = len(h5_boxes_3d)
    if n_boxes != len(h5_boxes_2d) or n_boxes != len(h5_boxes_meta):
        raise RuntimeError("Number of boxes does not match.")

    bboxes_spatial_3d = np.ndarray(shape=(n_boxes, 10), dtype=np.float32)
    bboxes_spatial_2d = np.ndarray(shape=(n_boxes, 4), dtype=np.float32)
    bboxes_meta = np.ndarray(shape=(n_boxes, 5), dtype=np.float32)
    bboxes_class = np.ndarray(shape=(n_boxes,), dtype=np.int64)

    id_meta = np.array(h5_boxes_meta)["box_id"]
    id_points = np.array(h5_boxes_points)["box_id"]
    id_2d = np.array(h5_boxes_2d)["box_id"]
    id_3d = np.array(h5_boxes_3d)["box_id"]
    if (
        not np.array_equal(id_meta, id_3d)
        or not np.array_equal(id_meta, id_2d)
        or not np.array_equal(id_meta, id_points)
    ):
        raise NotImplementedError("Box IDs of meta, 2d, 3d do not match.")

    for i in range(n_boxes):
        box_3d = h5_boxes_3d[i]
        box_2d = h5_boxes_2d[i]
        box_meta = h5_boxes_meta[i]

        bboxes_class[i] = OBJ_CLASS_FROM_STRING[
            box_3d["object_class"].decode("utf-8")
        ].value
        bboxes_spatial_3d[i, :3] = box_3d["box_position"]
        bboxes_spatial_3d[i, 3:7] = box_3d["box_orientation"]
        if (bboxes_spatial_3d[i, 3:7] > 1.0).any() or (
            bboxes_spatial_3d[i, 3:7] < -1.0
        ).any():
            raise ValueError("Quaternion out of range {}, {}".format(h5_obj.name, i))
        bboxes_spatial_3d[i, 7:] = box_3d["box_size"]

        bboxes_spatial_2d[i, :] = box_2d["bounding_box"]

        # saved as 255 instead of -1?
        occlusion = box_meta["occlusion_level"]
        if occlusion == 255:
            occlusion = -1.0
        else:
            occlusion = float(occlusion)

        bboxes_meta[i, 0] = occlusion
        bboxes_meta[i, 1] = box_meta["truncation_level"]
        bboxes_meta[i, 2] = box_meta["box_difficulty"]
        bboxes_meta[i, 3] = box_meta["box_rotation_y"]
        bboxes_meta[i, 4] = box_meta["observation_alpha"]

    values = {
        "bounding_boxes_3d_class": bboxes_class,
        "bounding_boxes_3d_spatial": bboxes_spatial_3d.flatten(),
        "bounding_boxes_2d_spatial": bboxes_spatial_2d.flatten(),
        "bounding_boxes_meta": bboxes_meta.flatten(),
        "bounding_boxes_points": np.array(h5_boxes_points)["total_points"].astype(
            np.int64
        ),
        "point_cloud_box_mapping": np.array(h5_point_cloud_mapping).astype(np.int64),
    }
    return values


def read_boxes_3d(h5_obj, _sample_id, variant=None):
    """This is for the current standard hdf5 dataset format."""

    if not variant:
        key = "bounding_boxes_3d"
    else:
        key = "bounding_boxes_3d[{}]".format(variant)

    label = h5_obj[key]
    n_boxes = len(label)
    bboxes_spatial = np.ndarray(shape=(n_boxes, 10), dtype=np.float32)
    bboxes_class = np.ndarray(shape=(n_boxes,), dtype=np.int64)

    for i in range(len(label)):
        box = label[i]
        bboxes_class[i] = OBJ_CLASS_FROM_STRING[
            box["object_class"].decode("utf-8")
        ].value
        bboxes_spatial[i, :3] = box["box_position"]
        bboxes_spatial[i, 3:7] = box["box_orientation"]
        if (bboxes_spatial[i, 3:7] > 1.0).any() or (
            bboxes_spatial[i, 3:7] < -1.0
        ).any():
            raise ValueError("Quaternion out of range {}, {}".format(h5_obj.name, i))

        bboxes_spatial[i, 7:] = box["box_size"]

    values = {
        "bounding_boxes_3d_class": bboxes_class,
        "bounding_boxes_3d_spatial": bboxes_spatial.flatten(),
    }
    return values


def read_boxes_3d_and_box_numbers(
    h5_obj, _sample_id, variant=None, variant_points_meta=None
):
    """This is for the current standard hdf5 dataset format."""

    if not variant:
        key = "bounding_boxes_3d"
    else:
        key = "bounding_boxes_3d[{}]".format(variant)
    h5_boxes_3d = h5_obj[key]

    if not variant_points_meta:
        key = "bounding_boxes_points"
    else:
        key = "bounding_boxes_points[{}]".format(variant_points_meta)
    h5_boxes_points = h5_obj[key]

    id_meta = np.array(h5_boxes_points)["box_id"]
    id_3d = np.array(h5_boxes_3d)["box_id"]
    if not np.array_equal(id_meta, id_3d):
        raise NotImplementedError("Box IDs of 3d and points do not match.")

    n_boxes = len(h5_boxes_3d)
    bboxes_spatial = np.ndarray(shape=(n_boxes, 10), dtype=np.float32)
    bboxes_class = np.ndarray(shape=(n_boxes,), dtype=np.int64)

    for i in range(len(h5_boxes_3d)):
        box = h5_boxes_3d[i]
        bboxes_class[i] = OBJ_CLASS_FROM_STRING[
            box["object_class"].decode("utf-8")
        ].value
        bboxes_spatial[i, :3] = box["box_position"]
        bboxes_spatial[i, 3:7] = box["box_orientation"]
        if (bboxes_spatial[i, 3:7] > 1.0).any() or (
            bboxes_spatial[i, 3:7] < -1.0
        ).any():
            raise ValueError("Quaternion out of range {}, {}".format(h5_obj.name, i))

        bboxes_spatial[i, 7:] = box["box_size"]

    values = {
        "bounding_boxes_3d_class": bboxes_class,
        "bounding_boxes_3d_spatial": bboxes_spatial.flatten(),
        "bounding_boxes_points": np.array(h5_boxes_points)["total_points"].astype(
            np.int64
        ),
    }
    return values


def read_rgb_image(h5_obj, _sample_id, sensor="cam_2"):
    """This is for the current standard hdf5 dataset format."""
    image = np.array(h5_obj["image_rgb[{}]".format(sensor)])
    values = {
        "rgb_image_{}".format(sensor): image.flatten().tobytes(),
        "rgb_image_{}_dim".format(sensor): np.asarray(image.shape, dtype=np.int64),
    }
    return values


def read_2d_boxes(h5_obj, _sample_id):
    label = h5_obj["bounding_boxes_2d"]
    n_boxes = len(label)
    bboxes_spatial = np.ndarray(shape=(n_boxes, 4), dtype=np.float32)
    bboxes_class = np.ndarray(shape=(n_boxes,), dtype=np.int64)

    for i in range(len(label)):
        box = label[i]
        bboxes_class[i] = OBJ_CLASS_FROM_STRING[
            box["object_class"].decode("utf-8")
        ].value
        bboxes_spatial[i] = box["bounding_box"]

    values = {
        "bounding_boxes_2d_class": bboxes_class,
        "bounding_boxes_2d_spatial": bboxes_spatial.flatten(),
    }
    return values


def read_lidar_to_image_calibration(h5_obj, _sample_id):

    calib = h5_obj["coordinate_frames"]
    values = {
        "lidar_to_cam_2": np.dot(
            np.array(calib["P2"]), np.array(calib["velo_to_cam_2"])
        )
        .flatten()
        .astype(np.float32),
    }
    return values


def read_calibration(h5_obj, _sample_id):
    calib = h5_obj["coordinate_frames"]
    values = {}
    for k in calib.keys():
        c = calib[k]
        try:
            frame_parent = c.attrs["frame_id_parent"].decode("utf-8")
        except AttributeError:
            frame_parent = c.attrs["frame_id_parent"]
        try:
            frame_child = c.attrs["frame_id_child"].decode("utf-8")
        except AttributeError:
            frame_child = c.attrs["frame_id_child"]
        key = "calib_{}<-{}".format(frame_parent, frame_child)
        value = np.array(c).flatten().astype(np.float32)
        values[key] = value
    return values


def read_lidar_image_info(h5_obj, _sample_id):
    values = {
        "img_dim": np.array(h5_obj["lidar_image_dimensions"]).astype(np.int64),
        "indices": np.array(h5_obj["lidar_image_indices"]).astype(np.int64),
    }
    return values


def read_sequence_info(h5_obj, _sample_id):
    def read_attr(attr):
        try:
            return h5_obj.attrs[attr].decode("utf-8")
        except AttributeError:
            return h5_obj.attrs[attr]

    sequence_id = read_attr("sequence_id")
    sequence_length: np.uint32 = read_attr("sequence_length")
    dsin: np.int32 = read_attr("dsin")
    return {
        "dsin": dsin.astype(np.int64),
        "sequence_length": sequence_length.astype(np.int64),
        "sequence_id": sequence_id.encode("utf-8"),
    }


def merge_readers(readers):
    """I'm sorry for this mess."""

    def make(allow_missing=False):
        def read(h5_obj, sample_id, *args, **kwargs):
            def reduce_func(values, read_func):
                try:
                    v = read_func(h5_obj, sample_id, *args, **kwargs)
                # only allow KeyErrors!
                except KeyError as e:
                    if allow_missing:
                        return values
                    raise e
                return {**values, **v}

            return functools.reduce(reduce_func, readers, {})

        return read

    return make
