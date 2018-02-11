#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def assign_point_cloud_to_bounding_boxes(point_cloud: np.ndarray,
                                         bounding_boxes_tfs: np.ndarray,
                                         bounding_boxes_dims: np.ndarray,):
    """Checks which points are within each bounding box.

    :param point_cloud: [N x 4]
    :param bounding_boxes_tfs: [N x 4 x 4] Homogeneous transforms in point cloud coordinate system
    :param bounding_boxes_dims: [N x 3] Bounding box dimensions (length, width, height)
    :return: Tuple: number of points within each bounding box, mapping from point cloud to
    bounding box ID

    """
    total_points_per_box = np.zeros(shape=(len(bounding_boxes_tfs), ), dtype=np.int64)
    mapping = np.full(shape=(len(point_cloud), ), fill_value=-1, dtype=np.int64)

    for i, (box, dims) in enumerate(zip(bounding_boxes_tfs, bounding_boxes_dims)):

        transf = np.linalg.inv(box)
        pc = np.concatenate((point_cloud[:, :3], np.ones(shape=(point_cloud.shape[0], 1),
                                                         dtype=point_cloud.dtype)), axis=-1)

        pc = np.dot(pc, np.transpose(transf))
        in_box = np.all(np.abs(pc[:, :3]) < (dims / 2), axis=-1)
        total_points_per_box[i] = np.count_nonzero(in_box)
        mapping[in_box] = i
    return total_points_per_box, mapping



