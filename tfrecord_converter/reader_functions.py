#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .hdf5_reader import merge_readers,\
    read_boxes_3d,\
    read_boxes_kitti,\
    read_rgb_image,\
    read_pointcloud,\
    read_2d_boxes,\
    read_lidar_to_image_calibration,\
    read_sample_id,\
    read_lidar_image_info,\
    read_calibration,\
    read_boxes_3d_and_box_numbers,\
    read_sequence_info


"""Read KITTI Object data with calibration. Without RGB images."""
kitti_object = merge_readers([
    read_pointcloud,
    read_sample_id,
    read_boxes_kitti,
    read_calibration,
    read_rgb_image,
])


"""Read KITTI Object TEST data with calibration. Without RGB images."""
kitti_object_test = merge_readers([
    read_pointcloud,
    read_sample_id,
    read_calibration,
])


kitti_object_seqs = merge_readers([
    read_pointcloud,
    read_sample_id,
    read_boxes_kitti,
    read_calibration,
    read_sequence_info,
])


# define reader functions
reader_funcs = {
    'kitti_object': kitti_object,
    'kitti_object_test': kitti_object_test,
    'kitti_pointcloud': read_pointcloud,
    'kitti_object_seqs': kitti_object_seqs,
    'kitti_boxes_and_lidar': merge_readers([
        read_sample_id,
        read_boxes_3d,
        read_pointcloud
    ]),
    'kitti_licam': merge_readers([
        read_sample_id,
        read_boxes_3d,
        read_pointcloud,
        read_rgb_image,
        read_2d_boxes,
        read_lidar_to_image_calibration
    ]),
    'kitti_licam_no_annotations': merge_readers([
        read_sample_id,
        read_pointcloud,
        read_rgb_image,
        read_lidar_to_image_calibration
    ]),
}
