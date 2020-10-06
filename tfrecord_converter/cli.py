#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Console script for tfrecord_converter.
"""

import pathlib
import inspect
import click
import numpy as np

from .cli_helper import (
    get_data_files_from_user_argument,
    get_files_flat,
    make_output_directory,
    process_split,
)
from .files_io import read_split, write_data_as_yaml
from .kitti_semantics_reader import KittiSemanticsIterator
from .kitti_raw_reader import KittiRawReader
from .reader_functions import reader_funcs
from .converter import convert_from_split, convert_single_set


@click.command()
@click.argument("files", nargs=-1)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--reader", required=True)
@click.option("--split", type=click.Path(exists=True), required=True)
@click.option("--chunk-size", default=-1)
@click.option("--shuffle/--no-shuffle", default=False)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--allow-missing/--no-allow-missing", default=False)
def process_hdf5(
    files, output, reader, split, chunk_size, shuffle, overwrite, allow_missing
):
    from .hdf5_reader import HDF5SubgroupIterable

    output = pathlib.Path(output)
    make_output_directory(output, overwrite)

    if reader not in reader_funcs:
        print("Unknown reader function '{}'.".format(reader))
        return

    # save all options of this function
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    options = {i: values[i] for i in args}
    write_data_as_yaml(options, str(output / "tf_dataset_flags.txt"))

    hdf5_input_files = get_data_files_from_user_argument(files)
    # it might be faster to process the files 'in order'
    hdf5_input_files = sorted(hdf5_input_files)

    split_dict = process_split(split, output, shuffle)

    # iterable over all HDF5 subgroups
    sample_keys = HDF5SubgroupIterable(hdf5_input_files)

    # convert. id func removes the leading / of the hdf5 subgroup
    convert_from_split(
        sample_keys,
        reader_funcs[reader](allow_missing=allow_missing),
        map_samples_to_id_func=lambda x: x.name[1:],
        output_dir=output,
        split=split_dict,
        samples_per_file=chunk_size,
    )


@click.command()
@click.argument("input_kitti_velodyne", nargs=1)
@click.argument("input_kitti_semantics", nargs=1)
@click.option("--sequence-id", default="kitti_raw")
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--chunk-size", default=-1)
@click.option("--overwrite/--no-overwrite", default=False)
def process_kitti_semantics(
    input_kitti_velodyne,
    input_kitti_semantics,
    sequence_id,
    output,
    chunk_size,
    overwrite,
):
    output = pathlib.Path(output)
    make_output_directory(output, overwrite)

    # save all options of this function
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    options = {i: values[i] for i in args}
    write_data_as_yaml(options, str(output / "tf_dataset_flags.txt"))

    # we have to assume that sorting the files creates corresponding pairs
    velodyne_input_files = get_files_flat(
        pathlib.Path(input_kitti_velodyne), ending=".bin"
    )
    velodyne_input_files = sorted(velodyne_input_files)
    semantic_input_files = get_files_flat(
        pathlib.Path(input_kitti_semantics), ending=".tfrecords"
    )
    semantic_input_files = sorted(semantic_input_files)

    generator = KittiSemanticsIterator(
        velodyne_input_files, semantic_input_files, sequence_id=sequence_id
    )
    convert_single_set(
        generator,
        generator.reader,
        output_dir=output,
        dataset_name="kitti_{}_lidar_semantics".format(sequence_id),
        samples_per_file=chunk_size,
        map_samples_to_id_func=generator.make_id_from_sample,
    )


@click.command()
@click.argument("nuscenes_root", nargs=1)
@click.argument("output_folder", type=click.Path(exists=False), required=True)
@click.option("--chunk-size", default=50)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--max-number-scenes", default=-1)
@click.option("--ns-version", default="v1.0-trainval")
def process_nuscenes(
    nuscenes_root, output_folder, chunk_size, overwrite, max_number_scenes, ns_version
):
    output_folder = pathlib.Path(output_folder)
    make_output_directory(output_folder, overwrite)

    from .nuscenes_reader import NuscenesReader

    reader = NuscenesReader(
        nuscenes_root=nuscenes_root,
        max_scenes=max_number_scenes if max_number_scenes > -1 else None,
        version=ns_version,
    )

    # write split for reference
    split_file_name = str(
        output_folder / "objects_gt_sampling_split_{}.yaml".format(reader.split["name"])
    )
    write_data_as_yaml(reader.split, split_file_name)

    convert_from_split(
        reader,
        reader_func=reader.read,
        map_samples_to_id_func=reader.make_sample_id,
        output_dir=output_folder,
        split=reader.split,
        samples_per_file=chunk_size,
    )


@click.command()
@click.argument("nuscenes_root", nargs=1)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--chunk-size", default=50)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--ns-version", default="v1.0-trainval")
def create_objects_from_nuscenes(
    nuscenes_root, output, chunk_size, overwrite, ns_version
):
    output = pathlib.Path(output)
    make_output_directory(output, overwrite)

    from .nuscenes_reader import NuscenesObjectsReader

    reader = NuscenesObjectsReader(nuscenes_root=nuscenes_root, version=ns_version)

    # write split for reference
    split_file_name = str(
        output / "tf_dataset_split_{}.yaml".format(reader.split["name"])
    )
    write_data_as_yaml(reader.split, split_file_name)

    convert_from_split(
        reader,
        reader_func=reader.read,
        map_samples_to_id_func=reader.make_sample_id,
        output_dir=output,
        split=reader.split,
        samples_per_file=chunk_size,
    )


@click.command()
@click.argument("files", nargs=-1)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--chunk-size", default=-1)
@click.option("--split", type=click.Path(exists=True), required=True)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--dataset-name", default="objects_gt_sampling")
def create_objects_from_hdf5(files, output, chunk_size, split, overwrite, dataset_name):
    """ Creates a dataset of all individual objects in a given HDF5 dataset.

    Splits data according to given split and creates tfrecords for each object class
    separately. Sample IDs will be: {sample}_#{obj_id:03d}_{category}
    """
    from .hdf5_reader import HDF5Objects

    output = pathlib.Path(output)
    make_output_directory(output, overwrite)

    # save all options of this function
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    options = {i: values[i] for i in args}
    write_data_as_yaml(options, output / "tf_dataset_flags.txt")

    hdf5_input_files = get_data_files_from_user_argument(files)
    # it might be faster to process the files 'in order'
    hdf5_input_files = sorted(hdf5_input_files)

    split_dict = read_split(split)
    splits_to_process = {k: set(split_dict["data"][k]) for k in ["train", "val"]}

    def select_split(sample_name):
        for k, v in splits_to_process.items():
            if sample_name in v:
                return k
        return None

    # iterable over all HDF5 subgroups
    objects = HDF5Objects(hdf5_input_files, select_split, dataset_name)

    # write split for reference
    split_file_name = str(output / "split_{}.yaml".format(objects.split["name"]))
    write_data_as_yaml(objects.split, split_file_name)

    # convert. id func removes the leading / of the hdf5 subgroup
    convert_from_split(
        objects,
        objects.read,
        output,
        map_samples_to_id_func=objects.map_to_id,
        split=objects.split,
        samples_per_file=chunk_size,
    )


@click.command()
@click.argument("kitti_raw_path", nargs=1)
@click.argument("kitti_raw_processed_path", nargs=1)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--split", type=click.Path(exists=True), required=True)
@click.option("--chunk-size", default=-1)
@click.option("--shuffle/--no-shuffle", default=False)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--performance/--no-performance")
def process_kitti_raw(
    kitti_raw_path,
    kitti_raw_processed_path,
    output,
    split,
    chunk_size,
    shuffle,
    overwrite,
    performance,
):
    """ Create dataset of KITTI autolabeled semantics.

    The split typically separates KITTI raw on a per-sequence basis.
    Sample IDs: kitti_raw_{day}_{seq:04d}_{frame:04d}
    """
    _process_kitti_raw_with(
        "kitti_semantics",
        kitti_raw_path,
        kitti_raw_processed_path,
        pathlib.Path(output),
        split,
        chunk_size,
        shuffle,
        overwrite,
        generator_kwargs={"performance": performance},
    )


@click.command()
@click.argument("kitti_raw_path", nargs=1)
@click.argument("kitti_raw_processed_path", nargs=1)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--split", type=click.Path(exists=True), required=True)
@click.option("--chunk-size", default=-1)
@click.option("--shuffle/--no-shuffle", default=False)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--cutoff-radius", default=np.inf)
def process_kitti_accumulated(
    kitti_raw_path,
    kitti_raw_processed_path,
    output,
    split,
    chunk_size,
    shuffle,
    overwrite,
    cutoff_radius,
):
    """

    The split typically separates KITTI raw on a per-sequence basis.
    Sample IDs: kitti_raw_{day}_{seq:04d}_{frame:04d}
    """
    _process_kitti_raw_with(
        "kitti_accumulated",
        kitti_raw_path,
        kitti_raw_processed_path,
        pathlib.Path(output),
        split,
        chunk_size,
        shuffle,
        overwrite,
        generator_kwargs={"cutoff_radius": cutoff_radius},
    )


@click.command()
@click.argument("kitti_raw_path", nargs=1)
@click.argument("kitti_odometry_path", nargs=1)
@click.argument("kitti_semantic_lidar_path", nargs=1)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--chunk-size", default=-1)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--testset/--no-testset", default=False)
def process_semantic_kitti(
    kitti_raw_path,
    kitti_odometry_path,
    kitti_semantic_lidar_path,
    output,
    chunk_size,
    overwrite,
    testset,
):
    """ Process KITTI LiDAR labels ('semantic-kitti')

    """
    output = pathlib.Path(output)
    make_output_directory(output, overwrite)

    # save all options of this function
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    options = {i: values[i] for i in args}
    write_data_as_yaml(options, str(output / "tf_dataset_flags.txt"))

    from .semantic_kitti_reader import SemanticKittiReader

    reader = SemanticKittiReader(
        kitti_raw_path, kitti_odometry_path, kitti_semantic_lidar_path, testset=testset
    )

    # write split for reference
    split_file_name = str(output / "split_{}.yaml".format(reader.split["name"]))
    write_data_as_yaml(reader.split, split_file_name)

    convert_from_split(
        reader,
        reader_func=reader.read,
        map_samples_to_id_func=reader.make_sample_id,
        output_dir=output,
        split=reader.split,
        samples_per_file=chunk_size,
    )


@click.command()
@click.argument("kitti_odometry_path", nargs=1)
@click.argument("semantic_kitti_path", nargs=1)
@click.argument("semantic_kitti_voxel_path", nargs=1)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--chunk-size", default=10)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--compress/--no-compress", default=False)
@click.option("--voxel_version", default="voxels_v2")
def process_semantic_kitti_voxels(
    kitti_odometry_path,
    semantic_kitti_path,
    semantic_kitti_voxel_path,
    output,
    chunk_size,
    overwrite,
    compress: bool,
    voxel_version: str,
):
    """ Process SemanticKITTI. Use voxelized data and accumulated point clouds.

    """
    output = pathlib.Path(output)
    make_output_directory(output, overwrite)

    # save all options of this function
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    options = {i: values[i] for i in args}
    write_data_as_yaml(options, str(output / "tf_dataset_flags.txt"))

    from .semantic_kitti_reader_voxels import SemanticKittiReaderVoxels

    reader = SemanticKittiReaderVoxels(
        kitti_odometry_path,
        semantic_kitti_path,
        semantic_kitti_voxel_path,
        voxel_version=voxel_version,
    )

    # write split for reference
    split_file_name = str(output / "split_{}.yaml".format(reader.split["name"]))
    write_data_as_yaml(reader.split, split_file_name)

    convert_from_split(
        reader,
        reader_func=reader.read,
        map_samples_to_id_func=reader.make_sample_id,
        output_dir=output,
        split=reader.split,
        samples_per_file=chunk_size,
        compression=compress,
    )


def _process_kitti_raw_with(
    sequence_generator_name,
    kitti_raw_path,
    kitti_raw_processed_path,
    output: pathlib.Path,
    split,
    chunk_size,
    shuffle,
    overwrite,
    generator_kwargs=None,
):
    """ Helper function. Calls the KittiRawReader
    with a given data reader from a sequence

    """
    make_output_directory(output, overwrite)

    # save all options of this function
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    options = {i: values[i] for i in args}
    write_data_as_yaml(options, str(output / "tf_dataset_flags.txt"))

    split_dict = process_split(split, output, shuffle)

    reader = KittiRawReader(
        kitti_raw_path,
        kitti_raw_processed_path,
        split_dict,
        sequence_data_generator=sequence_generator_name,
        generator_kwargs=generator_kwargs,
    )

    # convert
    convert_from_split(
        reader,
        reader.read,
        map_samples_to_id_func=reader.make_sample_id,
        output_dir=output,
        split=split_dict,
        samples_per_file=chunk_size,
    )
