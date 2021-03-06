#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converts data to TFRecords file format with example protos.
"""

import pathlib
import math
import collections
import logging
import typing

import tensorflow as tf
import numpy as np
import tqdm

from .files_io import write_data_as_yaml


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def create_data_description(data, **kwargs):

    feature_dict = make_feature_dict(data)
    types = ["bytes_list", "float_list", "int64_list"]

    def to_string(feature) -> str:
        return next(
            iter(
                filter(lambda x: x[1], zip(types, [feature.HasField(x) for x in types]))
            )
        )[0]

    type_dict = {k: to_string(v) for k, v in feature_dict.items()}
    description_string = "\n".join(
        ["{} ({})".format(k, v) for k, v in type_dict.items()]
    )
    description_string += "\nMeta:\n" + "\n".join(
        "{}: {}".format(k, v) for k, v in kwargs.items()
    )
    description_dict = {
        "meta": kwargs,
        "entries": type_dict,
    }
    return description_string, description_dict


def make_feature_dict(data):
    def _float_value_feature(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[v]))

    def _int_value_feature(v):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

    def _float_array_feature(array):
        return tf.train.Feature(float_list=tf.train.FloatList(value=array))

    def _int_array_feature(array):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=array))

    def _bytes_feature(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    feature_dict = {}
    for key, value in data.items():

        if isinstance(value, tf.Tensor):
            if value.shape.ndims > 1:
                raise TypeError(
                    "Only one-dimensional arrays supported."
                    "Conflicting entry: {}".format(key)
                )
            if value.dtype == tf.string:
                if value.shape != ():
                    raise TypeError(
                        "Only scalar string tensors supported."
                        "Conflicting entry: {}".format(key)
                    )
                feature = _bytes_feature(value.numpy())
            elif value.dtype == tf.dtypes.float32:
                feature = (
                    _float_array_feature(value)
                    if value.ndim == 1
                    else _float_value_feature(value)
                )
            elif value.dtype == tf.dtypes.int64:
                feature = (
                    _int_array_feature(value)
                    if value.ndim == 1
                    else _int_value_feature(value)
                )
            else:
                raise TypeError(
                    "Unsupported dtype. Only int64 and float32 currently supported."
                    " Key: {}. Data type: {}".format(key, type(value))
                )

        elif isinstance(value, np.ndarray):
            if value.ndim > 1:
                raise TypeError(
                    "Only one-dimensional arrays supported."
                    "Conflicting entry: {}".format(key)
                )

            if value.dtype == np.float32:
                feature = (
                    _float_array_feature(value)
                    if value.ndim == 1
                    else _float_value_feature(value)
                )
            elif value.dtype == np.int64:
                feature = (
                    _int_array_feature(value)
                    if value.ndim == 1
                    else _int_value_feature(value)
                )
            else:
                raise TypeError(
                    "Unsupported dtype. Only int64 and float32 currently supported."
                    " Key: {}. Data type: {}".format(key, type(value))
                )

        elif isinstance(value, np.float32):
            feature = _float_value_feature(value)
        elif isinstance(value, np.int64):
            feature = _int_value_feature(value)
        elif isinstance(value, bytes):
            feature = _bytes_feature(value)
        elif isinstance(value, str):
            feature = _bytes_feature(value.encode("utf-8"))
        elif isinstance(value, int):
            feature = _int_value_feature(np.int64(value))
        else:
            raise TypeError(
                "Unsupported data type returned. Key: {}. Data type: {}".format(
                    key, type(value)
                )
            )

        feature_dict[key] = feature
    return feature_dict


def make_example(data):
    feature_dict = make_feature_dict(data)
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def tf_writer_coroutine(
    reader_func: typing.Union[
        typing.Callable[[typing.Any, str], typing.Dict[str, typing.Any]], None
    ],
    output_dir: pathlib.Path,
    dataset_name: str,
    total_number_of_samples: int,
    samples_per_file: int,
    samples_to_write=None,
    file_logger_obj=None,
    progress_bar=None,
    tfrecord_options=None,
):
    """

    :param reader_func: Must return a data ndarray and accept a sample
    :param output_dir:
    :param dataset_name:
    :param total_number_of_samples:
    :param samples_per_file: How many samples to pack into one output file.
    :param samples_to_write: If given, writer will only write these samples
    in the given order.
    :param file_logger_obj: If given, log all written files to this object
    :param progress_bar:
    :param tfrecord_options:
    :return:
    """

    # calc number of files
    total_number_of_files = ((total_number_of_samples - 1) // samples_per_file) + 1

    # filename formatting
    digits = math.ceil(math.log10(total_number_of_files + 1))
    f_str = "0{}d".format(digits)
    filename_template = dataset_name + "_{{:{}}}_of_{{:{}}}.tfrecords".format(
        f_str, f_str
    )

    def make_filepath(c: int, total: int) -> pathlib.Path:
        return output_dir / filename_template.format(c, total)

    data = None
    processed_counter = 0

    samples_available = {}
    if samples_to_write is not None:
        samples_queue = collections.deque(samples_to_write)
    else:
        # simple mode: just write all samples the order they come in
        samples_queue = collections.deque()

    if progress_bar is not None:
        progress_bar.refresh()

    try:
        for file_counter in range(total_number_of_files):  # < loop over all files

            f_path = make_filepath(file_counter, total_number_of_files)
            if file_logger_obj:
                file_logger_obj.send((f_path, dataset_name))

            logger.debug("Opened {}".format(str(f_path)))
            with tf.io.TFRecordWriter(str(f_path), options=tfrecord_options) as writer:

                samples_written_in_file = 0
                while True:  # < loop over samples in current file
                    # < loop to write available samples
                    while samples_queue and samples_queue[0] in samples_available:
                        logger.debug(
                            "Writing sample {} to {} set".format(
                                samples_queue[0], dataset_name
                            )
                        )
                        sample_id = samples_queue.popleft()

                        if reader_func is not None:
                            try:
                                data = reader_func(
                                    samples_available[sample_id], sample_id
                                )
                            except (KeyError, OSError) as e:
                                logger.warning(
                                    "Could not read {}. Skipping.".format(sample_id)
                                )
                                raise e

                            example = make_example(data)
                            serialized = example.SerializeToString()
                        else:
                            # raw data mode
                            serialized = samples_available[sample_id]

                        writer.write(serialized)

                        # discard sample if it is not needed anymore
                        if sample_id not in samples_queue:
                            del samples_available[sample_id]

                        if progress_bar is not None:
                            progress_bar.update(n=1)
                        processed_counter += 1

                        samples_written_in_file += 1
                        if samples_written_in_file == samples_per_file:
                            logger.debug(
                                "{} samples written to '{}'.".format(
                                    samples_written_in_file, str(f_path)
                                )
                            )
                            break
                    else:
                        sample, sample_id = yield
                        samples_available[sample_id] = sample
                        # if simple mode: always write this sample next
                        if samples_to_write is None:
                            samples_queue.append(sample_id)

                        continue
                    break

        # this is not allowed to yield as all samples have been already processed.
        # So just wait for GeneratorExit
        _ = yield
        assert False

    except GeneratorExit:
        # Finalize write meta data
        if samples_queue:
            logger.warning(
                "Not all samples were received for dataset '{}'. "
                "{} samples are missing".format(dataset_name, len(samples_queue))
            )

        if data is None:
            return

        logger.info(
            "Total written files to split {} is {}.".format(
                dataset_name, processed_counter
            )
        )

        # write dataset description
        desc_string, desc_dict = create_data_description(
            data, no_samples=total_number_of_samples, no_files=total_number_of_files
        )
        data_desc_name = str(
            output_dir / "tf_dataset_description_{}.txt".format(dataset_name)
        )
        with open(data_desc_name, "w") as output_dataset_desc_file:
            output_dataset_desc_file.write(desc_string)
        write_data_as_yaml(
            desc_dict,
            str(output_dir / "tf_dataset_description_{}.yaml".format(dataset_name)),
        )


def file_logger(
    target_log_dir: pathlib.Path,
    dataset_root_dir: pathlib.Path,
    dataset_name="dataset",
    splits=None,
):
    file_dict = collections.defaultdict(list)
    if splits is not None:
        file_dict.update({k: [] for k in splits})

    try:
        while True:
            file, split_name = yield
            file_dict[split_name].append(file)

    except GeneratorExit:
        split_lists = []
        for split_name, file_list in file_dict.items():
            # write file_list
            f_list_name = target_log_dir / "{}.dataset".format(split_name)
            split_lists.append(f_list_name)
            f_list_name = str(f_list_name)

            with open(f_list_name, "w") as output_tf_list_file:
                for f in file_list:
                    output_tf_list_file.write(
                        str(pathlib.Path(f.resolve()).relative_to(dataset_root_dir))
                        + "\n"
                    )

            logger.info("List of files written to {}".format(f_list_name))

        split_list_name = str(target_log_dir / "{}.lists".format(dataset_name))
        with open(split_list_name, "w") as output_tf_list_file:
            for f in split_lists:
                output_tf_list_file.write(
                    str(pathlib.Path(f.resolve()).relative_to(dataset_root_dir)) + "\n"
                )
        logger.info("List of splits written to {}".format(split_list_name))


def convert_single_set(
    sample_objects,
    reader_func: typing.Callable[[typing.Any], typing.Dict[str, typing.Any]],
    output_dir: pathlib.Path,
    dataset_name,
    map_samples_to_id_func=lambda _: None,
    samples_per_file=None,
    total_number_of_samples=None,
):

    if not sample_objects:
        raise RuntimeError("No files")

    if total_number_of_samples is None:
        total_number_of_samples = len(sample_objects)
    if total_number_of_samples == 0:
        return

    if samples_per_file is None or samples_per_file < 1:
        samples_per_file = total_number_of_samples

    output_dir = output_dir.resolve()
    dataset_files_dir = output_dir / "dataset_files"
    dataset_files_dir.mkdir(exist_ok=True)
    file_logger_obj = file_logger(
        dataset_files_dir, output_dir, dataset_name=dataset_name
    )
    file_logger_obj.send(None)

    writer = tf_writer_coroutine(
        reader_func,
        output_dir,
        dataset_name,
        total_number_of_samples,
        samples_per_file,
        file_logger_obj=file_logger_obj,
    )
    writer.send(None)
    for sample in sample_objects:
        sample_id = map_samples_to_id_func(sample)
        writer.send((sample, sample_id))
    writer.close()


def convert_from_split(
    sample_objects: collections.Iterable,
    reader_func: typing.Callable[[typing.Any], typing.Dict[str, typing.Any]],
    output_dir: pathlib.Path,
    map_samples_to_id_func,
    split: dict,
    samples_per_file=None,
    progress_bar: bool = True,
    compression: bool = False,
):
    assert reader_func is not None
    assert output_dir is not None

    s_list = {k: set(v) for k, v in split["data"].items()}

    if not s_list:
        print("Empty split. Nothing to do.")
        return

    output_dir = output_dir.resolve()
    dataset_paths: {str: pathlib.Path} = {k: output_dir / k for k in s_list}
    for p in dataset_paths.values():
        try:
            p.mkdir(exist_ok=True)
        except FileExistsError:
            pass

    if samples_per_file is None or samples_per_file == -1:
        samples_per_file = 1000000000

    try:
        dataset_name = split["name"]
    except KeyError:
        dataset_name = "dataset"

    dataset_files_dir = output_dir / "dataset_files"
    dataset_files_dir.mkdir(exist_ok=True)
    file_logger_obj = file_logger(
        dataset_files_dir, output_dir, dataset_name=dataset_name, splits=s_list.keys(),
    )
    file_logger_obj.send(None)

    max_split_name_len = max(map(len, s_list.keys()))
    if progress_bar:
        progress_bars = {
            k: tqdm.tqdm(
                desc=k.rjust(max_split_name_len),
                position=i,
                total=len(s_list[k]),
                unit="samples",
                smoothing=0.3,
            )
            for i, k in enumerate(s_list)
        }
    else:
        progress_bars = {k: None for k in s_list}

    options = tf.io.TFRecordOptions(
        compression_type="GZIP" if compression else None,
        flush_mode=None,
        input_buffer_size=None,
        output_buffer_size=None,
        window_bits=None,
        compression_level=None,
        compression_method=None,
        mem_level=None,
        compression_strategy=None,
    )

    writers = {
        k: tf_writer_coroutine(
            reader_func,
            output_dir=dataset_paths[k],
            dataset_name=k,
            total_number_of_samples=len(s_list[k]),
            samples_per_file=samples_per_file,
            samples_to_write=split["data"][k],
            file_logger_obj=file_logger_obj,
            progress_bar=progress_bars[k],
            tfrecord_options=options,
        )
        for i, k in enumerate(s_list)
    }

    for w in writers.values():
        w.send(None)

    k = None
    for sample in sample_objects:
        sample_id = map_samples_to_id_func(sample)
        for k, v in s_list.items():
            if sample_id in v:
                v.remove(sample_id)
                try:
                    writers[k].send((sample, sample_id))
                except StopIteration:
                    if v:
                        # Writer has finished but samples still available. Not allowed.
                        assert False
                break
        else:
            logger.debug(
                "Warning: generated sample '{}' not found in split. Skipping.".format(
                    sample_id
                )
            )

        if not s_list[k]:
            del s_list[k]
            if not s_list:
                break

    for w in writers.values():
        w.close()
