#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import collections
import pathlib
import random

from .files_io import write_data_as_yaml, read_split


def get_hdf5_files(dataset):
    def get_hdf5_files_in_directory(directory):
        return sorted(
            [
                os.path.join(directory, file)
                for file in os.listdir(directory)
                if file.endswith(".hdf5")
            ]
        )

    def get_hdf5_files_from_list(filelist):
        with open(filelist, "r") as f:
            files = sorted(
                [l_.strip() for l_ in f.readlines() if l_.strip().endswith(".hdf5")]
            )
        return files

    parts = dataset.split(":", 1)
    if len(parts) == 1:
        desc, filepath = None, parts[0]
    else:
        desc, filepath = parts[0], parts[1]

    filepath = os.path.expanduser(filepath)

    if os.path.isfile(filepath):
        return get_hdf5_files_from_list(filepath), desc
    elif os.path.isdir(filepath):
        return get_hdf5_files_in_directory(filepath), desc
    else:
        raise RuntimeError("Could not retrieve input files from {}".format(filepath))


def get_files_recursively(root_dir, ending="hdf5"):
    f_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            f_path = os.path.join(root, filename)
            if not os.path.isfile(f_path) or not f_path.endswith("." + ending):
                continue
            f_list.append(f_path)
    return f_list


def get_files_flat(directory: pathlib.Path, ending: str):
    return [x for x in directory.iterdir() if x.suffix == ending]


def get_data_files_from_user_argument(files):
    """Check for each entry in iterable files
    if it is a .hdf5 file or a list of .hdf5 files

    """
    import h5py

    if not isinstance(files, collections.Iterable) or isinstance(files, str):
        files = tuple(files)

    f_list = []

    for f in files:
        if os.path.isfile(f):
            try:
                # try opening to check if it is a valid hdf5 file
                with h5py.File(f, "r"):
                    pass
                f_list.append(f)
            except OSError:
                with open(f, "r") as filelist:
                    content = filelist.readlines()
                    content = [x.strip() for x in content]
                    f_list.extend(content)
        elif os.path.isdir(f):
            f_list.extend(get_files_recursively(f))
        else:
            print("Warning: ignoring argument {}".format(f))

    return f_list


def is_empty(p: pathlib.Path):
    try:
        p.iterdir().__iter__().__next__()
        return False
    except StopIteration:
        return True


def query_user_yes_no(question, default="yes") -> bool:
    """ StackOverflow https://stackoverflow.com/a/3041990

    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def make_output_directory(output: pathlib.Path, overwrite: bool = False):
    output = output.absolute()
    # if output dir exists, abort. Otherwise create it
    if output.exists():
        if not output.is_dir():
            raise RuntimeError(
                "Output path '{}' already exists and is not a directory.".format(
                    str(output)
                )
            )
        if not is_empty(output):

            msg = "Output directory {} already exists and is not empty.".format(
                str(output)
            )

            if overwrite:
                print(msg)
                # continue_to_delete = query_user_yes_no(
                #     "Continue to write into '{}'?".format(str(output)), default="no"
                # )
                continue_to_delete = True
                if not continue_to_delete:
                    raise RuntimeError("User abort.")

            else:
                raise RuntimeError(msg)

    output.mkdir(exist_ok=True, parents=True)


def process_split(split_file: str, output_path: pathlib.Path, shuffle: bool = False):
    split_dict = read_split(split_file)
    if shuffle:
        [random.shuffle(l_) for l_ in split_dict["data"].values()]

    # write split for reference
    split_file_name = str(
        output_path / "dataset_split_{}.yaml".format(split_dict["name"])
    )
    write_data_as_yaml(split_dict, split_file_name)
    return split_dict
