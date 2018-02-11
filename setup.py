#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = [
    "Click>=6.0",
    "h5py",
    "numpy",
    "ruamel.yaml",
    "pyquaternion",
]

setup_requirements = ["pytest-runner", "tqdm"]

test_requirements = [
    "pytest",
    # TODO: Put package test requirements here
]

extra_requirements = {}

setup(
    name="tfrecord_converter",
    version="0.1.0",
    description="Convert data to TF Record example protobufs.",
    long_description=readme,
    author="Christoph Rist",
    author_email="c.rist@posteo.de",
    url="https://github.com/risteon/tfrecord_converter",
    packages=find_packages(include=["tfrecord_converter"]),
    entry_points={
        "console_scripts": [
            "tfrecord_process_hdf5=tfrecord_converter.cli:process_hdf5",
            "tfrecord_process_kitti_raw=tfrecord_converter.cli:process_kitti_raw",
            "tfrecord_process_kitti_accumulated="
            "tfrecord_converter.cli:process_kitti_accumulated",
            "tfrecord_process_kitti_semantics="
            "tfrecord_converter.cli:process_kitti_semantics",
            "tfrecord_process_nuscenes=tfrecord_converter.cli:process_nuscenes",
            "tfrecord_create_objects_from_hdf5="
            "tfrecord_converter.cli:create_objects_from_hdf5",
            "tfrecord_create_objects_from_nuscenes=tfrecord_converter.cli:"
            "create_objects_from_nuscenes",
            "tfrecord_process_kitti_lidar_semantics=tfrecord_converter.cli:"
            "process_kitti_lidar_semantics",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords="tfrecord_converter",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite="tests",
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    extras_require=extra_requirements,
)
