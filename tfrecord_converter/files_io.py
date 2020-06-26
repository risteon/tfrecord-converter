#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ruamel.yaml import YAML


def read_split(split_file):
    with open(split_file, 'r') as y_f:
        yaml = YAML()
        d = yaml.load(y_f)
    return {
        'name': d['name'],
        'desc': d['desc'],
        'data': {k: list(d['data'][k]) for k in d['data']},
    }


def write_data_as_yaml(data, output_file):
    with open(str(output_file), 'w') as o:
        yaml = YAML()
        yaml.dump({k: str(v) for k, v in data.items()}, o)


def write_flags_to_file(flags, output_file):
    yaml = YAML()
    with open(output_file, 'w') as output_dataset_desc_file:
        yaml.dump(vars(flags), output_dataset_desc_file)
