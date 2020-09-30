import argparse
import importlib
import os
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Union


def timestamp_to_date_time(d) -> datetime:
    return datetime.fromtimestamp(float(d), tz=timezone.utc)


def get_property_recursively(properties: dict, *args, **kwargs) -> any:
    for arg in args:
        if arg in properties:
            properties = properties[arg]
        else:
            if 'default' not in kwargs:
                raise ValueError(f'Required property with name "{str(arg)}" is not set and no default value is provided.')
            else:
                return kwargs['default']

    return properties


def str2bool(value: str) -> bool:
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Unexpected string "{value}" that cannot be converted to a boolean.')


def instance_from_config(config: dict) -> any:
    module_name, class_name = config['class'].rsplit('.', 1)
    class_config = config['config'] if 'config' in config else {}

    return getattr(importlib.import_module(module_name), class_name)(class_config)


def function_from_string(function_string: str) -> Callable:
    module_name, function_name = function_string.rsplit('.', 1)

    return getattr(importlib.import_module(module_name), function_name)


def to_absolute_path(relative_path: Union[str, Path]) -> str:
    if os.path.isabs(relative_path):
        return relative_path
    else:
        relative_base_path = os.path.dirname(os.path.realpath(__file__))
        absolute_path = os.path.join(relative_base_path, '..', relative_path)

        return os.path.abspath(absolute_path)


def get_cli_arguments() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_path', help='Path to a config file or folder containing multiple config files.', type=str, required=True)
    parser.add_argument('-f', '--format', help='Format of the config file.', type=str, default='yaml')

    return parser.parse_args()


def out_path(file_name='') -> str:
    args = get_cli_arguments()
    config_file_name = os.path.basename(args.config_path)
    out_directory = to_absolute_path(os.path.join('out', config_file_name))
    out_path = os.path.join(out_directory, file_name)

    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    return out_path
