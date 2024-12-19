# Utilities for the library

import yaml

def parse_config(yml):
    with open(yml) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    return config
