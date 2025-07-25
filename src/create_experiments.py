import itertools
import os

import yaml


def build_exp_dicts(fpath_exp_cfg: str) -> dict:
    factor_levels = load_factor_levels_from_yaml(fpath_exp_cfg)
    factor_keys = [key for key in factor_levels.keys()]
    exp_name = "_".join(factor_keys)
    # generate full-factorial combinations
    exp_cfgs = generate_factorial_combinations(factor_levels)
    return exp_cfgs, exp_name


def load_factor_levels_from_yaml(yaml_file: str) -> dict:
    """Load factor levels from a YAML file."""
    with open(yaml_file) as file:
        data = yaml.safe_load(file)
    factor_levels = data["factor_levels"]
    # Should be of form:
    # factor_levels = {"f1": [val1, val2], "f2": [val1, val2, val3]}
    return factor_levels


def generate_factorial_combinations(factor_levels: dict) -> list:
    keys = list(factor_levels.keys())
    values = list(factor_levels.values())
    combination_list = [
        dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)
    ]
    return combination_list


if __name__ == "__main__":
    rootpath = os.path.dirname(os.path.dirname(__file__))
    fpath = os.path.join(rootpath, "config", "Exp_CldrFeats_ModReg.yaml")
    configs = build_exp_dicts(fpath)
    for cfg in configs:
        print(cfg)
