import os

from src.create_experiments import (
    build_exp_dicts,
    generate_factorial_combinations,
    load_factor_levels_from_yaml,
)

ROOTPATH = os.path.dirname(os.path.dirname(__file__))
TMP_PATH = os.path.join(ROOTPATH, "tmp")
os.makedirs(TMP_PATH, exist_ok=True)

example_yaml_content = """
factor_levels:
  factor1: [a, b]
  factor2: [x, y, z]
"""


def test_build_exp_dicts() -> None:
    # Create a temporary YAML file with factor levels
    yaml_file = os.path.join(TMP_PATH, "example.yaml")
    with open(yaml_file, "w") as f:
        f.write(example_yaml_content)

    # Call the build_exp_dicts function
    exp_cfgs, exp_name = build_exp_dicts(str(yaml_file))

    # remove the file
    os.remove(yaml_file)

    # Check the expected output
    expected_exp_cfgs = [
        {"factor1": "a", "factor2": "x"},
        {"factor1": "a", "factor2": "y"},
        {"factor1": "a", "factor2": "z"},
        {"factor1": "b", "factor2": "x"},
        {"factor1": "b", "factor2": "y"},
        {"factor1": "b", "factor2": "z"},
    ]
    expected_exp_name = "factor1_factor2"

    assert exp_cfgs == expected_exp_cfgs
    assert exp_name == expected_exp_name


def test_load_factor_levels_from_yaml() -> None:
    # Create a temporary YAML file with factor levels
    yaml_file = os.path.join(TMP_PATH, "example.yaml")
    with open(yaml_file, "w") as f:
        f.write(example_yaml_content)

    # Call the load_factor_levels_from_yaml function
    factor_levels = load_factor_levels_from_yaml(str(yaml_file))

    # remove the file
    os.remove(yaml_file)

    # Check the expected output
    expected_factor_levels = {"factor1": ["a", "b"], "factor2": ["x", "y", "z"]}

    assert factor_levels == expected_factor_levels


def test_generate_factorial_combinations() -> None:
    factor_levels = {
        "factor1": ["a", "b"],
        "factor2": ["x", "y", "z"],
    }
    expected_combinations = [
        {"factor1": "a", "factor2": "x"},
        {"factor1": "a", "factor2": "y"},
        {"factor1": "a", "factor2": "z"},
        {"factor1": "b", "factor2": "x"},
        {"factor1": "b", "factor2": "y"},
        {"factor1": "b", "factor2": "z"},
    ]
    combinations = generate_factorial_combinations(factor_levels)
    assert combinations == expected_combinations
