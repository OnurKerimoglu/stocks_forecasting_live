import os

import tomli


def get_pipreqs_from_pyproject(path: str) -> list[str]:
    """
    Parses your dependencies from pyproject.toml and returns them as a list
    of pip-style requirement strings.
    """
    with open(path, "rb") as f:
        pyproject = tomli.load(f)

    deps = pyproject.get("project", {}).get("dependencies", [])

    return deps


if __name__ == "__main__":
    rootpath = os.path.dirname(os.path.dirname(__file__))
    pipreqs = get_pipreqs_from_pyproject(os.path.join(rootpath, "pyproject.toml"))
    print(pipreqs)
