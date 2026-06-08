"""
Central filesystem paths for the BioPred project.

This module resolves the project root and exposes repo-relative paths.
It should not contain db credentials, environment variables, SQL logic, model logic, or feature-engineering logic.

"""

from pathlib import Path

PROJECT_MARKERS = (".git", "pyproject.toml", "README.md")

def find_project_root(start : Path | None = None) -> Path:
    """
    Find the BioPred project root by walking upward from 'start'.

    Root is identified by the presence of any file in PROJECT_MARKERS.
    This avoides hardcoded paths and fragile notebook-relative paths.

    Params
    ---------
    start:
        Starting directory for the search. If omitted, uses the current working directory.
    
    Returns
    ---------
    Path:
        Absolute path to the project root directory.

    Raises
    ---------
    FileNotFoundError:
        If no project marker is found in 'start' or any parent directory.
    """

    current = (start or Path.cwd()).resolve()

    for path in (current, *current.parents):
        if any((path / marker).exists() for marker in PROJECT_MARKERS):
            return path

    raise FileNotFoundError(
        f"Could not find project root from {current}. "
        f"Looked for markers: {PROJECT_MARKERS}"
    )

PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"