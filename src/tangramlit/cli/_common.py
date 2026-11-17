"""Helpers shared by the CLI commands."""

import os
from typing import Sequence


def validate_input_files(input_dir: str, file_names: Sequence[str]) -> None:
    """Ensure input files exist in the given directory."""
    missing = []
    for name in file_names:
        path = os.path.join(input_dir, name)
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
