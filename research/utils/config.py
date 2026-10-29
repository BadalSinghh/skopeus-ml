"""YAML Configuration loader and validator."""

from pathlib import Path
from typing import Dict, Any, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Path to .yaml file.

    Returns:
        Dictionary of parsed parameters.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save dictionary as YAML configuration file.

    Args:
        config: Dictionary of settings.
        output_path: Target path to save .yaml.

    Returns:
        Path of saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    return path
