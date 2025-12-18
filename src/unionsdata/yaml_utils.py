from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Safely load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    with open(path, encoding='utf-8') as f:
        content = yaml.safe_load(f)
    return content if content is not None else {}


def parse_yaml(content: str) -> dict[str, Any]:
    """
    Safely parse YAML from a string.

    Args:
        content: YAML content as string

    Returns:
        Parsed YAML content as dictionary

    Raises:
        yaml.YAMLError: If content is not valid YAML
    """
    result = yaml.safe_load(content)
    return result if result is not None else {}


def save_yaml(path: Path, data: dict[str, Any], sort_keys: bool = False) -> None:
    """
    Safely save data to a YAML file.

    Args:
        path: Path to save to
        data: Dictionary to save
        sort_keys: Whether to sort dictionary keys
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=sort_keys, default_flow_style=False)


def yaml_to_string(data: dict[str, Any], sort_keys: bool = False) -> str:
    """Convert dictionary to YAML string."""
    return yaml.safe_dump(data, sort_keys=sort_keys, default_flow_style=False)
