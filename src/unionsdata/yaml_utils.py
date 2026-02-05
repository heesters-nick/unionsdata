from pathlib import Path
from typing import Any

import yaml


def _normalize_paths(data: Any) -> Any:
    """
    Recursively normalize path strings in a data structure.

    Converts backslashes to forward slashes for cross-platform YAML compatibility.
    Python handles forward slashes on all platforms, and this prevents YAML
    escape sequence issues with Windows paths.

    Args:
        data: Any data structure (dict, list, str, etc.)

    Returns:
        Data structure with normalized path strings
    """
    if isinstance(data, dict):
        return {k: _normalize_paths(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_normalize_paths(item) for item in data]
    elif isinstance(data, str):
        # Heuristic: if it looks like a Windows path, normalize it
        # Matches patterns like "C:\..." or contains backslashes with path-like content
        if '\\' in data and (
            (len(data) > 2 and data[1] == ':')  # Drive letter: C:\...
            or data.startswith('\\\\')  # UNC path: \\server\...
            or any(segment in data for segment in ['\\Users\\', '\\home\\', '\\.ssl\\'])
        ):
            return data.replace('\\', '/')
        return data
    else:
        return data


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

    # Normalize paths for cross-platform compatibility
    normalized_data = _normalize_paths(data)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(normalized_data, f, sort_keys=sort_keys, default_flow_style=False)


def yaml_to_string(data: dict[str, Any], sort_keys: bool = False) -> str:
    """Convert dictionary to YAML string."""
    # Normalize paths here too for consistency
    normalized_data = _normalize_paths(data)
    return yaml.safe_dump(normalized_data, sort_keys=sort_keys, default_flow_style=False)
