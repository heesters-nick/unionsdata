from __future__ import annotations

import logging
import os
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


# ========== Schema Models ==========


class LoggingCfg(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra='forbid')
    name: str
    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'


class RuntimeCfg(BaseModel):
    """Runtime configuration."""

    model_config = ConfigDict(extra='forbid')
    n_download_threads: int = Field(ge=1, le=32)
    n_cutout_processes: int = Field(ge=1, le=32)
    bands: list[str] = Field(..., min_length=1, max_length=6)
    resume: bool = False
    max_retries: int = Field(ge=1, le=10, default=3)


class TilesCfg(BaseModel):
    """Tile management configuration."""

    model_config = ConfigDict(extra='forbid')
    update_tiles: bool = False
    show_tile_statistics: bool = True
    band_constraint: int = Field(ge=1, le=6)
    require_all_specified_bands: bool = False


class CutoutsCfg(BaseModel):
    """Cutouts configuration."""

    model_config = ConfigDict(extra='forbid')
    enable: bool = True
    cutouts_only: bool = False
    size_pix: int = Field(ge=1)
    output_subdir: str


class ColumnMap(BaseModel):
    """Column name mappings for dataframes."""

    model_config = ConfigDict(extra='forbid')
    ra: str = 'ra'
    dec: str = 'dec'
    id: str = 'ID'


class InputsDataFrame(BaseModel):
    """Dataframe input configuration."""

    model_config = ConfigDict(extra='forbid')
    path: Path = Path('')
    columns: ColumnMap = Field(default_factory=ColumnMap)


class InputsCfg(BaseModel):
    """Input source configuration."""

    model_config = ConfigDict(extra='forbid')
    source: Literal['all_available', 'tiles', 'coordinates', 'dataframe'] = 'tiles'
    tiles: list[tuple[int, int]] = Field(default_factory=list)
    coordinates: list[tuple[float, float]] = Field(default_factory=list)
    dataframe: InputsDataFrame = Field(default_factory=InputsDataFrame)


class PathsDatabase(BaseModel):
    """Common path directory names."""

    model_config = ConfigDict(extra='forbid')
    tile_info_dirname: str = 'tile_info'
    logs_dirname: str = 'logs'


class PathsByMachineEntry(BaseModel):
    """Machine-specific paths."""

    model_config = ConfigDict(extra='forbid')
    root_dir_main: Path
    root_dir_data: Path
    dir_tables: Path
    cert_path: Path


class BandCfg(BaseModel):
    """Band-specific configuration."""

    model_config = ConfigDict(extra='forbid')
    name: str
    band: str
    vos: str
    suffix: str
    delimiter: str
    fits_ext: int
    zfill: int
    zp: float


class BandDict(TypedDict):
    """Dictionary representation of band configuration (for legacy compatibility)."""

    name: str
    band: str
    vos: str
    suffix: str
    delimiter: str
    fits_ext: int
    zfill: int
    zp: float


class RawConfig(BaseModel):
    """Raw configuration loaded from YAML file."""

    model_config = ConfigDict(extra='forbid')
    machine: str
    logging: LoggingCfg
    runtime: RuntimeCfg
    tiles: TilesCfg
    cutouts: CutoutsCfg
    inputs: InputsCfg
    paths_database: PathsDatabase
    paths_by_machine: dict[str, PathsByMachineEntry]
    bands: dict[str, BandCfg]

    @model_validator(mode='after')
    def validate_machine(self) -> RawConfig:
        """Validate that machine exists in paths_by_machine."""
        if self.machine not in self.paths_by_machine:
            available = ', '.join(self.paths_by_machine.keys())
            raise ValueError(
                f'Machine "{self.machine}" not found in paths_by_machine. '
                f'Available machines: {available}'
            )
        return self


class PathsResolved(BaseModel):
    """Resolved paths after machine selection."""

    model_config = ConfigDict(extra='forbid')
    root_dir_main: Path
    root_dir_data: Path
    dir_tables: Path
    cert_path: Path
    tile_info_directory: Path
    log_directory: Path


class Settings(BaseModel):
    """Final validated and resolved settings."""

    model_config = ConfigDict(extra='forbid')
    machine: str
    logging: LoggingCfg
    runtime: RuntimeCfg
    tiles: TilesCfg
    cutouts: CutoutsCfg
    inputs: InputsCfg
    bands: dict[str, BandCfg]
    paths: PathsResolved
    config_source: Path | None = None  # Path to the config file used

    @model_validator(mode='after')
    def _validate(self) -> Settings:
        """Validate cross-field dependencies."""
        # Validate that all bands in runtime.bands exist in bands dict
        for band in self.runtime.bands:
            if band not in self.bands:
                raise ValueError(f'Unknown band in runtime.bands: {band}')

        # Validate band_constraint is reasonable
        if self.tiles.band_constraint > len(self.bands):
            raise ValueError(
                f'tiles.band_constraint ({self.tiles.band_constraint}) cannot be '
                f'greater than number of available bands ({len(self.bands)})'
            )

        # Validate input source specific requirements
        if self.inputs.source == 'dataframe':
            if not self.inputs.dataframe.path:
                raise ValueError('inputs.dataframe.path must be set when source is "dataframe"')

        if self.inputs.source == 'tiles':
            if not self.inputs.tiles:
                logger.warning('inputs.source is "tiles" but no tiles specified')

        if self.inputs.source == 'coordinates':
            if not self.inputs.coordinates:
                logger.warning('inputs.source is "coordinates" but no coordinates specified')

        if self.inputs.source not in ['coordinates', 'dataframe'] and self.cutouts.enable:
            logger.warning(
                'Cutout creation is enabled but no object coordinates were supplied. No cutouts will be created.'
            )

        if not self.paths.cert_path.exists():
            raise ValueError(f'Certificate file does not exist: {self.paths.cert_path}')

        return self


# ========== Loader ==========


def load_settings(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    check_first_run: bool = True,
) -> Settings:
    """
    Load and validate configuration from YAML file.

    Searches for config in expected location if path not provided.

    Args:
        config_path: Path to YAML configuration file (optional)
        cli_overrides: Dictionary of CLI overrides to apply (optional)
        check_first_run: Whether to check for first run and enable tile updates (optional)

    Returns:
        Validated Settings object

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist in any search location
    """
    # Check if we're in editable/dev mode
    is_editable = determine_install_mode()
    # Get config path based on install mode
    config_path = get_config_path(is_editable, config_path)
    logger.info(f'Loading configuration from: {config_path}')

    with open(config_path, encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    # Apply CLI overrides to raw YAML data before validation
    if cli_overrides:
        yaml_data = _apply_cli_overrides(yaml_data, cli_overrides)

    # Validate raw config
    raw = RawConfig.model_validate(yaml_data)

    # Resolve paths
    pm = raw.paths_by_machine[raw.machine]
    pc = raw.paths_database

    root = pm.root_dir_main

    if is_editable:
        # Use config-specified paths (development/custom setup)
        paths = PathsResolved(
            root_dir_main=root,
            root_dir_data=pm.root_dir_data,
            dir_tables=pm.dir_tables,
            cert_path=pm.cert_path,
            tile_info_directory=root / pc.tile_info_dirname,
            log_directory=root / pc.logs_dirname,
        )
    else:
        # Use XDG data directory
        data_base = get_user_data_dir()
        paths = PathsResolved(
            root_dir_main=data_base,
            root_dir_data=pm.root_dir_data,
            dir_tables=pm.dir_tables,
            cert_path=pm.cert_path,
            tile_info_directory=data_base / pc.tile_info_dirname,
            log_directory=data_base / pc.logs_dirname,
        )

    # Check for first run and auto-enable update_tiles if needed
    if check_first_run and is_first_run(paths.tile_info_directory):
        logger.info('=' * 70)
        logger.info('FIRST RUN DETECTED')
        logger.info('Tile information not found. Will download from CANFAR vault.')
        logger.info('This one-time setup takes approximately 5 minutes.')
        logger.info('=' * 70)

        # Override the setting to force update
        raw.tiles.update_tiles = True
        logger.debug('Automatically enabled tiles.update_tiles for first run.')

    # Build final settings
    settings = Settings(
        machine=raw.machine,
        logging=raw.logging,
        runtime=raw.runtime,
        tiles=raw.tiles,
        cutouts=raw.cutouts,
        inputs=raw.inputs,
        bands=raw.bands,
        paths=paths,
        config_source=config_path,
    )

    logger.info(f'Configuration loaded successfully for machine: {raw.machine}')

    return settings


def _apply_cli_overrides(yaml_data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Apply CLI overrides to YAML data.

    Args:
        yaml_data: Original YAML data
        overrides: Dictionary of overrides from CLI

    Returns:
        Modified YAML data
    """
    # Runtime overrides
    if 'bands' in overrides and overrides['bands'] is not None:
        yaml_data['runtime']['bands'] = overrides['bands']
        logger.debug(f'CLI override: runtime.bands = {overrides["bands"]}')

    # Tile overrides
    if 'update_tiles' in overrides and overrides['update_tiles'] is not None:
        yaml_data['tiles']['update_tiles'] = overrides['update_tiles']
        logger.debug(f'CLI override: tiles.update_tiles = {overrides["update_tiles"]}')

    # Input overrides
    if 'input_source' in overrides and overrides['input_source'] is not None:
        yaml_data['inputs']['source'] = overrides['input_source']
        logger.debug(f'CLI override: inputs.source = {overrides["input_source"]}')

    if 'tiles' in overrides and overrides['tiles'] is not None:
        yaml_data['inputs']['tiles'] = overrides['tiles']
        yaml_data['inputs']['source'] = 'tiles'
        logger.debug(f'CLI override: inputs.tiles = {overrides["tiles"]}, source = tiles')

    if 'coordinates' in overrides and overrides['coordinates'] is not None:
        yaml_data['inputs']['coordinates'] = overrides['coordinates']
        yaml_data['inputs']['source'] = 'coordinates'
        logger.debug(
            f'CLI override: inputs.coordinates = {overrides["coordinates"]}, source = coordinates'
        )

    if 'dataframe' in overrides and overrides['dataframe'] is not None:
        yaml_data['inputs']['dataframe']['path'] = overrides['dataframe']
        yaml_data['inputs']['source'] = 'dataframe'
        logger.debug(
            f'CLI override: inputs.dataframe.path = {overrides["dataframe"]}, source = dataframe'
        )

    if 'all_tiles' in overrides and overrides['all_tiles']:
        yaml_data['inputs']['source'] = 'all_available'
        logger.debug('CLI override: inputs.source = all_available')

    if 'cutouts' in overrides and overrides['cutouts'] is not None:
        yaml_data['cutouts']['enable'] = overrides['cutouts']
        logger.debug(f'CLI override: cutouts.enable = {overrides["cutouts"]}')

    return yaml_data


def ensure_runtime_dirs(cfg: Settings) -> None:
    """
    Ensure that all runtime directories exist.

    Args:
        cfg: Settings object with resolved paths
    """
    directories = [
        cfg.paths.tile_info_directory,
        cfg.paths.log_directory,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Ensured directory exists: {directory}')


def settings_to_dict(cfg: Settings) -> dict[str, Any]:
    """
    Convert Settings to a JSON-serializable dictionary.

    Args:
        cfg: Settings object

    Returns:
        Dictionary with all paths converted to strings
    """
    return cfg.model_dump(mode='json')


def get_band_config(cfg: Settings, band: str) -> BandCfg:
    """
    Get configuration for a specific band.

    Args:
        cfg: Settings object
        band: Band name

    Returns:
        Band configuration

    Raises:
        KeyError: If band not found
    """
    if band not in cfg.bands:
        raise KeyError(f'Band "{band}" not found. Available bands: {list(cfg.bands.keys())}')
    return cfg.bands[band]


def get_bands_subset(cfg: Settings, band_list: list[str]) -> dict[str, BandCfg]:
    """
    Get configurations for a subset of bands.

    Args:
        cfg: Settings object
        band_list: List of band names

    Returns:
        Dictionary with band configurations
    """
    return {band: get_band_config(cfg, band) for band in band_list}


def purge_previous_run(cfg: Settings) -> None:
    """If resume == False, delete existing log files."""
    if cfg.runtime.resume:
        return

    # Logs: remove current + rotated files for this logger name
    log_dir: Path = cfg.paths.log_directory
    stem = cfg.logging.name
    for p in log_dir.glob(f'{stem}.log*'):
        p.unlink(missing_ok=True)
        logger.debug(f'Removed old log file: {p}')


def get_user_config_dir() -> Path:
    """
    Get the user configuration directory following OS conventions.

    Returns user config directory, but does NOT create it.
    Creation happens only when actually needed.

    Linux/Mac: ~/.config/unionsdata/
    Windows: %APPDATA%/unionsdata/
    """
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:
        # XDG Base Directory specification
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))

    return base / 'unionsdata'


def get_user_data_dir() -> Path:
    """
    Get the user data directory following OS conventions.

    This is where tile_info, kdtree, etc. would be stored for pip installs.
    Returns directory path, but does NOT create it.

    Linux/Mac: ~/.local/share/unionsdata/
    Windows: %LOCALAPPDATA%/unionsdata/
    """
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        # XDG Base Directory specification
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))

    return base / 'unionsdata'


def is_first_run(tile_info_dir: Path) -> bool:
    """
    Check if this appears to be a first run (missing tile info files).

    Args:
        tile_info_dir: Directory where tile info should be stored

    Returns:
        True if tile info files are missing, False otherwise
    """

    logger.debug(f'Checking for first run in tile info directory: {tile_info_dir}')

    if not tile_info_dir.exists():
        return True

    # Check for any band tile files
    tile_files = list(tile_info_dir.glob('*_tiles.txt'))
    kdtree_file = tile_info_dir / 'kdtree_xyz.joblib'

    if not tile_files or not kdtree_file.exists():
        return True

    return False


def determine_install_mode() -> bool:
    """Determine if the package is installed in editable/development mode or from PyPI via pip install unionsdata."""
    try:
        dist = distribution('unionsdata')
        # Editable installs have .egg-link or direct_url.json
        is_editable = (
            any(f.name in ('direct_url.json', 'top_level.txt') for f in dist.files or [])
            if dist.files
            else False
        )
    except Exception:
        is_editable = False

    return is_editable


def get_config_path(is_editable: bool, config_path: Path | None = None) -> Path:
    """
    Get the configuration file path.

    Args:
        is_editable: Whether the package is in editable/development mode
        config_path: User-provided config path (optional)

    Returns:
        Path to config file
    Raises:
        FileNotFoundError: If specified config file does not exist.
    """
    if config_path is not None:
        if config_path.exists():
            return config_path
        else:
            raise FileNotFoundError(f'Specified config file not found: {config_path}')

    if is_editable:
        # Use config from source directory
        config_path = Path(str(files('unionsdata').joinpath('config.yaml')))
        if not config_path.exists():
            raise FileNotFoundError(
                f'We are in editable mode. Config file not found at: {config_path}'
            )
    else:
        # Use user config directory
        config_path = get_user_config_dir() / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(
                f'We are in user mode. Config file not found in user config directory at: {config_path}'
            )

    return config_path
