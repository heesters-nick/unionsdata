from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

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
    bands: list[str]
    resume: bool = False


class TilesCfg(BaseModel):
    """Tile management configuration."""

    model_config = ConfigDict(extra='forbid')
    update_tiles: bool = False
    show_tile_statistics: bool = True
    band_constraint: int = Field(ge=1, le=6)


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


class PathsCommon(BaseModel):
    """Common path directory names."""

    model_config = ConfigDict(extra='forbid')
    table_dirname: str = 'tables'
    tile_info_dirname: str = 'tile_info'
    logs_dirname: str = 'logs'


class PathsByMachineEntry(BaseModel):
    """Machine-specific paths."""

    model_config = ConfigDict(extra='forbid')
    root_dir_main: Path
    root_dir_data: Path
    download_directory: Path


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


class RawConfig(BaseModel):
    """Raw configuration loaded from YAML file."""

    model_config = ConfigDict(extra='forbid')
    machine: str
    logging: LoggingCfg
    runtime: RuntimeCfg
    tiles: TilesCfg
    inputs: InputsCfg
    paths_common: PathsCommon
    paths_by_machine: dict[str, PathsByMachineEntry]
    bands: dict[str, BandCfg]


class PathsResolved(BaseModel):
    """Resolved paths after machine selection."""

    model_config = ConfigDict(extra='forbid')
    root_dir_main: Path
    root_dir_data: Path
    download_directory: Path
    table_directory: Path
    tile_info_directory: Path
    log_directory: Path


class Settings(BaseModel):
    """Final validated and resolved settings."""

    model_config = ConfigDict(extra='forbid')
    machine: str
    logging: LoggingCfg
    runtime: RuntimeCfg
    tiles: TilesCfg
    inputs: InputsCfg
    bands: dict[str, BandCfg]
    paths: PathsResolved

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

        return self


# ========== Loader ==========


def load_settings(
    config_path: Path = Path('configs/download_config.yaml'),
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        cli_overrides: Dictionary of CLI overrides to apply

    Returns:
        Validated Settings object

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        # Try searching in common locations
        search_paths = [
            Path('configs/download_config.yaml'),
            Path('download_config.yaml'),
            Path('../configs/download_config.yaml'),
        ]
        for search_path in search_paths:
            if search_path.exists():
                config_path = search_path
                break
        else:
            raise FileNotFoundError(
                f'Config file not found: {config_path}. Searched: {search_paths}'
            )

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
    pc = raw.paths_common

    root = pm.root_dir_main
    paths = PathsResolved(
        root_dir_main=root,
        root_dir_data=pm.root_dir_data,
        download_directory=pm.download_directory,
        table_directory=root / pc.table_dirname,
        tile_info_directory=root / pc.tile_info_dirname,
        log_directory=root / pc.logs_dirname,
    )

    # Build final settings
    settings = Settings(
        machine=raw.machine,
        logging=raw.logging,
        runtime=raw.runtime,
        tiles=raw.tiles,
        inputs=raw.inputs,
        bands=raw.bands,
        paths=paths,
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

    return yaml_data


def ensure_runtime_dirs(cfg: Settings) -> None:
    """
    Ensure that all runtime directories exist.

    Args:
        cfg: Settings object with resolved paths
    """
    directories = [
        cfg.paths.table_directory,
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


def purge_previous_run(cfg) -> None:
    """If resume == False, delete existing log files and the progress database."""
    if cfg.runtime.resume:
        return

    # Logs: remove current + rotated files for this logger name
    log_dir: Path = cfg.paths.log_directory
    stem = cfg.logging.name
    for p in log_dir.glob(f'{stem}.log*'):
        p.unlink(missing_ok=True)
        logger.debug(f'Removed old log file: {p}')
