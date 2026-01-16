from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, TypedDict

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unionsdata.yaml_utils import load_yaml

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
    bands: list[str] | None = None
    resume: bool = False
    max_retries: int = Field(ge=1, le=10, default=3)


class TilesCfg(BaseModel):
    """Tile management configuration."""

    model_config = ConfigDict(extra='forbid')
    update_tiles: bool = False
    show_tile_statistics: bool = True
    band_constraint: int = Field(ge=1, le=7)
    require_all_specified_bands: bool = False


class CutoutsCfg(BaseModel):
    """Cutouts configuration."""

    model_config = ConfigDict(extra='forbid')
    mode: Literal['disabled', 'after_download', 'direct_only'] = 'after_download'
    size_pix: int = Field(ge=1)
    output_subdir: str


class RGBCfg(BaseModel):
    """RGB scaling configuration."""

    model_config = ConfigDict(extra='forbid')
    scaling_type: Literal['asinh', 'linear'] = 'asinh'
    stretch: float = 125.0
    Q: float = 7.0
    gamma: float = 0.25
    standard_zp: float = 30.0


class MonoCfg(BaseModel):
    """Monochromatic scaling configuration"""

    model_config = ConfigDict(extra='forbid')
    scaling_type: Literal['asinh', 'linear'] = 'asinh'
    stretch: float = 125.0
    Q: float = 7.0
    gamma: float = 0.25


class PlottingCfg(BaseModel):
    """Plotting configuration."""

    model_config = ConfigDict(extra='forbid')
    enable: bool = False
    catalog_name: str = 'auto'
    bands: list[str] | None = None
    size_pix: int = Field(ge=1)
    mode: Literal['grid', 'channel'] = 'grid'
    max_cols: int = Field(ge=1, default=5)
    figsize: tuple[int, int] | None = None
    save_plot: bool = True
    show_plot: bool = False
    save_format: Literal['pdf', 'png', 'jpg', 'svg'] = 'pdf'
    rgb: RGBCfg = Field(default_factory=RGBCfg)
    mono: MonoCfg = Field(default_factory=MonoCfg)


class ColumnMap(BaseModel):
    """Column name mappings for tables."""

    model_config = ConfigDict(extra='forbid')
    ra: str = 'ra'
    dec: str = 'dec'
    id: str = 'ID'


class InputsTable(BaseModel):
    """Table input configuration."""

    model_config = ConfigDict(extra='forbid')
    path: Path = Path('')
    columns: ColumnMap = Field(default_factory=ColumnMap)


class InputsCfg(BaseModel):
    """Input source configuration."""

    model_config = ConfigDict(extra='forbid')
    source: Literal['all_available', 'tiles', 'coordinates', 'table'] = 'tiles'
    tiles: list[tuple[int, int]] = Field(default_factory=list)
    coordinates: list[tuple[float, float]] = Field(default_factory=list)
    table: InputsTable = Field(default_factory=InputsTable)


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
    dir_figures: Path
    cert_path: Path

    @field_validator('root_dir_data', 'dir_tables', 'dir_figures', 'cert_path')
    @classmethod
    def validate_non_empty_path(cls, v: Path) -> Path:
        # Pydantic converts empty strings to Path('.'), so we must check for that
        if str(v) == '.' or not str(v).strip():
            raise ValueError('Path cannot be empty')
        return v


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
    plotting: PlottingCfg
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
    dir_figures: Path
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
    plotting: PlottingCfg
    inputs: InputsCfg
    bands: dict[str, BandCfg]
    paths: PathsResolved
    config_source: Path | None = None  # Path to the config file used

    @model_validator(mode='after')
    def _validate(self) -> Settings:
        """Validate cross-field dependencies."""
        # Validate that all bands in runtime.bands exist in bands dict
        if self.runtime.bands is None or len(self.runtime.bands) == 0:
            raise ValueError('runtime.bands must be set to a list of band names')
        for band in self.runtime.bands:
            if band not in self.bands:
                raise ValueError(f'Unknown band in runtime.bands: {band}')
        # validate that plotting.bands is not None if plotting.enable is True
        if self.plotting.enable:
            if self.plotting.bands is None or len(self.plotting.bands) == 0:
                raise ValueError('plotting.bands must be set when plotting.enable is True')
            for band in self.plotting.bands:
                if band not in self.bands:
                    raise ValueError(f'Unknown band in plotting.bands: {band}')

        # Validate band_constraint is reasonable
        if self.tiles.band_constraint > len(self.bands):
            raise ValueError(
                f'tiles.band_constraint ({self.tiles.band_constraint}) cannot be '
                f'greater than number of available bands ({len(self.bands)})'
            )

        # Validate input source specific requirements
        if self.inputs.source == 'table':
            if not self.inputs.table.path:
                raise ValueError('inputs.table.path must be set when source is "table"')

        if self.inputs.source == 'tiles':
            if not self.inputs.tiles:
                logger.warning('inputs.source is "tiles" but no tiles specified')

        if self.inputs.source == 'coordinates':
            if not self.inputs.coordinates:
                logger.warning('inputs.source is "coordinates" but no coordinates specified')

        if self.inputs.source not in ['coordinates', 'table'] and self.cutouts.mode != 'disabled':
            logger.warning(
                'Cutout creation is enabled but no object coordinates were supplied. No cutouts will be created.'
            )

        if not self.paths.cert_path.exists():
            raise ValueError(f'Certificate file does not exist: {self.paths.cert_path}')

        try:
            check_cert_expiry(self.paths.cert_path)
        except ValueError:
            sys.exit(1)

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

    yaml_data = load_yaml(config_path)

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
            dir_figures=pm.dir_figures,
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
            dir_figures=pm.dir_figures,
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
        plotting=raw.plotting,
        inputs=raw.inputs,
        bands=raw.bands,
        paths=paths,
        config_source=config_path,
    )

    logger.debug(f'Configuration loaded successfully for machine: {raw.machine}')

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

    if 'table' in overrides and overrides['table'] is not None:
        yaml_data['inputs']['table']['path'] = overrides['table']
        yaml_data['inputs']['source'] = 'table'
        logger.debug(f'CLI override: inputs.table.path = {overrides["table"]}, source = table')

    if 'all_tiles' in overrides and overrides['all_tiles']:
        yaml_data['inputs']['source'] = 'all_available'
        logger.debug('CLI override: inputs.source = all_available')

    if 'cutouts' in overrides and overrides['cutouts'] is not None:
        yaml_data['cutouts']['mode'] = 'after_download' if overrides['cutouts'] else 'disabled'
        logger.debug(f'CLI override: cutouts.mode = {yaml_data["cutouts"]["mode"]}')

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
        cfg.paths.dir_tables,
        cfg.paths.dir_figures,
        cfg.paths.root_dir_data,
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
    kdtree_file = tile_info_dir / 'kdtree_xyz.pkl'

    if not tile_files or not kdtree_file.exists():
        return True

    return False


def determine_install_mode() -> bool:
    """
    Determine if the package is installed in editable/development mode
    by checking PEP 660 metadata (direct_url.json).
    """
    try:
        # Get the distribution metadata for this package
        dist = distribution('unionsdata')

        # Look for the direct_url.json file which contains install details
        if dist.files:
            for f in dist.files:
                if f.name == 'direct_url.json':
                    # Parse the JSON to check the 'editable' flag
                    # We use json.loads() to avoid brittle string matching
                    data = json.loads(f.read_text())
                    return data.get('dir_info', {}).get('editable', False)

    except Exception:
        # If the package isn't installed or metadata is missing,
        # it's definitely not an editable install.
        pass

    return False


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


def check_cert_expiry(cert_path: Path, days_warning: int = 1) -> None:
    """Check certificate expiry and log warnings if close to expiry.
    Args:
        cert_path: Path to the PEM certificate file
        days_warning: Number of days before expiry to issue a warning

    Raises:
        ValueError: If the certificate is expired

    Returns:
        None
    """
    try:
        with open(cert_path, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
    except (ValueError, OSError) as e:
        # ValueError: Invalid PEM format
        # OSError: Permission denied / IO error
        logger.error(f'Invalid or inaccessible certificate at {cert_path}: {e}')
        raise ValueError('Certificate validation failed') from e

    # Simple UTC comparison to avoid timezone issues
    expiry = cert.not_valid_after_utc
    time_left = expiry - datetime.now(UTC)

    if time_left.total_seconds() < 0:
        logger.error(f'Certificate at {cert_path} EXPIRED on {expiry}.')
        raise ValueError('Certificate expired')

    if time_left < timedelta(days=days_warning):
        logger.warning(f'Certificate expires in {time_left.days} days ({expiry}).')
