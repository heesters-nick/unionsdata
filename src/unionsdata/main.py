import argparse
import importlib.metadata
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError
from rich.logging import RichHandler
from yaml import YAMLError

from unionsdata.config import (
    BandCfg,
    ensure_runtime_dirs,
    get_config_path,
    get_user_config_dir,
    load_settings,
    purge_previous_run,
    settings_to_dict,
)
from unionsdata.cutout_stream import stream_direct_cutouts
from unionsdata.cutouts import create_cutouts_for_existing_tiles
from unionsdata.download import download_tiles
from unionsdata.logging_setup import setup_logger
from unionsdata.plotting import (
    build_plot_filename,
    cutouts_to_rgb,
    find_most_recent_catalog,
    load_cutouts,
    plot_cutouts,
)
from unionsdata.stats import (
    RunStatistics,
    compute_cutout_availability,
    compute_cutout_skipped,
    compute_tile_availability_from_catalog,
    compute_tile_availability_from_tiles,
    report_summary,
)
from unionsdata.tui import run_config_editor
from unionsdata.utils import (
    TileAvailability,
    clean_process_columns,
    filter_for_processing,
    get_wavelength_order,
    input_to_tile_list,
    jobs_from_catalog,
    jobs_from_tiles,
    load_existing_catalog,
    merge_w_existing_catalog,
    process_download_results,
    query_availability,
    update_catalog,
)
from unionsdata.yaml_utils import yaml_to_string

logger = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version('unionsdata')
except importlib.metadata.PackageNotFoundError:
    # Development mode fallback
    try:
        import tomllib

        pyproject_path = Path(__file__).parent.parent.parent / 'pyproject.toml'
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
        __version__ = data['project']['version']
    except Exception:
        __version__ = '0.1.0-dev'


def build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Build dictionary of CLI overrides - only bands and input."""
    overrides = {}

    # Bands override
    if hasattr(args, 'bands') and args.bands:
        overrides['bands'] = args.bands

    if hasattr(args, 'update_tiles') and args.update_tiles:
        overrides['update_tiles'] = True

    # Input overrides (mutually exclusive)
    if hasattr(args, 'tiles') and args.tiles:
        if len(args.tiles) % 2 != 0:
            raise ValueError('Tiles must be provided as pairs')
        overrides['tiles'] = [
            [args.tiles[i], args.tiles[i + 1]] for i in range(0, len(args.tiles), 2)
        ]
    elif hasattr(args, 'coordinates') and args.coordinates:
        if len(args.coordinates) % 2 != 0:
            raise ValueError('Coordinates must be provided as pairs')
        overrides['coordinates'] = [
            [args.coordinates[i], args.coordinates[i + 1]]
            for i in range(0, len(args.coordinates), 2)
        ]
    elif hasattr(args, 'table') and args.table:
        overrides['table'] = args.table
    elif hasattr(args, 'all_tiles') and args.all_tiles:
        overrides['all_tiles'] = True

    if hasattr(args, 'cutouts') and args.cutouts:
        overrides['cutouts'] = True

    return overrides


def run_download(args: argparse.Namespace) -> None:
    """
    CLI entry point to download UNIONS survey image tiles.
    """
    start = time.time()

    # Parse arguments and load configuration
    overrides = build_cli_overrides(args)
    try:
        cfg = load_settings(config_path=args.config, cli_overrides=overrides)
        logger.info(f'Loaded config from: {cfg.config_source}')
    except FileNotFoundError:
        logger.error('No config file found! Run "unionsdata init".')
        sys.exit(1)
    except (ValidationError, ValueError) as e:
        logger.error(f'Configuration error:\n{e}')
        sys.exit(1)
    except YAMLError as e:
        logger.error(f'Corrupt config file (invalid YAML):\n{e}')
        sys.exit(1)
    # Get rid of any previous log files if resume = False
    purge_previous_run(cfg)

    # Define frequently used variables
    bands = cfg.runtime.bands or []
    tile_info_dir = cfg.paths.tile_info_directory
    download_dir = cfg.paths.root_dir_data
    download_threads = cfg.runtime.n_download_threads
    cert_path = cfg.paths.cert_path
    max_retries = cfg.runtime.max_retries
    require_all = cfg.tiles.require_all_specified_bands
    cutouts_enabled = cfg.cutouts.mode != 'disabled'
    object_input = cfg.inputs.source not in ['tiles', 'all_available']
    cutout_mode = cfg.cutouts.mode

    setup_logger(
        log_dir=cfg.paths.log_directory,
        name=cfg.logging.name,
        logging_level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        force=True,
    )

    logger.debug('Starting UNIONS data download...')

    cfg_dict = settings_to_dict(cfg)

    # Print settings in human readable format
    cfg_yaml = yaml_to_string(cfg_dict)
    logger.debug(f'Resolved config (YAML):\n{cfg_yaml}')

    all_band_dict: dict[str, BandCfg] = dict(cfg.bands)

    # sort bands by wavelength
    bands = get_wavelength_order(bands, all_band_dict)

    # filter considered bands from the full band dictionary
    selected_band_dict: dict[str, BandCfg] = {k: cfg.bands[k] for k in bands}
    # make sure necessary directories exist
    ensure_runtime_dirs(cfg=cfg)

    # Initialize RunStatistics
    run_stats = RunStatistics()
    run_stats.initialize_bands(bands)

    # Query availability of tiles
    logger.debug('Querying tile availability...')
    availability_all, all_tiles = query_availability(
        update=cfg.tiles.update_tiles,
        in_dict=all_band_dict,
        show_stats=cfg.tiles.show_tile_statistics,
        tile_info_dir=tile_info_dir,
    )

    all_unique_tiles = availability_all.unique_tiles

    # Filter tiles to only those in selected bands
    selected_tiles = [all_tiles[list(all_band_dict.keys()).index(band)] for band in bands]

    availability_sel = TileAvailability(selected_tiles, selected_band_dict)

    # Process input to get list of tiles to download
    logger.debug('Processing input to determine tiles to download...')
    # get the list of tiles
    _, tiles_x_bands, catalog = input_to_tile_list(
        avail_all=availability_all,
        avail_sel=availability_sel,
        all_unique_tiles=all_unique_tiles,
        band_constr=cfg.tiles.band_constraint,
        inputs=cfg.inputs,
        tile_info_dir=tile_info_dir,
    )

    catalog_to_process: pd.DataFrame | None = catalog.copy()
    catalog_path = Path()

    tile_success_map: dict[str, set[str]] = {}  # {'123_456': {'r', 'i'}}
    cutout_success_map: dict[str, set[str]] = {}  # {'objectID': {'r', 'i'}}

    # Only relevant for object-based input
    if object_input:
        catalog_existing, catalog_path = load_existing_catalog(
            inputs=cfg.inputs, tables_dir=cfg.paths.dir_tables
        )

        catalog = merge_w_existing_catalog(
            fresh_catalog=catalog,
            existing_catalog=catalog_existing,
        )

        # Compute availability statistics
        compute_tile_availability_from_catalog(catalog, bands, run_stats)
        if cutouts_enabled:
            compute_cutout_availability(catalog, bands, run_stats)
            compute_cutout_skipped(catalog, bands, run_stats)

        catalog_to_process = filter_for_processing(
            catalog=catalog,
            bands=bands,
            download_col='bands_downloaded',
            cutout_col='cutout_bands',
            require_all=require_all,
            cutouts_enabled=cutouts_enabled,
        )

        # Early exit if nothing to do
        if catalog_to_process is None:
            logger.info('Everything is already downloaded and cut out. Nothing to do.')
            # Cleanup: remove bands_to_process column if it exists
            for col_name in ['bands_to_download', 'bands_to_cutout']:
                if col_name in catalog.columns:
                    catalog.drop(columns=[col_name], inplace=True)
            # Save augmented catalog
            catalog.to_csv(catalog_path, index=False)

            # Still show summary for visibility
            report_summary(
                stats=run_stats,
                cutouts_enabled=cutouts_enabled,
                cutout_mode=cutout_mode,
                show_tiles=(cutout_mode != 'direct_only'),
            )

            sys.exit(0)

    else:
        # Tile-based input: compute tile availability
        if tiles_x_bands:
            compute_tile_availability_from_tiles(
                requested_tiles=tiles_x_bands,
                bands=bands,
                avail=availability_sel,
                stats=run_stats,
            )

    # Branch off for direct cutout streaming
    if cutout_mode == 'direct_only':
        if cfg.inputs.source not in ['coordinates', 'table']:
            logger.error('Direct cutout mode requires coordinates or table input')
            sys.exit(1)

        if catalog.empty:
            logger.error('No objects in catalog for direct cutout mode')
            sys.exit(1)

        assert catalog_to_process is not None

        logger.info(f'Processing {len(catalog_to_process)}/{len(catalog)} objects...')
        stream_direct_cutouts(
            catalog=catalog_to_process,
            bands=bands,
            band_dict=all_band_dict,
            download_dir=download_dir,
            cutout_subdir=cfg.cutouts.output_subdir,
            cutout_size=cfg.cutouts.size_pix,
            cert_path=cert_path,
            max_retries=max_retries,
            n_workers=download_threads,
            run_stats=run_stats,
            cutout_success_map=cutout_success_map,
        )

        # Save augmented catalog
        if not catalog.empty and cutout_success_map:
            catalog = update_catalog(
                catalog=catalog, successful_bands_map=cutout_success_map, band_dict=all_band_dict
            )

        # Cleanup: remove bands_to_process column if it exists
        catalog = clean_process_columns(catalog)

        if object_input:
            catalog.to_csv(catalog_path, index=False)
            logger.info(f'Saved augmented catalog to {catalog_path}')
    else:
        # Build list of download jobs
        if catalog_to_process is not None and not catalog_to_process.empty and object_input:
            needed_jobs = jobs_from_catalog(catalog_to_process)
        elif not object_input and tiles_x_bands:
            needed_jobs = jobs_from_tiles(
                tiles=tiles_x_bands, bands=bands, avail=availability_sel, require_all=require_all
            )
        else:
            logger.info('No tiles or objects to process.')
            report_summary(
                stats=run_stats,
                cutouts_enabled=cutouts_enabled,
                cutout_mode=cutout_mode,
                show_tiles=True,
            )
            return

        # Sort for deterministic execution
        download_jobs = sorted(needed_jobs)

        # Log download summary
        total_downloads = len(download_jobs)
        if total_downloads > 0:
            logger.info(f'Total download jobs: {total_downloads}')
            jobs_by_band: dict[str, int] = dict.fromkeys(bands, 0)
            for _, band in download_jobs:
                jobs_by_band[band] = jobs_by_band.get(band, 0) + 1
            for band, count in jobs_by_band.items():
                logger.info(f'  {band}: {count} tiles')

        if download_jobs:
            # Download the tiles
            download_threads = min(download_threads, len(download_jobs))
            logger.debug(f'Starting downloads using {download_threads} threads...')
            try:
                download_tiles(
                    tiles_to_download=download_jobs,
                    band_dictionary=selected_band_dict,
                    all_band_dictionary=all_band_dict,
                    download_dir=download_dir,
                    requested_bands=set(bands),
                    num_threads=download_threads,
                    num_cutout_workers=cfg.runtime.n_cutout_processes,
                    cert_path=cert_path,
                    max_retries=max_retries,
                    catalog=catalog,
                    cutouts=cfg.cutouts,
                    log_dir=cfg.paths.log_directory,
                    log_name=cfg.logging.name,
                    log_level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
                    run_stats=run_stats,
                    tile_success_map=tile_success_map,
                    cutout_success_map=cutout_success_map,
                )
            except Exception as e:
                logger.error(f'Error during download: {e}')
                raise
        else:
            logger.info('No tiles to download.')

        # Check if additional cutouts needed for existing tiles
        if cutouts_enabled and catalog_to_process is not None and object_input:
            # Check if any objects still need cutouts
            still_need_cutouts = catalog_to_process[
                catalog_to_process['bands_to_cutout'].map(len) > 0
            ]

            # Exclude objects that already got cutouts during download phase
            already_done = set(cutout_success_map.keys())
            still_need_cutouts = still_need_cutouts[
                ~still_need_cutouts['ID'].astype(str).isin(already_done)
            ]

            if not still_need_cutouts.empty:
                logger.info(
                    f'Creating cutouts for {len(still_need_cutouts)} objects on existing tiles...'
                )

                create_cutouts_for_existing_tiles(
                    catalog=still_need_cutouts,
                    bands=bands,
                    download_dir=download_dir,
                    all_band_dictionary=all_band_dict,
                    cutout_size=cfg.cutouts.size_pix,
                    cutout_subdir=cfg.cutouts.output_subdir,
                    num_workers=cfg.runtime.n_cutout_processes,
                    log_dir=cfg.paths.log_directory,
                    log_name=cfg.logging.name,
                    log_level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
                    run_stats=run_stats,
                    cutout_success_map=cutout_success_map,
                )

        if not catalog.empty:
            catalog = process_download_results(
                catalog=catalog,
                tile_progress=tile_success_map,
                cutout_success_map=cutout_success_map,
                cutouts_enabled=cutouts_enabled,
                band_dict=all_band_dict,
            )
            # Cleanup: remove bands_to_process column if it exists
            catalog = clean_process_columns(catalog)

            # Save augmented catalog
            catalog.to_csv(catalog_path, index=False)
            logger.info(f'Saved augmented catalog to {catalog_path}')

    # Report summary using new rich tables
    report_summary(
        stats=run_stats,
        cutouts_enabled=cutouts_enabled,
        cutout_mode=cutout_mode,
        show_tiles=(cutout_mode != 'direct_only'),
    )

    end = time.time()
    elapsed = end - start
    elapsed_string = str(timedelta(seconds=elapsed))
    hours, minutes, seconds = (
        np.float32(elapsed_string.split(':')[0]),
        np.float32(elapsed_string.split(':')[1]),
        np.float32(elapsed_string.split(':')[2]),
    )
    logger.info(
        f'Done! Execution took {hours} hours, {minutes} minutes, and {seconds:.2f} seconds.'
    )


def run_init(args: argparse.Namespace) -> None:
    """
    Initialize user configuration by creating a config.yaml file. If the file
    already exists, it will not be overwritten unless --force is specified.
    """

    user_config_dir = get_user_config_dir()
    user_config_path = user_config_dir / 'config.yaml'

    logger.info(f'User config directory: {user_config_dir}')

    # Create the directory if it doesn't exist
    try:
        user_config_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f'Failed to create config directory: {e}')
        sys.exit(1)

    # Check if config file already exists
    if user_config_path.exists() and not args.force:
        logger.warning(f'Config file already exists at: {user_config_path}')
        logger.info('Run with --force to overwrite.')
        return

    # Read the template file from the package
    try:
        template_path = files('unionsdata').joinpath('config.yaml')
        template_content = template_path.read_text(encoding='utf-8')

        with open(user_config_path, 'w', encoding='utf-8') as f:
            f.write(template_content)

        logger.info(f'Successfully created config file at: {user_config_path}')
        logger.info('You can now edit this file by running "unionsdata config"')

    except FileNotFoundError:
        logger.error('Could not find the config template within the package.')
        sys.exit(1)
    except Exception as e:
        logger.error(f'Failed to write config file: {e}')
        sys.exit(1)


def run_edit(args: argparse.Namespace) -> None:
    """
    Open the configuration file in the system's default editor. This is only relevant for pip installs.
    Always targets user config (unless --config is used) and validates after editing.
    """

    # Resolve config path. Force is_editable=False to target user config (pip install mode).
    try:
        user_config_path = get_config_path(is_editable=False, config_path=args.config)
    except FileNotFoundError:
        if args.config:
            logger.error(f'Config file not found at: {args.config}')
        else:
            logger.error('No config file found.')
            logger.info("Please run 'unionsdata init' first.")
        sys.exit(1)

    # Get the system's default editor
    editor = os.environ.get('EDITOR', 'vim' if sys.platform != 'win32' else 'notepad')

    while True:
        logger.info(f'Opening {user_config_path} with {editor}...')
        try:
            subprocess.run([editor, str(user_config_path)], check=True)
        except subprocess.CalledProcessError:
            logger.error('Editor exited with an error')
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f'Editor "{editor}" not found. Set EDITOR environment variable.')
            sys.exit(1)
        except Exception as e:
            logger.error(f'Failed to open editor: {e}')
            sys.exit(1)

        # Validate the config immediately after closing editor
        logger.info('Validating configuration...')
        try:
            # Validate the specific file we just edited by loading it
            _ = load_settings(config_path=user_config_path, check_first_run=False)
            logger.info('✓ Configuration is valid!')
            break

        except Exception as e:
            logger.error(f'✗ Config validation failed:\n{e}')

            try:
                response = (
                    input('\nWould you like to re-open the editor to fix this? [Y/n]: ')
                    .lower()
                    .strip()
                )
            except KeyboardInterrupt:
                print()
                sys.exit(1)

            if response not in ('', 'y', 'yes'):
                logger.warning('Exiting with invalid configuration.')
                sys.exit(1)


def run_config(args: argparse.Namespace) -> None:
    """
    Launch the interactive TUI configuration editor.

    This provides a user-friendly interface for editing the configuration file
    with dropdowns, checkboxes, and real-time validation.
    """
    config_path = args.config if hasattr(args, 'config') and args.config else None
    run_config_editor(config_path=config_path)


def run_plot(args: argparse.Namespace) -> None:
    """Plot RGB object cutouts from input table or coordinates.
    Uses configuration from config.yaml plotting section."""

    plot_start = time.time()
    # Parse arguments and load configuration
    overrides = build_cli_overrides(args)
    try:
        cfg = load_settings(config_path=args.config, cli_overrides=overrides)
        logger.info(f'Loaded config from: {cfg.config_source}')
    except FileNotFoundError:
        logger.error('No config file found! Run "unionsdata init".')
        sys.exit(1)
    except (ValidationError, ValueError) as e:
        logger.error(f'Configuration error:\n{e}')
        sys.exit(1)
    except YAMLError as e:
        logger.error(f'Corrupt config file (invalid YAML):\n{e}')
        sys.exit(1)
    # Get rid of any previous log files if resume = False
    purge_previous_run(cfg)

    setup_logger(
        log_dir=cfg.paths.log_directory,
        name=cfg.logging.name,
        logging_level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        force=True,
    )

    logger.info('Plotting object cutouts..')

    cfg_dict = settings_to_dict(cfg)

    # Print settings in human readable format
    cfg_yaml = yaml_to_string(cfg_dict)
    logger.debug(f'Resolved config (YAML):\n{cfg_yaml}')

    # make sure necessary directories exist
    ensure_runtime_dirs(cfg=cfg)

    figure_dir = cfg.paths.dir_figures
    table_dir = cfg.paths.dir_tables
    data_dir = cfg.paths.root_dir_data
    cutout_size = cfg.plotting.size_pix
    catalog_name = cfg.plotting.catalog_name
    plot_bands = cfg.plotting.bands or []

    # filter considered bands from the full band dictionary
    selected_band_dict: dict[str, BandCfg] = {k: cfg.bands[k] for k in plot_bands}

    # CLI override takes precedence
    if hasattr(args, 'catalog') and args.catalog:
        catalog_name = args.catalog
        logger.info(f'Using catalog from CLI: {catalog_name}')
    elif catalog_name.lower() == 'auto' or not catalog_name.strip():
        # Auto-detect most recent augmented catalog
        catalog_path = find_most_recent_catalog(table_dir)
        if catalog_path is None:
            logger.error('No augmented catalogs found in tables directory.')
            logger.info(
                "Run 'unionsdata download' with coordinates or a table first "
                'to create cutouts and augmented catalogs.'
            )
            sys.exit(1)
        # Extract catalog name from path (remove _augmented.csv suffix)
        catalog_name = catalog_path.stem.replace('_augmented', '')

    catalog_path = table_dir / f'{catalog_name}_augmented.csv'

    save_filename = build_plot_filename(
        catalog_name=catalog_name,
        size_pix=cutout_size,
        bands=plot_bands,
        band_dict=selected_band_dict,
        extension=cfg.plotting.save_format,
    )
    save_path = figure_dir / save_filename

    try:
        cutout_data = load_cutouts(
            catalog_path=catalog_path,
            bands_to_plot=plot_bands,
            data_dir=data_dir,
            cutout_subdir=cfg.cutouts.output_subdir,
            cutout_size=cutout_size,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        logger.error(f'{e}')
        sys.exit(1)

    cutout_data = cutouts_to_rgb(
        cutout_data=cutout_data,
        band_config=selected_band_dict,
        scaling_type=cfg.plotting.rgb.scaling_type,
        stretch=cfg.plotting.rgb.stretch,
        Q=cfg.plotting.rgb.Q,
        gamma=cfg.plotting.rgb.gamma,
        standard_zp=cfg.plotting.rgb.standard_zp,
    )

    plot_cutouts(
        cutout_data=cutout_data,
        mode=cfg.plotting.mode,
        max_cols=cfg.plotting.max_cols,
        figsize=cfg.plotting.figsize,
        save_path=save_path if cfg.plotting.save_plot else None,
        show_plot=cfg.plotting.show_plot,
    )

    plot_end = time.time()
    elapsed = plot_end - plot_start
    logger.debug(f'Plotting completed in {elapsed:.2f} seconds.')
    logger.info(f'Done! Plot saved to {save_path}')


def cli_entry() -> None:
    """
    The main CLI entry point that dispatches to subcommands.
    """

    # Set up a basic logger for initial messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[
            RichHandler(
                rich_tracebacks=True, show_time=False, show_level=True, show_path=False, markup=True
            )
        ],
        force=True,
    )

    parser = argparse.ArgumentParser(
        description='UNIONS data download tool.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Global arguments ---
    parser.add_argument(
        '--config', type=Path, help='Path to a specific config file (overrides default search)'
    )

    parser.add_argument(
        '--show-config', action='store_true', help='Print the resolved config path and exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )

    subparsers = parser.add_subparsers(dest='command', required=False)

    # --- 'init' subcommand ---
    parser_init = subparsers.add_parser('init', help='Create a new user config file')
    parser_init.add_argument('--force', action='store_true', help='Overwrite existing config file')
    parser_init.set_defaults(func=run_init)

    # --- 'edit' subcommand ---
    parser_edit = subparsers.add_parser(
        'edit', help='Open the user config file in your default editor'
    )
    parser_edit.set_defaults(func=run_edit)

    # --- 'config' subcommand ---
    parser_config = subparsers.add_parser(
        'config',
        help='Open the interactive TUI configuration editor (recommended)',
    )
    parser_config.set_defaults(func=run_config)

    # --- 'download' subcommand ---
    parser_download = subparsers.add_parser(
        'download',
        help='Download UNIONS survey imaging data (default command)',
        aliases=['run'],
    )

    parser_plot = subparsers.add_parser(
        'plot',
        help='Plot RGB object cutouts from input table or coordinates',
    )
    parser_plot.set_defaults(func=run_plot)

    # Add all original arguments from parse_arguments() here
    input_group = parser_download.add_mutually_exclusive_group()
    input_group.add_argument(
        '--coordinates',
        nargs='+',
        type=float,
        help='List of RA/Dec coordinate pairs: --coordinates ra1 dec1 ra2 dec2 ...',
    )
    input_group.add_argument(
        '--table',
        type=str,
        help='Path to CSV file containing coordinates',
    )
    input_group.add_argument(
        '--tiles',
        type=int,
        nargs='+',
        help='List of tile number pairs: --tiles x1 y1 x2 y2 ...',
    )
    input_group.add_argument(
        '--all-tiles',
        action='store_true',
        help='Download all available tiles (use with caution!)',
    )
    parser_download.add_argument(
        '--bands',
        nargs='+',
        type=str,
        choices=['cfis-u', 'whigs-g', 'cfis-r', 'cfis_lsb-r', 'ps-i', 'wishes-z', 'ps-z'],
        help='Bands to download (overrides config file)',
    )

    parser_download.add_argument(
        '--update-tiles',
        action='store_true',
        help='Update the local tile availability information before downloading',
    )

    parser_download.add_argument(
        '--cutouts',
        action='store_true',
        help='Enable cutout creation (overrides config file)',
    )

    parser_download.set_defaults(func=run_download)

    # --- Determine known subcommands and global flags ---
    known_subcommands = set(subparsers.choices.keys())

    global_flags = {
        action.option_strings[0]  # Get first option string (e.g., '--config')
        for action in parser._actions
        if action.option_strings  # Only actions with option strings (flags)
    }

    if len(sys.argv) > 1:
        first_arg = sys.argv[1]

        # Only inject 'download' if first arg is NOT a subcommand and NOT a global flag
        if first_arg not in known_subcommands and first_arg not in global_flags:
            sys.argv.insert(1, 'download')
    else:
        # No arguments at all: default to 'download'
        sys.argv.append('download')

    # Parse args
    args = parser.parse_args()

    # Handle --show-config global flag
    if hasattr(args, 'show_config') and args.show_config:
        try:
            cfg = load_settings(config_path=args.config)
            cfg_dict = settings_to_dict(cfg)
            cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=False)
            print(f'#################### Configuration ####################\n\n{cfg_yaml}')
            print('########################################################')
            sys.exit(0)
        except FileNotFoundError as e:
            logger.error(f'Config file not found: {e}')
            logger.info("Run 'unionsdata init' to create a config file")
            sys.exit(1)
        except Exception as e:
            logger.error(f'Error loading config: {e}')
            sys.exit(1)

    # Call the function associated with the chosen subcommand
    args.func(args)


if __name__ == '__main__':
    cli_entry()
