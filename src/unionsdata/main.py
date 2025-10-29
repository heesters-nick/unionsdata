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
from typing import cast

import numpy as np
import yaml

from unionsdata.config import (
    BandDict,
    ensure_runtime_dirs,
    get_user_config_dir,
    load_settings,
    purge_previous_run,
    settings_to_dict,
)
from unionsdata.download import download_tiles
from unionsdata.logging_setup import setup_logger
from unionsdata.utils import (
    TileAvailability,
    input_to_tile_list,
    query_availability,
)

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
    elif hasattr(args, 'dataframe') and args.dataframe:
        overrides['dataframe'] = args.dataframe
    elif hasattr(args, 'all_tiles') and args.all_tiles:
        overrides['all_tiles'] = True

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
        logger.error('No config file found!')
        logger.info(
            "Run 'unionsdata init' to create a config, or use --config /path/to/config.yaml"
        )
        sys.exit(1)
    # Get rid of any previous log files if resume = False
    purge_previous_run(cfg)

    setup_logger(
        log_dir=cfg.paths.log_directory,
        name=cfg.logging.name,
        logging_level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        force=True,
    )

    logger.info('Starting UNIONS data download')

    cfg_dict = settings_to_dict(cfg)

    # Print settings in human readable format
    cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=False)
    logger.info(f'Resolved config (YAML):\n{cfg_yaml}')

    all_band_dict: dict[str, BandDict] = {
        k: cast(BandDict, cfg.bands[k].model_dump(mode='python'))
        for k in cfg.bands.keys()  # ALL bands, not just cfg.runtime.bands
    }

    # filter considered bands from the full band dictionary
    selected_band_dict: dict[str, BandDict] = {
        k: cast(BandDict, cfg.bands[k].model_dump(mode='python')) for k in cfg.runtime.bands
    }
    # make sure necessary directories exist
    ensure_runtime_dirs(cfg=cfg)

    # Define frequently used variables
    bands = cfg.runtime.bands
    tile_info_dir = cfg.paths.tile_info_directory
    download_dir = cfg.paths.root_dir_data

    # Query availability of tiles
    logger.info('Querying tile availability...')
    _, all_tiles = query_availability(
        update=cfg.tiles.update_tiles,
        in_dict=all_band_dict,
        show_stats=cfg.tiles.show_tile_statistics,
        tile_info_dir=tile_info_dir,
    )

    # Filter tiles to only those in selected bands
    selected_tiles = [
        all_tiles[list(all_band_dict.keys()).index(band)] for band in cfg.runtime.bands
    ]
    availability = TileAvailability(selected_tiles, selected_band_dict)

    # Process input to get list of tiles to download
    logger.info('Processing input to determine tiles to download...')
    # get the list of tiles
    _, tiles_x_bands, _ = input_to_tile_list(
        availability,
        cfg.tiles.band_constraint,
        cfg.inputs,
        tile_info_dir,
    )

    if tiles_x_bands is not None:
        logger.info(f'Filtering to {len(tiles_x_bands)} tiles based on input criteria')
        tiles_set = set(tiles_x_bands)  # Convert list to set for faster lookup

        filtered_tiles = [
            [tile for tile in band_tiles if tile in tiles_set] for band_tiles in selected_tiles
        ]
        availability = TileAvailability(filtered_tiles, selected_band_dict)

    # Get tiles available in the specified bands and create download jobs
    logger.info(f'Getting tiles available in bands: {bands}')
    tiles_to_process = availability.get_tiles_for_bands(bands)

    # Create list of (tile, band) pairs for downloading
    download_jobs = []
    for tile in tiles_to_process:
        available_bands, _ = availability.get_availability(tile)
        for band in available_bands:
            if band in bands:  # Only download requested bands
                download_jobs.append((tile, band))

    logger.info(f'Total download jobs: {len(download_jobs)}')

    # Group jobs by band for logging
    jobs_by_band: dict[str, int] = {}
    for _, band in download_jobs:
        jobs_by_band[band] = jobs_by_band.get(band, 0) + 1

    for band, count in jobs_by_band.items():
        logger.info(f'  {band}: {count} tiles')

    if not download_jobs:
        logger.warning('No tiles found to download!')
        return

    # Download the tiles
    download_threads = min(cfg.runtime.n_download_threads, len(download_jobs))
    logger.info(f'Starting downloads using {download_threads} threads...')
    try:
        total_jobs, completed_jobs, failed_jobs = download_tiles(
            tiles_to_download=download_jobs,
            band_dictionary=selected_band_dict,
            download_dir=download_dir,
            requested_bands=set(bands),  # Pass requested bands as a set
            num_threads=download_threads,
        )
    except Exception as e:
        logger.error(f'Error during download: {e}')
        raise

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
        logger.info('You can now edit this file with your custom paths.')

    except FileNotFoundError:
        logger.error('Could not find the config template within the package.')
        sys.exit(1)
    except Exception as e:
        logger.error(f'Failed to write config file: {e}')
        sys.exit(1)


def run_edit(args: argparse.Namespace) -> None:
    """
    Open the user configuration file in the system's default editor.
    """

    user_config_path = get_user_config_dir() / 'config.yaml'

    if not user_config_path.exists():
        logger.error(f'No config file found at: {user_config_path}')
        logger.info("Please run 'unionsdata init' first.")
        sys.exit(1)

    # Get the system's default editor
    editor = os.environ.get('EDITOR', 'vim' if sys.platform != 'win32' else 'notepad')

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


def run_validate(args: argparse.Namespace) -> None:
    """Validate the configuration file."""

    try:
        cfg = load_settings(config_path=args.config)
        logger.info('✓ Config is valid!')
        logger.info(f'Loaded from: {cfg.config_source}')
    except Exception as e:
        logger.error(f'✗ Config validation failed: {e}')
        sys.exit(1)


def cli_entry() -> None:
    """
    The main CLI entry point that dispatches to subcommands.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)

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

    # --- 'validate' subcommand ---
    parser_validate = subparsers.add_parser('validate', help='Check if config is valid')
    parser_validate.set_defaults(func=run_validate)

    # --- 'download' subcommand ---
    parser_download = subparsers.add_parser(
        'download',
        help='Download UNIONS survey imaging data (default command)',
        aliases=['run'],  # You can add aliases
    )
    # Add all your original arguments from parse_arguments() here
    input_group = parser_download.add_mutually_exclusive_group()
    input_group.add_argument(
        '--coordinates',
        nargs='+',
        type=float,
        help='List of RA/Dec coordinate pairs: --coordinates ra1 dec1 ra2 dec2 ...',
    )
    input_group.add_argument(
        '--dataframe',
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
        choices=['cfis-u', 'whigs-g', 'cfis_lsb-r', 'ps-i', 'wishes-z', 'ps-z'],
        help='Bands to download (overrides config file)',
    )

    parser_download.add_argument(
        '--update-tiles',
        action='store_true',
        help='Update the local tile availability information before downloading',
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
