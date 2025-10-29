import argparse
import logging
import time
from datetime import timedelta
from typing import cast

import numpy as np
import yaml

from unionsdata.config import (
    BandDict,
    ensure_runtime_dirs,
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download UNIONS survey imaging data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific tiles
  python main.py --tiles 217 292 234 295

  # Download by coordinates
  python main.py --coordinates 227.3042 52.5285 231.4445 52.4447

  # Use a dataframe with custom columns
  python main.py --dataframe objects.csv

  # Override config settings
  python main.py --bands whigs-g cfis_lsb-r ps-i --threads 10 --update-tiles
        """,
    )
    # Input source arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
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

    # Runtime options
    parser.add_argument(
        '--bands',
        nargs='+',
        type=str,
        choices=['cfis-u', 'whigs-g', 'cfis_lsb-r', 'ps-i', 'wishes-z', 'ps-z'],
        help='Bands to download (overrides config file)',
    )

    parser.add_argument(
        '--update-tiles',
        action='store_true',
        help='Update the local tile availability information before downloading',
    )

    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Build dictionary of CLI overrides - only bands and input."""
    overrides = {}

    # Bands override
    if args.bands:
        overrides['bands'] = args.bands

    if args.update_tiles:
        overrides['update_tiles'] = True

    # Input overrides (mutually exclusive)
    if args.tiles:
        if len(args.tiles) % 2 != 0:
            raise ValueError('Tiles must be provided as pairs')
        overrides['tiles'] = [
            [args.tiles[i], args.tiles[i + 1]] for i in range(0, len(args.tiles), 2)
        ]
    elif args.coordinates:
        if len(args.coordinates) % 2 != 0:
            raise ValueError('Coordinates must be provided as pairs')
        overrides['coordinates'] = [
            [args.coordinates[i], args.coordinates[i + 1]]
            for i in range(0, len(args.coordinates), 2)
        ]
    elif args.dataframe:
        overrides['dataframe'] = args.dataframe
    elif args.all_tiles:
        overrides['all_tiles'] = True

    return overrides


def main() -> None:
    """
    CLI entry point to download UNIONS survey image tiles.
    """
    start = time.time()

    # Parse arguments and load configuration
    args = parse_arguments()
    overrides = build_cli_overrides(args)
    cfg = load_settings(cli_overrides=overrides)
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

    # filter considered bands from the full band dictionary
    band_dict: dict[str, BandDict] = {
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
    availability, all_tiles = query_availability(
        update=cfg.tiles.update_tiles,
        in_dict=band_dict,
        show_stats=cfg.tiles.show_tile_statistics,
        tile_info_dir=tile_info_dir,
    )

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
        tiles_set = set(tiles_x_bands)  # Convert list to set for faster lookup
        selected_all_tiles = [
            [tile for tile in band_tiles if tile in tiles_set] for band_tiles in all_tiles
        ]
        availability = TileAvailability(selected_all_tiles, band_dict)

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
            band_dictionary=band_dict,
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


if __name__ == '__main__':
    main()
