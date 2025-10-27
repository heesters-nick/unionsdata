import argparse
import logging
import multiprocessing
import time
from datetime import timedelta

import numpy as np
import yaml

from unionsdata.config import (
    ensure_runtime_dirs,
    load_settings,
    purge_previous_run,
    settings_to_dict,
)
from unionsdata.download import download_tiles
from unionsdata.utils import (
    TileAvailability,
    input_to_tile_list,
    query_availability,
)

logger = logging.getLogger(__name__)

# To work with the client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .


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


def main():
    """
    Main function to download UNIONS survey image tiles.
    """
    args = parse_arguments()
    overrides = vars(args)
    cfg = load_settings(cli_overrides=overrides)
    purge_previous_run(cfg)
    logger.info('Starting UNIONS data download')

    cfg_dict = settings_to_dict(cfg)

    # Print settings in human readable format
    cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=False)
    logger.info(f'Resolved config (YAML):\n{cfg_yaml}')

    # filter considered bands from the full band dictionary
    band_dict = {k: cfg.bands[k].model_dump(mode='python') for k in cfg.runtime.bands}
    # make sure necessary directories exist
    ensure_runtime_dirs(cfg=cfg)

    # Define frequently used variables
    download_threads = cfg.runtime.n_download_threads
    bands = cfg.runtime.bands
    tile_info_dir = cfg.paths.tile_info_directory
    download_dir = cfg.paths.download_directory

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
    jobs_by_band = {}
    for _, band in download_jobs:
        jobs_by_band[band] = jobs_by_band.get(band, 0) + 1

    for band, count in jobs_by_band.items():
        logger.info(f'  {band}: {count} tiles')

    if not download_jobs:
        logger.warning('No tiles found to download!')
        return

    # Download the tiles
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


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    start = time.time()
    main()
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
