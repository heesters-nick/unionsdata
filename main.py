import argparse
import logging
import multiprocessing
import os
import time
from datetime import timedelta

import numpy as np

from logging_setup import setup_logger

setup_logger(
    log_dir="./logs",
    name="test",
    logging_level=logging.INFO,
)
logger = logging.getLogger()

from download import download_tiles  # noqa: E402
from utils import (  # noqa: E402
    TileAvailability,
    import_coordinates,
    import_dataframe,
    import_tiles,
    query_availability,
    tile_finder,
)

# To work with the client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .

# define the band directory containing
# information on the different
# photometric bands in the
# survey and their file systems


band_dictionary = {
    "cfis-u": {
        "name": "CFIS",
        "band": "u",
        "vos": "vos:cfis/tiles_DR5/",
        "suffix": ".u.fits",
        "delimiter": ".",
        "fits_ext": 0,
        "zfill": 3,
        "zp": 30.0,
    },
    "whigs-g": {
        "name": "calexp-CFIS",
        "band": "g",
        "vos": "vos:cfis/whigs/stack_images_CFIS_scheme/",
        "suffix": ".fits",
        "delimiter": "_",
        "fits_ext": 1,
        "zfill": 0,
        "zp": 27.0,
    },
    "cfis_lsb-r": {
        "name": "CFIS_LSB",
        "band": "r",
        "vos": "vos:cfis/tiles_LSB_DR5/",
        "suffix": ".r.fits",
        "delimiter": ".",
        "fits_ext": 0,
        "zfill": 3,
        "zp": 30.0,
    },
    "ps-i": {
        "name": "PSS.DR4",
        "band": "i",
        "vos": "vos:cfis/panstarrs/DR4/resamp/",
        "suffix": ".i.fits",
        "delimiter": ".",
        "fits_ext": 0,
        "zfill": 3,
        "zp": 30.0,
    },
    "wishes-z": {
        "name": "WISHES",
        "band": "z",
        "vos": "vos:cfis/wishes_1/coadd/",
        "suffix": ".z.fits",
        "delimiter": ".",
        "fits_ext": 1,
        "zfill": 0,
        "zp": 27.0,
    },
    "ps-z": {
        "name": "PSS.DR4",
        "band": "ps-z",
        "vos": "vos:cfis/panstarrs/DR4/resamp/",
        "suffix": ".z.fits",
        "delimiter": ".",
        "fits_ext": 0,
        "zfill": 3,
        "zp": 30.0,
    },
}
########################################################################################
################################# Setup ################################################
########################################################################################

### Multiprocessing constants
DOWNLOAD_THREADS = 5

# define the bands to consider
considered_bands = ["cfis-u", "whigs-g", "cfis_lsb-r", "ps-i", "wishes-z"]
# create a dictionary with the bands to consider
band_dict_incl = {key: band_dictionary.get(key) for key in considered_bands}

update_tiles = True  # whether to update the available tiles
# build kd tree with updated tiles otherwise use the already saved tree
if update_tiles:
    build_new_kdtree = True
else:
    build_new_kdtree = False
# return the number of available tiles that are available in at least 5, 4, 3, 2, 1 bands
at_least_key = False
# show stats on currently available tiles, remember to update
show_tile_statistics = True
band_constraint = (
    1  # minimum number of bands that should be available for a tile to be considered
)

### paths ###
platform = "LOCAL"  #'CANFAR'
if platform == "CANFAR":
    root_dir_main = "/arc/home/heestersnick/UNIONS-DL"
    download_directory = os.path.join(root_dir_main, "data")
    os.makedirs(download_directory, exist_ok=True)
elif platform == "LOCAL":
    root_dir_main = "/home/nick/astro/UNIONS_data_download"
    download_directory = "/home/nick/astro/UNIONS_data_download/data"
    os.makedirs(download_directory, exist_ok=True)

# paths
# define the root directory
main_directory = root_dir_main
table_directory = os.path.join(main_directory, "tables")
os.makedirs(table_directory, exist_ok=True)

ra_key_script, dec_key_script, id_key_script = "ra", "dec", "ID"
# define where the information about the currently available tiles should be saved
tile_info_directory = os.path.join(main_directory, "tile_info/")
os.makedirs(tile_info_directory, exist_ok=True)
# define where figures should be saved
figure_directory = os.path.join(main_directory, "figures/")
os.makedirs(figure_directory, exist_ok=True)
# define where the logs should be saved
log_directory = os.path.join(main_directory, "logs/")
os.makedirs(log_directory, exist_ok=True)


########################################################################################
########################################################################################
########################################################################################


def input_to_tile_list(
    availability,
    band_constr,
    coordinates=None,
    dataframe_path=None,
    tiles=None,
    ra_key=None,
    dec_key=None,
    id_key=None,
    tile_info_dir=None,
    ra_key_default="ra",
    dec_key_default="dec",
    id_key_default="ID",
):
    """
    Process the input to get a list of tiles that are available in r and at least two other bands.

    Args:
        availability (TileAvailability): instance of the TileAvailability class
        band_constr (int): minimum number of bands that should be available
        coordinates (nested list, optional): coordinates from the command line. Defaults to None.
        dataframe_path (str, optional): path to dataframe. Defaults to None.
        tiles (nested list, optional): tiles from the command line. Defaults to None.
        ra_key (str, optional): right ascention key. Defaults to None.
        dec_key (str_, optional): declination key. Defaults to None.
        id_key (str, optional): ID key. Defaults to None.
        tile_info_dir (str, optional): path to save the tile information. Defaults to None.
        ra_key_default (str, optional): default right ascention key. Defaults to 'ra'.
        dec_key_default (str, optional): default declination key. Defaults to 'dec'.
        id_key_default (str, optional): default ID key. Defaults to 'ID'.

    Returns:
        list: list of tiles that are available in r and at least two other bands
        catalog (dataframe): updated catalog with tile information
    """

    if coordinates is not None:
        catalog, coord_c = import_coordinates(
            coordinates, ra_key_default, dec_key_default, id_key_default
        )
    elif dataframe_path is not None:
        catalog, coord_c = import_dataframe(
            dataframe_path,
            ra_key,
            dec_key,
            id_key,
            ra_key_default,
            dec_key_default,
            id_key_default,
        )
    elif tiles is not None:
        return import_tiles(tiles, availability, band_constr), None, None
    else:
        logging.info(
            "No coordinates or DataFrame provided. Processing all available tiles.."
        )
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        return None, None, None

    unique_tiles, tiles_x_bands, catalog = tile_finder(
        availability, catalog, coord_c, tile_info_dir, band_constr
    )

    return unique_tiles, tiles_x_bands, catalog


def main(
    update,
    bands,
    band_dict,
    show_tile_stats,
    at_least,
    build_kdtree,
    tile_info_dir,
    download_dir,
    band_constr,
    coordinates,
    dataframe_path,
    tiles,
    ra_key,
    dec_key,
    id_key,
    ra_key_default="ra",
    dec_key_default="dec",
    id_key_default="ID",
):
    """
    Main function to download astronomical survey tiles.
    """
    logger.info("Starting UNIONS data download")

    # Query availability of tiles
    logger.info("Querying tile availability...")
    availability, all_tiles = query_availability(
        update, band_dict, at_least, show_tile_stats, build_kdtree, tile_info_dir
    )

    # Process input to get list of tiles to download
    logger.info("Processing input to determine tiles to download...")
    _, tiles_x_bands, _ = input_to_tile_list(
        availability,
        band_constr,
        coordinates,
        dataframe_path,
        tiles,
        ra_key,
        dec_key,
        id_key,
        tile_info_dir,
        ra_key_default,
        dec_key_default,
        id_key_default,
    )

    # If we have a specific subset of tiles, filter the availability
    if tiles_x_bands is not None:
        selected_all_tiles = [
            [tile for tile in band_tiles if tile in tiles_x_bands]
            for band_tiles in all_tiles
        ]
        availability = TileAvailability(selected_all_tiles, band_dict, at_least)

    # Get tiles available in the specified bands and create download jobs
    logger.info(f"Getting tiles available in bands: {bands}")
    tiles_to_process = availability.get_tiles_for_bands(bands)

    # Create list of (tile, band) pairs for downloading
    download_jobs = []
    for tile in tiles_to_process:
        available_bands, _ = availability.get_availability(tile)
        for band in available_bands:
            if band in bands:  # Only download requested bands
                download_jobs.append((tile, band))

    logger.info(f"Total download jobs: {len(download_jobs)}")

    # Group jobs by band for logging
    jobs_by_band = {}
    for tile, band in download_jobs:
        jobs_by_band[band] = jobs_by_band.get(band, 0) + 1

    for band, count in jobs_by_band.items():
        logger.info(f"  {band}: {count} tiles")

    if not download_jobs:
        logger.warning("No tiles found to download!")
        return

    # Download the tiles
    logger.info(f"Starting downloads using {DOWNLOAD_THREADS} threads...")
    try:
        total_jobs, completed_jobs, failed_jobs = download_tiles(
            download_jobs,
            band_dictionary,
            download_dir,
            set(bands),  # Pass requested bands as a set
            DOWNLOAD_THREADS,
        )
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coordinates",
        nargs="+",
        type=float,
        action="append",
        metavar=("ra", "dec"),
        help="list of pairs of coordinates to make cutouts from",
    )
    parser.add_argument(
        "--dataframe", type=str, help="path to a CSV file containing the DataFrame"
    )
    parser.add_argument(
        "--ra_key", type=str, help="right ascension key in the DataFrame"
    )
    parser.add_argument("--dec_key", type=str, help="declination key in the DataFrame")
    parser.add_argument("--id_key", type=str, help="id key in the DataFrame")
    parser.add_argument(
        "--tiles",
        type=int,
        nargs="+",
        action="append",
        metavar=("tile"),
        help="list of tiles to make cutouts from",
    )

    args = parser.parse_args()

    # define the arguments for the main function

    arg_dict_main = {
        "update": update_tiles,
        "bands": considered_bands,
        "band_dict": band_dict_incl,
        "show_tile_stats": show_tile_statistics,
        "at_least": at_least_key,
        "build_kdtree": build_new_kdtree,
        "tile_info_dir": tile_info_directory,
        "download_dir": download_directory,
        "band_constr": band_constraint,
        "coordinates": args.coordinates,
        "dataframe_path": args.dataframe,
        "tiles": args.tiles,
        "ra_key": args.ra_key,
        "dec_key": args.dec_key,
        "id_key": args.id_key,
        "ra_key_default": "ra",
        "dec_key_default": "dec",
        "id_key_default": "ID",
    }

    start = time.time()
    main(**arg_dict_main)
    end = time.time()
    elapsed = end - start
    elapsed_string = str(timedelta(seconds=elapsed))
    hours, minutes, seconds = (
        np.float32(elapsed_string.split(":")[0]),
        np.float32(elapsed_string.split(":")[1]),
        np.float32(elapsed_string.split(":")[2]),
    )
    logger.info(
        f"Done! Execution took {hours} hours, {minutes} minutes, and {seconds:.2f} seconds."
    )
