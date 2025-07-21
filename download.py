import os
import queue
import subprocess
import threading
import time

import numpy as np

from logging_setup import get_logger
from utils import tile_str

logger = get_logger()
QUEUE_TIMEOUT = 1  # seconds


def tile_band_specs(tile, in_dict, band, download_dir):
    """
    Get the necessary information for downloading a tile in a specific band.

    Args:
        tile (tuple): tile numbers
        in_dict (dictionary): band dictionary containing the necessary info on the file properties
        band (str): band name
        download_dir (str): download directory

    Returns:
        tuple: tile_fitsfilename, file_path after download complete, temp_path while download ongoing, vos_path (path to file on server), fits extension of the data, zero point
    """
    vos_dir = in_dict[band]["vos"]
    prefix = in_dict[band]["name"]
    suffix = in_dict[band]["suffix"]
    delimiter = in_dict[band]["delimiter"]
    zfill = in_dict[band]["zfill"]
    fits_ext = in_dict[band]["fits_ext"]
    zp = in_dict[band]["zp"]
    tile_dir = os.path.join(download_dir, f"{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}")
    os.makedirs(tile_dir, exist_ok=True)
    tile_band_dir = os.path.join(tile_dir, band)
    os.makedirs(tile_band_dir, exist_ok=True)
    tile_fitsfilename = f"{prefix}{delimiter}{str(tile[0]).zfill(zfill)}{delimiter}{str(tile[1]).zfill(zfill)}{suffix}"
    temp_name = ".".join(tile_fitsfilename.split(".")[:-1]) + "_temp.fits"
    temp_path = os.path.join(tile_band_dir, temp_name)
    final_path = os.path.join(tile_band_dir, tile_fitsfilename)
    vos_path = os.path.join(vos_dir, tile_fitsfilename)
    return {
        "fitsfilename": tile_fitsfilename,
        "final_path": final_path,
        "temp_path": temp_path,
        "vos_path": vos_path,
        "fits_ext": fits_ext,
        "zp": zp,
        "tile_dir": tile_dir,
    }


def download_tile_one_band(tile_numbers, tile_fitsname, final_path, temp_path, vos_path, band):
    """
    Download a tile in a specific band.

    Args:
        tile_numbers (tuple): tile numbers
        tile_fitsname (str): tile fits filename
        final_path (str): path to file after download complete
        temp_path (str): path to file while download ongoing
        vos_path (str): path to file on server
        band (str): band name

    Returns:
        bool: success/failure
    """
    if os.path.exists(final_path):
        logger.info(f"File {tile_fitsname} was already downloaded for band {band}.")
        return True

    try:
        logger.info(f"Downloading {tile_fitsname} for band {band}...")
        start_time = time.time()
        result = subprocess.run(
            f"vcp -v {vos_path} {temp_path}", shell=True, stderr=subprocess.PIPE, text=True
        )

        result.check_returncode()

        os.rename(temp_path, final_path)
        logger.info(
            f"Successfully downloaded tile {tile_str(tile_numbers)} for band {band} in {np.round(time.time() - start_time, 1)} seconds."
        )
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed downloading tile {tile_str(tile_numbers)} for band {band}.")
        logger.error(f"Subprocess error details: {e}")
        return False

    except FileNotFoundError:
        logger.error(f"Failed downloading tile {tile_str(tile_numbers)} for band {band}.")
        logger.exception(f"Tile {tile_str(tile_numbers)} not available in {band}.")
        return False

    except Exception as e:
        logger.error(f"Tile {tile_str(tile_numbers)} in {band}: an unexpected error occurred: {e}")
        return False


def download_worker(
    download_queue,
    required_bands,
    band_dictionary,
    download_dir,
    shutdown_flag,
    queue_lock,
    processed_in_current_run,
    downloaded_bands,  # shared dictionary for tracking across threads
):
    """Worker that downloads data and tracks completion per tile"""
    worker_id = threading.get_ident()
    logger.debug(f"Download worker {worker_id} started")

    while not shutdown_flag.is_set():
        try:
            tile, band = download_queue.get(timeout=1)

            if tile is None:
                logger.info(f"Download worker {worker_id} received sentinel, exiting")
                break

            try:
                paths = tile_band_specs(
                    tile=tile, in_dict=band_dictionary, band=band, download_dir=download_dir
                )

                success = download_tile_one_band(
                    tile_numbers=tile,
                    tile_fitsname=paths["fitsfilename"],
                    final_path=paths["final_path"],
                    temp_path=paths["temp_path"],
                    vos_path=paths["vos_path"],
                    band=band,
                )

                if success:
                    # Get current state for this tile
                    tile_string = tile_str(tile)
                    if tile_string not in downloaded_bands:
                        downloaded_bands[tile_string] = {"bands": set(), "paths": {}}

                    # Check if all required bands are present
                    if downloaded_bands[tile_string]["bands"] == required_bands:
                        logger.info(f"Tile {tile} downloaded in all bands.")
                        del downloaded_bands[tile_string]
                else:
                    with queue_lock:
                        processed_in_current_run[band] += 1

            except Exception as e:
                logger.error(f"Error downloading tile {tile} band {band}: {str(e)}")
                with queue_lock:
                    processed_in_current_run[band] += 1

            finally:
                download_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Unexpected error in download worker {worker_id}: {str(e)}")
            if shutdown_flag.is_set():
                break

    logger.info(f"Download worker {worker_id} exiting")
