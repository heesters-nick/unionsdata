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
    tile_dir = os.path.join(
        download_dir, f"{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}"
    )
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


def download_tile_one_band(
    tile_numbers, tile_fitsname, final_path, temp_path, vos_path, band
):
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
            f"vcp -v {vos_path} {temp_path}",
            shell=True,
            stderr=subprocess.PIPE,
            text=True,
        )

        result.check_returncode()

        os.rename(temp_path, final_path)
        logger.info(
            f"Successfully downloaded tile {tile_str(tile_numbers)} for band {band} in {np.round(time.time() - start_time, 1)} seconds."
        )
        return True

    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed downloading tile {tile_str(tile_numbers)} for band {band}."
        )
        logger.error(f"Subprocess error details: {e}")
        return False

    except FileNotFoundError:
        logger.error(
            f"Failed downloading tile {tile_str(tile_numbers)} for band {band}."
        )
        logger.exception(f"Tile {tile_str(tile_numbers)} not available in {band}.")
        return False

    except Exception as e:
        logger.error(
            f"Tile {tile_str(tile_numbers)} in {band}: an unexpected error occurred: {e}"
        )
        return False


def download_worker(
    download_queue,
    band_dictionary,
    download_dir,
    shutdown_flag,
    requested_bands,
    tile_progress,
):
    """
    Worker thread that downloads tiles from the queue.

    Args:
        download_queue: Queue containing (tile, band) tuples to download
        band_dictionary: Dictionary with band specifications
        download_dir: Directory to download files to
        shutdown_flag: Event to signal worker shutdown
        requested_bands: Set of bands that were requested for download
        tile_progress: Shared dict to track download progress per tile
    """
    worker_id = threading.get_ident()
    logger.debug(f"Download worker {worker_id} started")

    downloads_completed = 0
    downloads_failed = 0

    while not shutdown_flag.is_set():
        try:
            # Get next download job
            tile, band = download_queue.get(timeout=QUEUE_TIMEOUT)

            # Check for sentinel value (shutdown signal)
            if tile is None:
                logger.info(f"Download worker {worker_id} received shutdown signal")
                break

            try:
                # Get file paths and specifications
                paths = tile_band_specs(
                    tile=tile,
                    in_dict=band_dictionary,
                    band=band,
                    download_dir=download_dir,
                )

                # Download the file
                success = download_tile_one_band(
                    tile_numbers=tile,
                    tile_fitsname=paths["fitsfilename"],
                    final_path=paths["final_path"],
                    temp_path=paths["temp_path"],
                    vos_path=paths["vos_path"],
                    band=band,
                )

                if success:
                    downloads_completed += 1

                    # Track progress for this tile (fix for Manager dict with sets)
                    tile_str_key = f"{tile[0]:03d}_{tile[1]:03d}"

                    # Get current set, modify it, and reassign to ensure Manager detects change
                    current_bands = tile_progress.get(tile_str_key, set())
                    current_bands.add(band)
                    tile_progress[tile_str_key] = current_bands

                    # Check if tile is complete in all requested bands
                    tile_bands = tile_progress[tile_str_key]
                    remaining_bands = requested_bands - tile_bands

                    logger.info(f"Tile {tile_str_key} downloaded in band {band}")

                    if not remaining_bands:
                        logger.info(
                            f"✓ Tile {tile_str_key} COMPLETE in all requested bands: {sorted(tile_bands)}"
                        )
                    else:
                        logger.info(
                            f"  Tile {tile_str_key} progress: {sorted(tile_bands)}, remaining: {sorted(remaining_bands)}"
                        )

                else:
                    downloads_failed += 1

            except Exception as e:
                logger.error(
                    f"Error processing download job for tile {tile} band {band}: {e}"
                )
                downloads_failed += 1

            finally:
                download_queue.task_done()

        except queue.Empty:
            # Timeout waiting for queue item, continue loop
            continue
        except Exception as e:
            logger.error(f"Unexpected error in download worker {worker_id}: {e}")
            if shutdown_flag.is_set():
                break

    logger.info(
        f"Download worker {worker_id} exiting. Completed: {downloads_completed}, Failed: {downloads_failed}"
    )


def download_tiles(
    tiles_to_download, band_dictionary, download_dir, requested_bands, num_threads=4
):
    """
    Download a list of tiles using multiple worker threads.

    Args:
        tiles_to_download: List of (tile, band) tuples to download
        band_dictionary: Dictionary with band specifications
        download_dir: Directory to download files to
        requested_bands: Set of bands that were requested
        num_threads: Number of worker threads to use

    Returns:
        tuple: (total_jobs, completed_jobs, failed_jobs)
    """
    import multiprocessing

    # Create queue and threading objects
    download_queue = queue.Queue()
    shutdown_flag = threading.Event()

    # Shared dictionary to track download progress per tile
    manager = multiprocessing.Manager()
    tile_progress = manager.dict()

    # Add all download jobs to queue
    for tile, band in tiles_to_download:
        download_queue.put((tile, band))

    total_jobs = len(tiles_to_download)
    logger.info(
        f"Starting download of {total_jobs} tile-band combinations using {num_threads} threads"
    )

    # Start worker threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(
            target=download_worker,
            args=(
                download_queue,
                band_dictionary,
                download_dir,
                shutdown_flag,
                requested_bands,
                tile_progress,
            ),
            name=f"DownloadWorker-{i}",
        )
        t.start()
        threads.append(t)

    try:
        # Wait for all downloads to complete
        download_queue.join()
        logger.info("All download jobs completed")

        # Final summary of completed tiles
        complete_tiles = []
        incomplete_tiles = []

        for tile_key, bands in tile_progress.items():
            if requested_bands.issubset(bands):  # Fix: use issubset instead of >=
                complete_tiles.append(tile_key)
            else:
                missing = requested_bands - bands
                incomplete_tiles.append((tile_key, sorted(bands), sorted(missing)))

        logger.info(
            f"Final summary: {len(complete_tiles)} tiles completed in all requested bands"
        )
        if incomplete_tiles:
            logger.warning(f"{len(incomplete_tiles)} tiles incomplete:")
            for tile_key, downloaded, missing in incomplete_tiles:
                logger.warning(
                    f"  {tile_key}: downloaded {downloaded}, missing {missing}"
                )

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")

    finally:
        # Signal workers to shutdown
        shutdown_flag.set()

        # Send sentinel values to wake up workers
        for _ in range(num_threads):
            download_queue.put((None, None))

        # Wait for all threads to finish
        for t in threads:
            t.join(timeout=10)
            if t.is_alive():
                logger.warning(f"Thread {t.name} did not shut down cleanly")

    # Count remaining jobs (failures)
    remaining_jobs = download_queue.qsize()
    completed_jobs = total_jobs - remaining_jobs
    failed_jobs = remaining_jobs

    logger.info(
        f"Download summary: {completed_jobs}/{total_jobs} completed, {failed_jobs} failed"
    )

    return total_jobs, completed_jobs, failed_jobs
