import logging
import posixpath
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue
from threading import Event
from types import FrameType
from typing import TypedDict

import numpy as np

from unionsdata.config import BandDict
from unionsdata.utils import tile_str

logger = logging.getLogger(__name__)
QUEUE_TIMEOUT = 1  # seconds


class TileBandSpec(TypedDict):
    fitsfilename: str
    final_path: Path
    temp_path: Path
    vos_path: str
    fits_ext: int
    zp: float
    tile_dir: Path


def tile_band_specs(
    tile: tuple[int, int], in_dict: dict[str, BandDict], band: str, download_dir: Path
) -> TileBandSpec:
    """
    Get the necessary information for downloading a tile in a specific band.

    Args:
        tile: tile numbers
        in_dict: band dictionary containing the necessary info on the file properties
        band: band name
        download_dir: download directory

    Returns:
        dict: tile_fitsfilename, file_path after download complete, temp_path while download ongoing, vos_path (path to file on server), fits extension of the data, zero point
    """
    vos_dir = in_dict[band]['vos']
    prefix = in_dict[band]['name']
    suffix = in_dict[band]['suffix']
    delimiter = in_dict[band]['delimiter']
    zfill = in_dict[band]['zfill']
    fits_ext = in_dict[band]['fits_ext']
    zp = in_dict[band]['zp']

    tile_dir = Path(download_dir) / f'{tile[0]:0>3}_{tile[1]:0>3}'
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_band_dir = tile_dir / band
    tile_band_dir.mkdir(parents=True, exist_ok=True)

    tile_fitsfilename = (
        f'{prefix}{delimiter}{tile[0]:0>{zfill}}{delimiter}{tile[1]:0>{zfill}}{suffix}'
    )
    final_path = tile_band_dir / tile_fitsfilename
    temp_path = final_path.with_name(final_path.stem + '_temp.fits')
    vos_path = posixpath.join(vos_dir, tile_fitsfilename)
    if fits_ext != 0:
        # add [fits_ext] for non-primary extensions
        vos_path = vos_path + f'[{fits_ext}]'

    return {
        'fitsfilename': tile_fitsfilename,
        'final_path': final_path,
        'temp_path': temp_path,
        'vos_path': vos_path,
        'fits_ext': fits_ext,
        'zp': zp,
        'tile_dir': tile_dir,
    }


def download_tile_one_band(
    tile_numbers: tuple[int, int],
    tile_fitsname: str,
    final_path: Path,
    temp_path: Path,
    vos_path: str,
    band: str,
    shutdown_flag: Event,
) -> bool:
    """
    Download a tile in a specific band.

    Args:
        tile_numbers: tile numbers
        tile_fitsname: tile fits filename
        final_path: path to file after download complete
        temp_path: path to file while download ongoing
        vos_path: path to file on server
        band: band name
        shutdown_flag: Event to signal shutdown

    Returns:
        success/failure
    """
    if final_path.is_file():
        logger.info(f'File {tile_fitsname} was already downloaded for band {band}.')
        return True

    try:
        logger.info(f'Downloading {tile_fitsname} for band {band}...')
        start_time = time.time()
        process = subprocess.Popen(
            ['vcp', '-v', vos_path, str(temp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        while process.poll() is None:
            if shutdown_flag.is_set():
                logger.warning(f'Shutdown requested, terminating download of {tile_fitsname}')
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                # Clean up temp file if it exists
                cleanup_temp_file(temp_path)
                return False
            time.sleep(0.1)

        # Check if process completed successfully
        if process.returncode != 0:
            stdout, stderr = process.communicate()
            if stdout:
                logger.debug('vcp stdout:\n%s', stdout)
            if stderr:
                logger.debug('vcp stderr:\n%s', stderr)
            raise subprocess.CalledProcessError(process.returncode, process.args)

        # change to path mode
        temp_path.rename(final_path)
        logger.info(
            f'Successfully downloaded tile {tile_str(tile_numbers)} for band {band} in {np.round(time.time() - start_time, 1)} seconds.'
        )
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f'Failed downloading tile {tile_str(tile_numbers)} for band {band}.')
        logger.error(f'Subprocess error details: {e}')
        # Clean up temp file on error
        cleanup_temp_file(temp_path)
        return False

    except FileNotFoundError:
        logger.error(f'Failed downloading tile {tile_str(tile_numbers)} for band {band}.')
        logger.exception(f'Tile {tile_str(tile_numbers)} not available in {band}.')
        # Clean up temp file on error
        cleanup_temp_file(temp_path)
        return False

    except Exception as e:
        logger.error(f'Tile {tile_str(tile_numbers)} in {band}: an unexpected error occurred: {e}')
        # Clean up temp file on error
        cleanup_temp_file(temp_path)
        return False


def download_worker(
    download_queue: Queue[tuple[tuple[int, int], str]],
    band_dictionary: dict[str, BandDict],
    download_dir: Path,
    shutdown_flag: Event,
    requested_bands: set[str],
    tile_progress: dict[str, set[str]],
    tile_progress_lock: threading.Lock,
) -> None:
    """
    Worker thread that downloads tiles from the queue.

    Args:
        download_queue: Queue containing (tile, band) tuples to download
        band_dictionary: Dictionary with band specifications
        download_dir: Directory to download files to
        shutdown_flag: Event to signal worker shutdown
        requested_bands: Set of bands that were requested for download
        tile_progress: Shared dict to track download progress per tile
        tile_progress_lock: Lock to synchronize access to tile_progress
    """
    worker_id = threading.get_ident()
    logger.debug(f'Download worker {worker_id} started')

    downloads_completed = 0
    downloads_failed = 0

    while not shutdown_flag.is_set():
        try:
            # Get next download job
            tile, band = download_queue.get(timeout=QUEUE_TIMEOUT)

            # Check for sentinel value (shutdown signal)
            if tile is None:
                logger.debug(f'Download worker {worker_id} received shutdown signal')
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
                    tile_fitsname=paths['fitsfilename'],
                    final_path=paths['final_path'],
                    temp_path=paths['temp_path'],
                    vos_path=paths['vos_path'],
                    band=band,
                    shutdown_flag=shutdown_flag,
                )

                if success:
                    downloads_completed += 1

                    # Track progress for this tile
                    tile_str_key = f'{tile[0]:03d}_{tile[1]:03d}'

                    # Thread-safe update of tile progress
                    with tile_progress_lock:
                        tile_progress[tile_str_key].add(band)

                        # Check if tile is complete in all requested bands
                        tile_bands = tile_progress[tile_str_key].copy()  # copy for thread safety
                        remaining_bands = requested_bands - tile_bands

                    logger.info(f'Tile {tile_str_key} downloaded in band {band}')

                    if not remaining_bands:
                        logger.info(
                            f'✓ Tile {tile_str_key} COMPLETE in all requested bands: {sorted(tile_bands)}'
                        )
                    else:
                        logger.info(
                            f'  Tile {tile_str_key} progress: {sorted(tile_bands)}, remaining: {sorted(remaining_bands)}'
                        )

                else:
                    downloads_failed += 1

            except Exception as e:
                logger.error(f'Error processing download job for tile {tile} band {band}: {e}')
                downloads_failed += 1

            finally:
                download_queue.task_done()

        except queue.Empty:
            # Timeout waiting for queue item, continue loop
            continue
        except Exception as e:
            logger.error(f'Unexpected error in download worker {worker_id}: {e}')
            if shutdown_flag.is_set():
                break

    logger.debug(
        f'Download worker {worker_id} exiting. Completed: {downloads_completed}, Failed: {downloads_failed}'
    )


def download_tiles(
    tiles_to_download: list[tuple[tuple[int, int], str]],
    band_dictionary: dict[str, BandDict],
    download_dir: Path,
    requested_bands: set[str],
    num_threads: int = 4,
) -> tuple[int, int, int]:
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

    # Create queue and threading objects
    download_queue: Queue[tuple[tuple[int, int] | None, str | None]] = Queue()
    shutdown_flag = Event()

    # Shared dictionary to track download progress per tile
    tile_progress: dict[str, set[str]] = {}
    for tile, _ in tiles_to_download:
        tile_key = f'{tile[0]:03d}_{tile[1]:03d}'
        tile_progress[tile_key] = set()
    tile_progress_lock = threading.Lock()

    # Signal handler for graceful shutdown
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        logger.warning(f'Received signal {signum}, initiating graceful shutdown...')
        shutdown_flag.set()

    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    # Add all download jobs to queue
    for tile, band in tiles_to_download:
        download_queue.put((tile, band))

    total_jobs = len(tiles_to_download)
    logger.info(
        f'Starting download of {total_jobs} tile-band combinations using {num_threads} threads'
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
                tile_progress_lock,
            ),
            name=f'DownloadWorker-{i}',
        )
        t.start()
        threads.append(t)

    # Final summary of completed tiles
    complete_tiles: list[str] = []
    incomplete_tiles: list[tuple[str, list[str], list[str]]] = []

    try:
        # Wait for all downloads to complete
        while not shutdown_flag.is_set():
            try:
                if download_queue.unfinished_tasks == 0:
                    break
                time.sleep(1)
            except KeyboardInterrupt:
                logger.warning('Keyboard interrupt detected, shutting down...')
                shutdown_flag.set()
                break

        if not shutdown_flag.is_set():
            logger.info('All download jobs completed')

        for tile_key, downloaded_bands in tile_progress.items():
            if requested_bands.issubset(downloaded_bands):
                complete_tiles.append(tile_key)
            else:
                missing = requested_bands - downloaded_bands
                incomplete_tiles.append((tile_key, sorted(downloaded_bands), sorted(missing)))

        if incomplete_tiles:
            logger.warning(f'{len(incomplete_tiles)} tiles incomplete:')
            for tile_key, downloaded, missing_bands in incomplete_tiles:
                logger.warning(f'  {tile_key}: downloaded {downloaded}, missing {missing_bands}')

    except KeyboardInterrupt:
        logger.info('Download interrupted by user')

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        # Signal workers to shutdown
        logger.debug('Signaling workers to shut down...')
        shutdown_flag.set()

        # Send sentinel values to wake up workers
        for _ in range(num_threads):
            download_queue.put((None, None))

        # Wait for all threads to finish
        logger.info('Waiting for worker threads to finish...')
        for t in threads:
            t.join(timeout=30)
            if t.is_alive():
                logger.warning(f'Thread {t.name} did not shut down cleanly')

        cleanup_temp_files(download_dir)

    # Count remaining jobs (failures)
    remaining_jobs = download_queue.qsize()
    completed_jobs = total_jobs - remaining_jobs
    failed_jobs = remaining_jobs

    indent = ' ' * 50
    logger.info(
        f'Download summary:\n'
        f'{indent}{len(complete_tiles)} tiles downloaded in all requested bands.\n'
        f'{indent}{completed_jobs}/{total_jobs} jobs completed.\n'
        f'{indent}{failed_jobs} jobs failed.'
    )

    return total_jobs, completed_jobs, failed_jobs


def cleanup_temp_file(temp_path: Path) -> None:
    """
    Clean up temp file OR associated .part file created by vcp.
    Only one will exist at a time:
    - During download: *_temp.fits-<hash>.part
    - After vcp completes: *_temp.fits

    Args:
        temp_path: Path to the temp file (e.g., *_temp.fits)
    """
    files_removed = 0

    # Clean up the main temp file (exists after vcp completes but before rename)
    if temp_path.exists():
        try:
            temp_path.unlink()
            logger.debug(f'Removed temp file: {temp_path}')
            files_removed += 1
        except Exception as e:
            logger.warning(f'Could not remove temp file {temp_path}: {e}')

    # Clean up any .part files (exist during active download)
    # Pattern: *_temp.fits-<hash>.part
    parent = temp_path.parent
    pattern = f'{temp_path.name}*.part'
    for part_file in parent.glob(pattern):
        try:
            part_file.unlink()
            logger.debug(f'Removed part file: {part_file}')
            files_removed += 1
        except Exception as e:
            logger.warning(f'Could not remove part file {part_file}: {e}')

    if files_removed > 0:
        logger.debug(f'Cleaned up {files_removed} file(s) for {temp_path.name}')


def cleanup_temp_files(download_dir: Path) -> None:
    """Clean up any temporary files left behind from interrupted downloads."""
    try:
        # Look for both _temp.fits files and .part files created by vcp
        temp_fits_files = list(download_dir.rglob('*_temp.fits'))
        part_files = list(download_dir.rglob('*_temp.fits*.part'))

        temp_files = temp_fits_files + part_files

        if temp_files:
            logger.info(f'Cleaning up {len(temp_files)} temporary files...')
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    logger.debug(f'Removed temp file: {temp_file}')
                except Exception as e:
                    logger.warning(f'Could not remove temp file {temp_file}: {e}')
    except Exception as e:
        logger.warning(f'Error during temp file cleanup: {e}')
