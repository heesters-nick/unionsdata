import logging
import multiprocessing
import posixpath
import queue
import signal
import subprocess
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from pathlib import Path
from queue import Queue
from threading import Event
from types import FrameType
from typing import TypedDict

import numpy as np
import pandas as pd
import requests

from unionsdata.config import BandDict, CutoutsCfg
from unionsdata.cutouts import create_cutouts_for_tile
from unionsdata.utils import decompress_fits, split_by_tile, tile_str
from unionsdata.verification import get_file_size, verify_download

logger = logging.getLogger(__name__)
QUEUE_TIMEOUT = 1  # seconds


class TileBandSpec(TypedDict):
    fitsfilename: str
    final_path: Path
    temp_path: Path
    vos_path: str
    http_url: str
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
    # get HTTP URL for size verification
    http_url = vos_path.replace('vos:', 'https://ws-cadc.canfar.net/vault/files/')

    if fits_ext != 0:
        # VCP path gets [fits_ext] for non-primary extensions
        vos_path = vos_path + f'[{fits_ext}]'

    return {
        'fitsfilename': tile_fitsfilename,
        'final_path': final_path,
        'temp_path': temp_path,
        'vos_path': vos_path,
        'http_url': http_url,
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
    http_url: str,
    band: str,
    shutdown_flag: Event,
    cert_path: Path,
    max_retries: int,
) -> bool:
    """
    Download a tile in a specific band.

    Args:
        tile_numbers: tile numbers
        tile_fitsname: tile fits filename
        final_path: path to file after download complete
        temp_path: path to file while download ongoing
        vos_path: path to file on server
        http_url: HTTP URL to file on server
        band: band name
        shutdown_flag: Event to signal shutdown
        cert_path: path to SSL certificate for verification
        max_retries: maximum number of download retries

    Returns:
        success/failure
    """
    session = None
    expected_file_size = None
    try:
        session = make_session(cert_path)
        header_content = fetch_fits_header(url=http_url, session=session)
        if header_content is not None:
            expected_file_size = get_file_size(header_content)
        else:
            logger.warning(f'Could not fetch header for download verification: {http_url}')
    except Exception as e:
        logger.warning(f'Error fetching header for download verification: {e}')
    finally:
        if session is not None:
            session.close()

    if final_path.is_file():
        if verify_download(final_path, expected_file_size):
            logger.info(f'File {tile_fitsname} already downloaded and verified for band {band}.')
            return True
        else:
            logger.warning(f'File {tile_fitsname} exists but failed verification, re-downloading.')
            final_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.info(
                    f'Retry {attempt}/{max_retries} for tile {tile_fitsname} in band {band}...'
                )
            else:
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

            # Decompress for specific bands
            if band in ['whigs-g', 'wishes-z']:
                try:
                    decompress_fits(final_path)
                    logger.debug(f'Decompressed {final_path} successfully.')
                except RuntimeError as e:
                    logger.error(f'Failed to decompress {final_path}: {e}')

                    # Delete the corrupted file
                    cleanup_temp_file(final_path)

                    if attempt < max_retries:
                        logger.info(
                            f'Retrying download for {tile_fitsname} (attempt {attempt}/{max_retries})...'
                        )
                        time.sleep(2**attempt)  # Exponential backoff
                        continue
                    else:
                        logger.error(
                            f'Maximum retries reached for {tile_fitsname}. Download failed.'
                        )
                        return False

            # Verify the download
            if not verify_download(final_path, expected_file_size):
                logger.error(f'Download verification failed for {tile_fitsname}')
                cleanup_temp_file(final_path)

                if attempt < max_retries:
                    logger.warning(f'Verification failed, retrying ({attempt}/{max_retries})...')
                    time.sleep(2**attempt)
                    continue
                else:
                    return False

            logger.info(
                f'Successfully downloaded tile {tile_str(tile_numbers)} for band {band} in {np.round(time.time() - start_time, 1)} seconds.'
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f'Failed downloading tile {tile_str(tile_numbers)} for band {band}.')
            logger.error(f'Subprocess error details: {e}')
            # Clean up temp file on error
            cleanup_temp_file(temp_path)

            if attempt < max_retries:
                logger.warning(f'Download failed, retrying ({attempt}/{max_retries})...')
                time.sleep(2**attempt)
                continue
            else:
                return False

        except FileNotFoundError:
            logger.error(f'Failed downloading tile {tile_str(tile_numbers)} for band {band}.')
            logger.exception(f'Tile {tile_str(tile_numbers)} not available in {band}.')
            # Clean up temp file on error
            cleanup_temp_file(temp_path)
            return False

        except Exception as e:
            logger.error(
                f'Tile {tile_str(tile_numbers)} in {band}: an unexpected error occurred: {e}'
            )
            # Clean up temp file on error
            cleanup_temp_file(temp_path)

            if attempt < max_retries:
                logger.warning(f'Unexpected error, retrying ({attempt}/{max_retries})...')
                time.sleep(2**attempt)
                continue
            else:
                return False

    # Should never get to this point but just in case
    return False


def download_worker(
    download_queue: Queue[tuple[tuple[int, int], str]],
    band_dictionary: dict[str, BandDict],
    download_dir: Path,
    shutdown_flag: Event,
    requested_bands: set[str],
    bands_with_jobs: set[str],
    tile_progress: dict[str, set[str]],
    tile_progress_lock: threading.Lock,
    cert_path: Path,
    max_retries: int,
    tile_catalogs: dict[str, pd.DataFrame],
    cutouts: CutoutsCfg,
    cutout_executor: ProcessPoolExecutor | None,
    cutout_futures: dict[str, Future[int]],
    cutout_futures_lock: threading.Lock,
) -> None:
    """
    Worker thread that downloads tiles from the queue.

    Args:
        download_queue: Queue containing (tile, band) tuples to download
        band_dictionary: Dictionary with band specifications
        download_dir: Directory to download files to
        shutdown_flag: Event to signal worker shutdown
        requested_bands: Set of bands that were requested for download
        bands_with_jobs: Set of bands that have download jobs
        tile_progress: Shared dict to track download progress per tile
        tile_progress_lock: Lock to synchronize access to tile_progress
        cert_path: Path to SSL certificate for verification
        max_retries: Maximum number of download retries
        tile_catalogs: Dictionary of different tile catalogs
        cutouts: Configuration for cutouts
        cutout_executor: ProcessPoolExecutor for cutout creation
        cutout_futures: Dictionary to track cutout futures by tile_key
        cutout_futures_lock: Lock to synchronize access to cutout_futures
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
                    http_url=paths['http_url'],
                    band=band,
                    shutdown_flag=shutdown_flag,
                    cert_path=cert_path,
                    max_retries=max_retries,
                )

                if success:
                    downloads_completed += 1

                    # Track progress for this tile
                    tile_str_key = tile_str(tile)

                    # Thread-safe update of tile progress
                    with tile_progress_lock:
                        tile_progress[tile_str_key].add(band)

                        # Check if tile is complete in all requested bands
                        tile_bands = tile_progress[tile_str_key].copy()  # copy for thread safety
                        remaining_bands_total = requested_bands - tile_bands
                        remaining_bands_jobs = bands_with_jobs - tile_bands

                    logger.debug(f'Tile {tile_str_key} downloaded in band {band}')

                    # Check if tile is complete
                    tile_complete = not remaining_bands_total or not remaining_bands_jobs

                    if tile_complete:
                        if not remaining_bands_total:
                            logger.info(
                                f'✓ Tile {tile_str_key} COMPLETE in all requested bands: {sorted(tile_bands)}'
                            )
                        else:
                            logger.info(
                                f'✓ Tile {tile_str_key} COMPLETE in all available bands: {sorted(tile_bands)}; missing band(s): {sorted(remaining_bands_total)}'
                            )

                        tile_catalog = tile_catalogs.get(tile_str_key, pd.DataFrame())

                        if cutout_executor is not None and len(tile_catalog) > 0:
                            # Make sure bands are in wavelength order
                            bands_sorted_by_wavelength = [
                                b for b in band_dictionary.keys() if b in tile_bands
                            ]
                            cutout_save_dir = paths['tile_dir'] / cutouts.output_subdir
                            try:
                                future = cutout_executor.submit(
                                    create_cutouts_for_tile,
                                    tile=tile,
                                    tile_dir=paths['tile_dir'],
                                    bands=bands_sorted_by_wavelength,
                                    catalog=tile_catalog,
                                    band_dictionary=band_dictionary,
                                    output_dir=cutout_save_dir,
                                    cutout_size=cutouts.size_pix,
                                )
                                with cutout_futures_lock:
                                    cutout_futures[tile_str_key] = future

                                logger.debug(
                                    f'Submitted tile {tile_str_key} for cutout creation '
                                    f'({len(tile_catalog)} objects)'
                                )
                            except Exception as e:
                                logger.error(f'Failed to submit cutout job for {tile_str_key}: {e}')

                    else:
                        logger.info(
                            f'  Tile {tile_str_key} progress: {sorted(tile_bands)}, remaining: {sorted(remaining_bands_jobs)}'
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
    num_threads: int,
    num_cutout_workers: int,
    cert_path: Path,
    max_retries: int,
    catalog: pd.DataFrame,
    cutouts: CutoutsCfg,
) -> tuple[int, int, int, dict[str, list[str]]]:
    """
    Download a list of tiles using multiple worker threads.

    Args:
        tiles_to_download: List of (tile, band) tuples to download
        band_dictionary: Dictionary with band specifications
        download_dir: Directory to download files to
        requested_bands: Set of bands that were requested
        num_threads: Number of worker threads to use
        num_cutout_workers: Number of cutout worker processes
        cert_path: Path to SSL certificate for verification
        max_retries: Maximum number of download retries
        catalog: DataFrame with information about input sources
        cutouts: Cutouts configuration

    Returns:
        tuple: (total_jobs, completed_jobs, failed_jobs, tile_cutout_info)
    """

    # Create queue and threading objects
    download_queue: Queue[tuple[tuple[int, int] | None, str | None]] = Queue()
    shutdown_flag = Event()

    # Shared dictionary to track download progress per tile
    tile_progress: dict[str, set[str]] = {}
    for tile, _ in tiles_to_download:
        tile_key = tile_str(tile)
        tile_progress[tile_key] = set()
    tile_progress_lock = threading.Lock()

    # Multiprocess setup for cutout creation
    cutout_executor: ProcessPoolExecutor | None = None
    cutout_futures: dict[str, Future[int]] = {}  # Track futures by tile_key
    cutout_futures_lock = threading.Lock()

    if cutouts.enable:
        # Use spawn method to avoid fork() warning with threads
        mp_context = multiprocessing.get_context('spawn')
        cutout_executor = ProcessPoolExecutor(max_workers=num_cutout_workers, mp_context=mp_context)
        logger.info(f'Started ProcessPoolExecutor with {num_cutout_workers} workers')

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

    bands_with_jobs = {band for _, band in tiles_to_download}
    logger.info(f'Bands with download jobs: {sorted(bands_with_jobs)}')
    logger.info(f'Requested bands: {sorted(requested_bands)}')

    # Split catalog by tile for cutout processing
    tile_catalogs = split_by_tile(catalog, list(tile_progress.keys()))

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
                bands_with_jobs,
                tile_progress,
                tile_progress_lock,
                cert_path,
                max_retries,
                tile_catalogs,
                cutouts,
                cutout_executor,
                cutout_futures,
                cutout_futures_lock,
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

        # Collect info about cutouts created
        tile_cutout_info: dict[str, list[str]] = {}

        # Shutdown cutout executor
        if cutout_executor is not None:
            with cutout_futures_lock:
                n_cutout_jobs = len(cutout_futures)

            if n_cutout_jobs > 0:
                logger.info(f'Waiting for {n_cutout_jobs} cutout jobs to complete...')

                # Don't submit new jobs, but wait for existing ones
                cutout_executor.shutdown(wait=True, cancel_futures=False)

                # Validate results
                cutouts_succeeded = 0
                cutouts_failed = 0
                failed_tiles = []

                with cutout_futures_lock:
                    for tile_key, future in cutout_futures.items():
                        try:
                            n_cutouts = future.result(timeout=0.1)  # already completed
                            if n_cutouts > 0:
                                with tile_progress_lock:
                                    tile_bands = sorted(tile_progress[tile_key])
                                tile_cutout_info[tile_key] = tile_bands

                                cutouts_succeeded += n_cutouts
                                logger.debug(f'✓ Cutouts for tile {tile_key}: {n_cutouts} objects')
                            else:
                                logger.warning(
                                    f'⚠ Tile {tile_key}: created 0 cutouts (objects outside bounds?)'
                                )
                        except TimeoutError:
                            cutouts_failed += 1
                            failed_tiles.append(tile_key)
                            logger.error(f'✗ Cutouts timed out for tile {tile_key} (>300s)')
                        except Exception as e:
                            cutouts_failed += 1
                            failed_tiles.append(tile_key)
                            logger.error(
                                f'✗ Cutouts failed for tile {tile_key}: {e}', exc_info=True
                            )

                logger.info('=' * 70)
                logger.info('CUTOUT SUMMARY')
                logger.info(f'  Total tiles processed: {n_cutout_jobs}')
                logger.info(f'  Successful: {n_cutout_jobs - cutouts_failed}')
                logger.info(f'  Failed: {cutouts_failed}')
                logger.info(f'  Total cutouts created: {cutouts_succeeded}')

                if failed_tiles:
                    logger.warning(f'  Failed tiles: {", ".join(failed_tiles)}')

                logger.info('=' * 70)
            else:
                logger.info('No cutout jobs were submitted')
                cutout_executor.shutdown(wait=False)

        cleanup_temp_files(download_dir)

    # Count remaining jobs (failures)
    remaining_jobs = download_queue.qsize()
    completed_jobs = total_jobs - remaining_jobs
    failed_jobs = remaining_jobs

    if not cutouts.enable:
        logger.info('=' * 70)
    logger.info('DOWNLOAD SUMMARY:')
    logger.info(f'  {len(complete_tiles)} tiles downloaded in all requested bands.')
    logger.info(f'  {completed_jobs}/{total_jobs} jobs completed.')
    logger.info(f'  {failed_jobs} jobs failed.')
    logger.info('=' * 70)

    return total_jobs, completed_jobs, failed_jobs, tile_cutout_info


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


def make_session(cert_path: Path) -> requests.Session:
    """Create a requests session with CADC certificate authentication."""
    if not cert_path.exists():
        raise FileNotFoundError(
            f'CADC certificate file not found: {cert_path}. '
            f'Generate it using cadc-get-cert -u YOUR_CANFAR_USERNAME'
        )
    session = requests.Session()
    session.cert = (str(cert_path), str(cert_path))
    return session


def fetch_fits_header(
    url: str, session: requests.Session, retries: int = 3, timeout: int = 10
) -> str | None:
    """
    Fetch FITS header from a file URL with retry logic.

    Args:
        url: Base FITS file URL (without ?META=true)
        session: Pre-configured requests.Session with authentication
        retries: Number of retry attempts on failure
        timeout: Request timeout in seconds

    Returns:
        FITS header content as string, or None if fetch fails
    """
    header_url = url + '?META=true'

    for attempt in range(retries + 1):
        try:
            response = session.get(header_url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f'Attempt {attempt + 1}/{retries + 1} failed for {url}: {e}')
            if attempt < retries:
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s

    logger.error(f'Failed to fetch header after {retries + 1} attempts: {url}')
    return None
