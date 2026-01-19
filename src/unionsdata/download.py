import logging
import multiprocessing
import posixpath
import queue
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from types import FrameType
from typing import TypedDict

import numpy as np
import pandas as pd
import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from unionsdata.config import BandCfg, CutoutsCfg
from unionsdata.cutouts import CutoutResult, create_cutouts_for_tile, worker_log_init
from unionsdata.stats import RunStatistics
from unionsdata.utils import decompress_fits, split_by_tile, tile_str
from unionsdata.verification import get_file_size, is_fits_valid, verify_download

logger = logging.getLogger(__name__)
QUEUE_TIMEOUT = 1  # seconds
_SHUTDOWN_SENTINEL = object()


# ========== Progress Tracking ==========


@dataclass
class DownloadJob:
    """Track a single file download with size information."""

    tile: tuple[int, int]
    band: str
    final_path: Path
    temp_path: Path
    expected_size: int = 0
    completed: bool = False


@dataclass
class ProgressTracker:
    """Thread-safe tracker for download progress with cached totals."""

    jobs: list[DownloadJob] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    _completed_bytes: int = field(default=0)  # Cache completed bytes
    _expected_total: int = field(default=0)  # Cache expected total

    def add_job(self, job: DownloadJob) -> None:
        with self._lock:
            self.jobs.append(job)
            self._expected_total += job.expected_size

    def mark_completed(self, final_path: Path) -> None:
        with self._lock:
            for job in self.jobs:
                if job.final_path == final_path and not job.completed:
                    job.completed = True
                    self._completed_bytes += job.expected_size
                    break

    def get_totals(self) -> tuple[int, int]:
        """Get (current_bytes, expected_bytes) efficiently."""
        with self._lock:
            if self._expected_total == 0:
                return 0, 0

            # Start with known completed bytes
            current_total = self._completed_bytes

            # Only check in-progress files (not completed ones)
            for job in self.jobs:
                if job.completed or job.expected_size == 0:
                    continue

                # Check actual file progress
                try:
                    if job.final_path.exists():
                        current_total += job.final_path.stat().st_size
                    elif job.temp_path.exists():
                        current_total += job.temp_path.stat().st_size
                    else:
                        # Check for .part files from vcp
                        for pf in job.temp_path.parent.glob(f'{job.temp_path.name}*.part'):
                            try:
                                current_total += pf.stat().st_size
                                break  # Only count one part file
                            except OSError:
                                pass
                except OSError:
                    pass

            return current_total, self._expected_total


def fetch_expected_sizes(
    jobs: list[tuple[tuple[int, int], str]],
    band_dictionary: dict[str, BandCfg],
    download_dir: Path,
    cert_path: Path,
    max_workers: int = 8,
) -> dict[Path, int]:
    """Fetch expected file sizes for all jobs in parallel."""

    path_to_size: dict[Path, int] = {}

    def fetch_one(tile: tuple[int, int], band: str) -> tuple[Path, int]:
        specs = tile_band_specs(tile, band_dictionary, band, download_dir)

        # Skip if file already exists (will be verified during download)
        if specs['final_path'].exists():
            try:
                return specs['final_path'], specs['final_path'].stat().st_size
            except OSError:
                pass

        session = make_session(cert_path)
        try:
            header = fetch_fits_header(specs['http_url'], session, retries=2, timeout=10)
            size = get_file_size(header) if header else 0
        except (requests.RequestException, ValueError, KeyError):
            size = 0
        finally:
            session.close()
        return specs['final_path'], size

    logger.info(f'Fetching file size information for {len(jobs)} downloads...')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, tile, band) for tile, band in jobs]

        for future in futures:
            try:
                path, size = future.result()
                path_to_size[path] = size
            except Exception:
                pass

    total_size = sum(path_to_size.values())
    logger.debug(f'Total expected download size: {total_size / (1024**3):.2f} GB')

    return path_to_size


def progress_monitor(
    tracker: ProgressTracker,
    shutdown_flag: Event,
    update_interval: float = 0.5,
) -> None:
    """Monitor thread that updates progress bar based on file sizes."""
    # Wait briefly for jobs to be populated
    time.sleep(1.0)

    _, total_bytes = tracker.get_totals()

    if total_bytes == 0:
        logger.debug('No downloads to track in progress monitor.')
        return

    with Progress(
        SpinnerColumn(),
        TextColumn('[bold blue]{task.description}'),
        BarColumn(bar_width=None),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        '•',
        DownloadColumn(),
        '•',
        TransferSpeedColumn(),
        '•',
        TimeElapsedColumn(),
        '•',
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task_id = progress.add_task('Downloading', total=total_bytes)

        while not shutdown_flag.is_set():
            current_bytes, _ = tracker.get_totals()
            progress.update(task_id, completed=current_bytes)

            # Check if complete
            if current_bytes >= total_bytes:
                break

            time.sleep(update_interval)

        # Final update
        current_bytes, _ = tracker.get_totals()
        progress.update(task_id, completed=current_bytes)


# ========= Download Functions ==========


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
    tile: tuple[int, int],
    in_dict: dict[str, BandCfg],
    band: str,
    download_dir: Path,
    cutout_mode: str = 'disabled',
) -> TileBandSpec:
    """
    Get the necessary information for downloading a tile in a specific band.

    Args:
        tile: tile numbers
        in_dict: band dictionary containing the necessary info on the file properties
        band: band name
        download_dir: download directory
        cutout_mode: mode for cutout ('disabled' by default)

    Returns:
        dict: tile_fitsfilename, file_path after download complete, temp_path while download ongoing, vos_path (path to file on server), fits extension of the data, zero point
    """
    vos_dir = in_dict[band].vos
    prefix = in_dict[band].name
    suffix = in_dict[band].suffix
    delimiter = in_dict[band].delimiter
    zfill = in_dict[band].zfill
    fits_ext = in_dict[band].fits_ext
    zp = in_dict[band].zp

    tile_dir = Path(download_dir) / f'{tile[0]:0>3}_{tile[1]:0>3}'
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_band_dir = tile_dir / band
    if not cutout_mode == 'direct_only':
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
    fits_ext: int,
    band: str,
    shutdown_flag: Event,
    cert_path: Path,
    max_retries: int,
    expected_file_size: int | None = None,
) -> str:
    """
    Download a tile in a specific band.

    Args:
        tile_numbers: tile numbers
        tile_fitsname: tile fits filename
        final_path: path to file after download complete
        temp_path: path to file while download ongoing
        vos_path: path to file on server
        http_url: HTTP URL to file on server
        fits_ext: FITS extension to verify
        band: band name
        shutdown_flag: Event to signal shutdown
        cert_path: path to SSL certificate for verification
        max_retries: maximum number of download retries
        expected_file_size: expected file size in bytes inferred from header

    Returns:
        'downloaded', 'skipped', or 'failed'
    """
    # Use pre-fetched expected size if available otherwise fetch it now
    if expected_file_size is None or expected_file_size == 0:
        session = None
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
            logger.debug(
                f'Tile {tile_str(tile_numbers)} already downloaded and verified for band {band}.'
            )
            return 'skipped'
        else:
            logger.warning(f'File {tile_fitsname} exists but failed verification, re-downloading.')
            final_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.info(
                    f'Download attempt {attempt}/{max_retries} for tile {tile_fitsname} in band {band}...'
                )
            else:
                logger.debug(f'Downloading tile {tile_str(tile_numbers)} for band {band}...')

            start_time = time.time()

            process = subprocess.Popen(
                ['vcp', '-v', vos_path, str(temp_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout_data = []
            stderr_data = []

            while True:
                try:
                    # Wait for output or completion for 0.5 seconds
                    # communicate() reads pipes preventing deadlock
                    out, err = process.communicate(timeout=0.5)

                    if out:
                        stdout_data.append(out)
                    if err:
                        stderr_data.append(err)

                    # If we get here, process finished
                    break

                except subprocess.TimeoutExpired:
                    # Process is still running, check shutdown signal
                    if shutdown_flag.is_set():
                        logger.warning(
                            f'Shutdown requested, terminating download of {tile_fitsname}'
                        )
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        cleanup_temp_file(temp_path)
                        return 'failed'
                    # Continue waiting
                    continue

            # Reconstruct full output for logging if needed
            stdout = ''.join(stdout_data)
            stderr = ''.join(stderr_data)

            # Check if process completed successfully
            if process.returncode != 0:
                if stdout:
                    logger.info('vcp stdout:\n%s', stdout)
                if stderr:
                    logger.info('vcp stderr:\n%s', stderr)
                raise subprocess.CalledProcessError(process.returncode, process.args)

            # change to path mode
            temp_path.rename(final_path)

            if not is_fits_valid(final_path, fits_ext=fits_ext):
                logger.warning(f'Download corrupted (header check failed) for {tile_fitsname}.')
                cleanup_temp_file(final_path)
                continue

            # Decompress for specific bands
            if band in ['whigs-g', 'wishes-z']:
                try:
                    decompress_fits(final_path)
                    logger.debug(f'Decompressed {final_path} successfully.')
                except RuntimeError as e:
                    logger.error(f'Failed to decompress {final_path}: {e}. Retrying download...')

                    # Delete the corrupted file
                    cleanup_temp_file(final_path)

                    if attempt < max_retries:
                        time.sleep(2**attempt)  # Exponential backoff
                        continue
                    else:
                        logger.error(
                            f'Maximum retries reached for {tile_fitsname}. Download failed.'
                        )
                        return 'failed'

            # Verify the download
            if not verify_download(final_path, expected_file_size):
                logger.error(f'Download verification failed for {tile_fitsname}')
                cleanup_temp_file(final_path)

                if attempt < max_retries:
                    logger.warning('Verification failed.')
                    time.sleep(2**attempt)
                    continue
                else:
                    return 'failed'

            logger.debug(
                f'Successfully downloaded tile {tile_str(tile_numbers)} for band {band} in {np.round(time.time() - start_time, 1)} seconds.'
            )
            return 'downloaded'

        except subprocess.CalledProcessError as e:
            logger.warning(
                f'Failed downloading tile {tile_str(tile_numbers)} for band {band}. Retrying...'
            )
            logger.warning(f'Subprocess error details: {e}')
            # Clean up temp file on error
            cleanup_temp_file(temp_path)

            if shutdown_flag.is_set():
                return 'failed'

            if attempt < max_retries:
                logger.warning('Download failed. Retrying...')
                time.sleep(2**attempt)
                continue
            else:
                return 'failed'

        except FileNotFoundError as e:
            # Check if the vcp executable is actually installed
            if shutil.which('vcp') is None:
                logger.critical("The 'vcp' command was not found in your PATH.")
                logger.critical('Please install the VOSpace tools: pip install vos')
                # Trigger shutdown to stop other threads from failing identically
                shutdown_flag.set()
                cleanup_temp_file(temp_path)
                return 'failed'

            # If vcp exists, then the remote file is missing
            logger.error(
                f'Failed downloading tile {tile_str(tile_numbers)} for band {band}. '
                f'File likely missing from VOSpace.'
            )
            logger.debug(f'Details: {e}')
            # Clean up temp file on error
            cleanup_temp_file(temp_path)
            return 'failed'

        except Exception as e:
            logger.error(
                f'Tile {tile_str(tile_numbers)} in {band}: an unexpected error occurred: {e}'
            )
            # Clean up temp file on error
            cleanup_temp_file(temp_path)

            if shutdown_flag.is_set():
                return 'failed'

            if attempt < max_retries:
                logger.warning('Unexpected error. Retrying...')
                time.sleep(2**attempt)
                continue
            else:
                return 'failed'

    # Should never get to this point but just in case
    return 'failed'


def download_worker(
    download_queue: Queue[tuple[tuple[int, int], str]],
    band_dictionary: dict[str, BandCfg],
    all_band_dictionary: dict[str, BandCfg],
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
    cutout_futures: dict[str, Future[CutoutResult]],
    cutout_futures_lock: threading.Lock,
    tiles_claimed_for_cutout: set[str],
    progress_tracker: ProgressTracker,
    expected_sizes: dict[Path, int],
    run_stats: RunStatistics,
) -> None:
    """
    Worker thread that downloads tiles from the queue.

    Args:
        download_queue: Queue containing (tile, band) tuples to download
        band_dictionary: Dictionary with selected band specifications
        all_band_dictionary: Dictionary with all band specifications
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
        tiles_claimed_for_cutout: Set to track tiles already submitted for cutouts
        progress_tracker: Shared ProgressTracker instance
        expected_sizes: Dictionary mapping final_path to expected file size
        run_stats: RunStatistics object to track download statistics

    Returns:
        None
    """
    worker_id = threading.get_ident()
    logger.debug(f'Download worker {worker_id} started')

    downloads_completed = 0
    downloads_failed = 0

    while not shutdown_flag.is_set():
        try:
            # Get next download job
            job = download_queue.get(timeout=QUEUE_TIMEOUT)

            # Check for sentinel
            if job is _SHUTDOWN_SENTINEL:
                logger.debug(f'Download worker {worker_id} received shutdown signal')
                download_queue.task_done()
                break

            # Now we know it's a valid job
            tile, band = job

            try:
                # Get file paths and specifications
                paths = tile_band_specs(
                    tile=tile,
                    in_dict=band_dictionary,
                    band=band,
                    download_dir=download_dir,
                )

                # Get pre-fetched expected size
                expected_size = expected_sizes.get(paths['final_path'], 0)

                # Download the file
                status = download_tile_one_band(
                    tile_numbers=tile,
                    tile_fitsname=paths['fitsfilename'],
                    final_path=paths['final_path'],
                    temp_path=paths['temp_path'],
                    vos_path=paths['vos_path'],
                    http_url=paths['http_url'],
                    fits_ext=paths['fits_ext'],
                    band=band,
                    shutdown_flag=shutdown_flag,
                    cert_path=cert_path,
                    max_retries=max_retries,
                    expected_file_size=expected_size if expected_size > 0 else None,
                )

                # Update per-band stats safely
                run_stats.record_tile_download(band, status)

                if status in ['downloaded', 'skipped']:
                    downloads_completed += 1

                    # Mark as completed in progress tracker
                    if progress_tracker is not None:
                        progress_tracker.mark_completed(paths['final_path'])

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
                            logger.debug(
                                f'✓ Tile {tile_str_key} COMPLETE in all requested bands: {sorted(tile_bands)}'
                            )
                        else:
                            logger.debug(
                                f'✓ Tile {tile_str_key} COMPLETE in all available bands: {sorted(tile_bands)}; missing band(s): {sorted(remaining_bands_total)}'
                            )

                        tile_catalog = tile_catalogs.get(tile_str_key, pd.DataFrame())

                        if cutout_executor is not None and len(tile_catalog) > 0:
                            # Make sure bands are in wavelength order
                            bands_sorted_by_wavelength = [
                                b for b in band_dictionary.keys() if b in tile_bands
                            ]
                            cutout_save_dir = paths['tile_dir'] / cutouts.output_subdir

                            # Check and claim under lock
                            with cutout_futures_lock:
                                if tile_str_key in tiles_claimed_for_cutout:
                                    pass  # already claimed
                                else:
                                    try:
                                        future = cutout_executor.submit(
                                            create_cutouts_for_tile,
                                            tile=tile,
                                            tile_dir=paths['tile_dir'],
                                            bands=bands_sorted_by_wavelength,
                                            catalog=tile_catalog,
                                            all_band_dictionary=all_band_dictionary,
                                            output_dir=cutout_save_dir,
                                            cutout_size=cutouts.size_pix,
                                        )

                                        tiles_claimed_for_cutout.add(tile_str_key)
                                        cutout_futures[tile_str_key] = future

                                        logger.debug(
                                            f'Submitted tile {tile_str_key} for cutout creation '
                                            f'({len(tile_catalog)} objects)'
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f'Failed to submit cutout job for {tile_str_key}: {e}'
                                        )

                    else:
                        logger.debug(
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
    band_dictionary: dict[str, BandCfg],
    all_band_dictionary: dict[str, BandCfg],
    download_dir: Path,
    requested_bands: set[str],
    num_threads: int,
    num_cutout_workers: int,
    cert_path: Path,
    max_retries: int,
    catalog: pd.DataFrame,
    cutouts: CutoutsCfg,
    log_dir: Path,
    log_name: str,
    log_level: int,
    run_stats: RunStatistics,
    tile_success_map: dict[str, set[str]],
    cutout_success_map: dict[str, set[str]],
) -> None:
    """
    Download a list of tiles using multiple worker threads.

    Args:
        tiles_to_download: List of (tile, band) tuples to download
        band_dictionary: Dictionary with selected band specifications
        all_band_dictionary: Dictionary with all band specifications
        download_dir: Directory to download files to
        requested_bands: Set of bands that were requested
        num_threads: Number of worker threads to use
        num_cutout_workers: Number of cutout worker processes
        cert_path: Path to SSL certificate for verification
        max_retries: Maximum number of download retries
        catalog: DataFrame with information about input sources
        cutouts: Cutouts configuration
        log_dir: Directory to save logs
        log_name: Name of the log file
        log_level: Logging level
        run_stats: RunStatistics object to track download statistics
        tile_success_map: Dictionary to track successful downloads per tile
        cutout_success_map: Dictionary to track successful cutouts per tiles

    Returns:
        None
    """

    # Create queue and threading objects
    download_queue: Queue[tuple[tuple[int, int], str] | object] = Queue()
    shutdown_flag = Event()

    tile_success_map_lock = threading.Lock()

    # Shared dictionary to track download progress per tile
    for tile, _ in tiles_to_download:
        tile_key = tile_str(tile)
        with tile_success_map_lock:
            if tile_key not in tile_success_map:
                tile_success_map[tile_key] = set()

    # Multiprocess setup for cutout creation
    tiles_claimed_for_cutout: set[str] = set()
    cutout_executor: ProcessPoolExecutor | None = None
    cutout_futures: dict[str, Future[CutoutResult]] = {}  # Track futures by tile_key
    cutout_futures_lock = threading.Lock()

    if cutouts.mode != 'disabled':
        # Use spawn method to avoid fork() warning with threads
        mp_context = multiprocessing.get_context('spawn')
        cutout_executor = ProcessPoolExecutor(
            max_workers=num_cutout_workers,
            mp_context=mp_context,
            initializer=worker_log_init,
            initargs=(log_dir, log_name, log_level),
        )
        logger.debug(f'Started ProcessPoolExecutor with {num_cutout_workers} workers')

    # Signal handler for graceful shutdown
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        logger.warning(f'Received signal {signum}, initiating graceful shutdown...')
        shutdown_flag.set()

    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    fetch_expected_sizes_start = time.time()
    # Fetch expected file sizes in parallel for progress tracking
    expected_sizes = fetch_expected_sizes(
        jobs=tiles_to_download,
        band_dictionary=band_dictionary,
        download_dir=download_dir,
        cert_path=cert_path,
        max_workers=min(16, num_threads * 2),
    )
    fetch_expected_sizes_end = time.time()
    logger.debug(
        f'Fetched expected file sizes in {fetch_expected_sizes_end - fetch_expected_sizes_start:.1f} seconds'
    )

    # Set up progress tracker
    progress_tracker = ProgressTracker()
    for tile, band in tiles_to_download:
        specs = tile_band_specs(tile, band_dictionary, band, download_dir)
        job = DownloadJob(
            tile=tile,
            band=band,
            final_path=specs['final_path'],
            temp_path=specs['temp_path'],
            expected_size=expected_sizes.get(specs['final_path'], 0),
        )
        progress_tracker.add_job(job)

    # Add all download jobs to queue
    for tile, band in tiles_to_download:
        download_queue.put((tile, band))

    total_jobs = len(tiles_to_download)
    logger.info(f'Start processing {total_jobs} tile-band combinations...')

    bands_with_jobs = {band for _, band in tiles_to_download}
    logger.debug(f'Bands with download jobs: {sorted(bands_with_jobs)}')
    logger.debug(f'Requested bands: {sorted(requested_bands)}')

    # Split catalog by tile for cutout processing
    tile_catalogs = split_by_tile(catalog, list(tile_success_map.keys()))

    # Start progress monitor thread
    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_tracker, shutdown_flag),
        name='ProgressMonitor',
        daemon=True,
    )
    monitor_thread.start()

    # Start worker threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(
            target=download_worker,
            args=(
                download_queue,
                band_dictionary,
                all_band_dictionary,
                download_dir,
                shutdown_flag,
                requested_bands,
                bands_with_jobs,
                tile_success_map,
                tile_success_map_lock,
                cert_path,
                max_retries,
                tile_catalogs,
                cutouts,
                cutout_executor,
                cutout_futures,
                cutout_futures_lock,
                tiles_claimed_for_cutout,
                progress_tracker,
                expected_sizes,
                run_stats,
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
            logger.debug('All download jobs completed.')

        for tile_key, downloaded_bands in tile_success_map.items():
            expected_bands = {
                band for tile, band in tiles_to_download if tile_str(tile) == tile_key
            }

            if expected_bands.issubset(downloaded_bands):
                complete_tiles.append(tile_key)
            else:
                missing = expected_bands - downloaded_bands
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

        # Wait for monitor thread to finish (for clean progress bar closure)
        monitor_thread.join(timeout=2)

        # Send sentinel values to wake up workers
        for _ in range(num_threads):
            download_queue.put(_SHUTDOWN_SENTINEL)

        # Wait for all threads to finish
        logger.debug('Waiting for worker threads to finish...')
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
                logger.debug(f'Waiting for {n_cutout_jobs} cutout jobs to complete...')

                # Wait for all futures to complete BEFORE shutdown
                with cutout_futures_lock:
                    futures_to_wait = list(cutout_futures.items())

                # Collect results first (this blocks until complete)
                # To make sure log messages look cleanly ordered
                collected_results: dict[str, CutoutResult | Exception] = {}
                for tile_key, future in futures_to_wait:
                    try:
                        collected_results[tile_key] = future.result(timeout=300)
                    except Exception as e:
                        collected_results[tile_key] = e

                # Don't submit new jobs, but wait for existing ones
                cutout_executor.shutdown(wait=True, cancel_futures=False)

                # Small delay to let any buffered log messages flush
                time.sleep(0.2)

                # Process collected results
                for tile_key, result in collected_results.items():
                    if isinstance(result, Exception):
                        failed_count = len(tile_catalogs.get(tile_key, []))
                        for band in requested_bands:
                            run_stats.record_cutout_result(band, failed=failed_count)
                        logger.error(f'❌ Cutouts failed for tile {tile_key}: {result}')
                        continue

                    n_objects_processed = len(result.object_bands)
                    if n_objects_processed > 0:
                        with tile_success_map_lock:
                            tile_bands = sorted(tile_success_map[tile_key])
                        tile_cutout_info[tile_key] = tile_bands

                        # Update per-band cutout stats from the result
                        for band, band_stats in result.band_stats.items():
                            run_stats.record_cutout_result(
                                band,
                                succeeded=band_stats.get('succeeded', 0),
                                skipped=band_stats.get('skipped', 0),
                                failed=band_stats.get('failed', 0),
                            )

                        # Aggregate per-object band info
                        cutout_success_map.update(result.object_bands)

                        logger.debug(
                            f'✅ Cutouts for tile {tile_key}: Processed {n_objects_processed} objects'
                        )
                    else:
                        logger.warning(
                            f' Tile {tile_key}: created 0 cutouts (objects outside bounds?)'
                        )

            else:
                logger.debug('No cutout jobs were submitted.')
                cutout_executor.shutdown(wait=False)

        cleanup_temp_files(download_dir)


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

    last_exception: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = session.get(header_url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.Timeout as e:
            last_exception = e
            logger.warning(f'Timeout on attempt {attempt + 1}/{retries + 1} for {url}')
        except requests.ConnectionError as e:
            last_exception = e
            logger.warning(f'Connection error on attempt {attempt + 1}/{retries + 1} for {url}')
        except requests.HTTPError as e:
            last_exception = e
            logger.warning(
                f'HTTP error {e.response.status_code} on attempt {attempt + 1}/{retries + 1} for {url}'
            )
            # Don't retry on 4xx client errors (except 429 rate limit)
            if e.response.status_code < 500 and e.response.status_code != 429:
                break
        except requests.RequestException as e:
            last_exception = e
            logger.warning(f'Request error on attempt {attempt + 1}/{retries + 1} for {url}: {e}')

        if attempt < retries:
            sleep_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
            time.sleep(sleep_time)

    logger.error(f'Failed to fetch header after {retries + 1} attempts: {url}')
    if last_exception:
        logger.debug(f'Last exception: {last_exception}')
    return None


def report_download_summary(
    download_stats: dict[str, int],
    cutout_stats: dict[str, int],
    cutouts_enabled: bool,
    cutout_mode: str,
) -> None:
    """
    Report a global summary of the download and cutout process.

    Args:
        download_stats: Dictionary with download counts (downloaded, skipped, failed, rejected)
        cutout_stats: Dictionary with cutout counts (new, skipped, failed)
        cutouts_enabled: Whether to display the cutout summary section
        cutout_mode: Mode of cutout operation ('disabled', 'direct_only', 'after_download')
    """
    # Width for the label column
    W = 32

    logger.info('=' * 45)
    logger.info('RUN STATISTICS')

    # --- DOWNLOADS ---
    if not cutout_mode == 'direct_only':
        logger.info('-' * 45)
        logger.info('⬇️  TILE DOWNLOADS:')
        logger.info(f'   ✨ {"New:":<{W}} {download_stats.get("downloaded", 0)}')
        logger.info(f'   ⏭️  {"Skipped (already on disk):":<{W}} {download_stats.get("skipped", 0)}')

        rejected_downloads = download_stats.get('rejected', 0)
        if rejected_downloads > 0:
            logger.info(
                f'   ⚠️ {"Rejected:":<{W}} {rejected_downloads} (Files skipped due to missing data)'
            )

        logger.info(f'   ❌ {"Failed:":<{W}} {download_stats.get("failed", 0)}')

    # --- CUTOUTS ---
    if cutouts_enabled:
        logger.info('-' * 45)
        logger.info('✂️  CUTOUTS:')
        logger.info(f'   ✨ {"New:":<{W}} {cutout_stats.get("new", 0)}')
        logger.info(f'   ⏭️  {"Skipped (already done):":<{W}} {cutout_stats.get("skipped", 0)}')
        rejected_cutouts = cutout_stats.get('rejected', 0)
        if rejected_cutouts > 0:
            logger.info(
                f'   ⚠️ {"Rejected:":<{W}} {rejected_cutouts} (Objects skipped due to missing data)'
            )
        logger.info(f'   ❌ {"Failed:":<{W}} {cutout_stats.get("failed", 0)}')

    logger.info('=' * 45)
