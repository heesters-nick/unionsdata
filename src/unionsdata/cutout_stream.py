from __future__ import annotations

import io
import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from typing import cast

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.io.fits import ImageHDU, PrimaryHDU
from astropy.utils.exceptions import AstropyWarning
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from unionsdata.config import BandDict
from unionsdata.cutouts import (
    analyze_existing_file,
    match_objects_by_coords,
    merge_bands_into_h5,
    write_to_h5,
)
from unionsdata.download import make_session, tile_band_specs
from unionsdata.stats import RunStatistics
from unionsdata.utils import get_wavelength_order

logger = logging.getLogger(__name__)


@dataclass
class CutoutJob:
    """A single cutout download job."""

    object_id: str
    tile: tuple[int, int]
    band: str
    ra: float
    dec: float
    cutout_query: str  # e.g., "?SUB=[0][100:612,200:712]"
    pad_x: int = 0
    pad_y: int = 0


@dataclass
class StreamingStats:
    """Thread-safe download statistics with per-band tracking."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    _lock: Lock = field(default_factory=Lock)
    # Per-band tracking
    band_succeeded: dict[str, int] = field(default_factory=dict)
    band_failed: dict[str, int] = field(default_factory=dict)

    def record_success(self, band: str) -> None:
        with self._lock:
            self.succeeded += 1
            self.band_succeeded[band] = self.band_succeeded.get(band, 0) + 1

    def record_failure(self, band: str) -> None:
        with self._lock:
            self.failed += 1
            self.band_failed[band] = self.band_failed.get(band, 0) + 1


def check_cutout_availability(catalog: pd.DataFrame, bands: list[str]) -> int:
    """
    Calculate and report per-band availability of cutouts on VOSpace.
    """
    has_availability = 'bands' in catalog.columns
    n_objects = len(catalog)
    total_requested = n_objects * len(bands)

    if not has_availability:
        logger.warning(
            "Catalog missing 'bands' column; cannot verify VOSpace availability. Attempting all downloads."
        )
        return total_requested

    total_available_cutouts = 0

    for band in bands:
        # Count how many objects have this specific band available
        n_band_available = int(catalog['bands'].apply(lambda x, b=band: b in x).sum())
        total_available_cutouts += n_band_available

        n_missing = n_objects - n_band_available

        if n_missing > 0:
            # Use warning for missing data to make it visible
            logger.warning(
                f'  {band:<10} {n_band_available}/{n_objects} available (❌ {n_missing} missing)'
            )
        else:
            logger.info(f'  {band:<10} {n_band_available}/{n_objects} available (✅ 100%)')

    logger.info(f'Total: {total_available_cutouts} downloads / {total_requested} requested')

    if total_available_cutouts == 0:
        logger.error('No data available for ANY of the requested bands/objects.')

    return total_available_cutouts


def build_cutout_queries(
    catalog: pd.DataFrame,
    cutout_size: int,
    band_dict: dict[str, BandDict],
    bands: list[str],
    image_limit: int = 10000,
) -> pd.DataFrame:
    """
    Add cutout query strings and padding info to catalog.

    Generates the ?SUB=[ext][x1:x2,y1:y2] query parameter for each object/band
    combination, handling edge cases where objects are near image boundaries.

    Args:
        catalog: DataFrame with 'x', 'y' pixel coordinates (1-based)
        cutout_size: Square cutout size in pixels
        band_dict: Band configuration (needed for fits_ext per band)
        bands: List of requested bands
        image_limit: Image dimensions (default 10000x10000)

    Returns:
        DataFrame with added columns:
            - 'cutout_query_{band}': Query string per band, or None if invalid
            - 'pad_x': Left padding needed for edge objects
            - 'pad_y': Bottom padding needed for edge objects
    """
    catalog = catalog.copy()
    half = cutout_size // 2

    xs = np.round(catalog['x'].to_numpy(dtype=np.float32)).astype(np.int32)
    ys = np.round(catalog['y'].to_numpy(dtype=np.float32)).astype(np.int32)

    # Raw boundaries (may extend outside image)
    x1_raw = xs - half
    x2_raw = xs + half - 1
    y1_raw = ys - half
    y2_raw = ys + half - 1

    # Clipped to valid range [1, image_limit]
    x1 = np.clip(x1_raw, 1, image_limit)
    x2 = np.clip(x2_raw, 1, image_limit)
    y1 = np.clip(y1_raw, 1, image_limit)
    y2 = np.clip(y2_raw, 1, image_limit)

    # Padding for edge objects
    catalog['pad_x'] = np.clip(x1 - x1_raw, 0, None).astype(np.int32)
    catalog['pad_y'] = np.clip(y1 - y1_raw, 0, None).astype(np.int32)

    # Geometrically valid if cutout overlaps with image at all
    geo_valid = (x1_raw < image_limit) & (x2_raw > 0) & (y1_raw < image_limit) & (y2_raw > 0)

    # Build query string for each band (different fits_ext)
    for band in bands:
        ext = band_dict[band]['fits_ext']
        needs_band = np.array([band in instruction for instruction in catalog['bands_to_cutout']])
        final_mask = geo_valid & needs_band

        # Initialize with None for invalids
        queries = np.full(len(catalog), None, dtype=object)

        if final_mask.any():
            query_strings = (
                '?SUB=['
                + str(ext)
                + ']['
                + x1.astype(str)
                + ':'
                + x2.astype(str)
                + ','
                + y1.astype(str)
                + ':'
                + y2.astype(str)
                + ']'
            )

            # Fill valid entries
            queries[final_mask] = query_strings[final_mask]

        catalog[f'cutout_query_{band}'] = queries

    n_invalid = (~geo_valid).sum()
    if n_invalid > 0:
        logger.warning(f'{n_invalid} objects outside image bounds will be skipped')

    return catalog


def parse_cutout_bytes(
    data: bytes,
    pad_x: int,
    pad_y: int,
    cutout_size: int,
    fits_ext: int,
) -> NDArray[np.float32]:
    """
    Parse FITS bytes into a zero-padded array of consistent size.

    For edge objects, the server returns a smaller cutout. This function
    places that data at the correct offset in a full-sized output array.

    Args:
        data: Raw FITS bytes from HTTP response
        pad_x: X offset for placement (left padding)
        pad_y: Y offset for placement (bottom padding)
        cutout_size: Target output size
        fits_ext: FITS extension containing image data

    Returns:
        Array of shape (cutout_size, cutout_size)
    """
    if len(data) <= 2880:
        raise RuntimeError('Server returned only FITS header (truncated data)')

    output = np.zeros((cutout_size, cutout_size), dtype=np.float32)

    try:
        # Silence Astropy warnings about truncation/checksums to avoid double logging
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
        with fits.open(io.BytesIO(data), memmap=False) as hdul:
            if fits_ext >= len(hdul):
                fits_ext = 0
            hdu = cast(ImageHDU | PrimaryHDU, hdul[fits_ext])

            if hdu.data is None:
                raise ValueError(f'No data in extension {fits_ext}')

            img = hdu.data.astype(np.float32)
            h, w = img.shape

            # Calculate valid placement region
            y_end = min(pad_y + h, cutout_size)
            x_end = min(pad_x + w, cutout_size)
            h_copy = y_end - pad_y
            w_copy = x_end - pad_x

            if h_copy > 0 and w_copy > 0:
                output[pad_y:y_end, pad_x:x_end] = img[:h_copy, :w_copy]

    except (OSError, TypeError, ValueError) as e:
        # OSError: Corrupt FITS header or truncated file
        # TypeError: Malformed data structure or unexpected type in FITS parsing
        # ValueError: Our explicit check for hdu.data is None
        raise RuntimeError(f'FITS parsing failed (corrupt download?): {e}') from e

    return output


def download_worker(
    job_queue: Queue[CutoutJob | None],
    results: dict[str, dict[str, NDArray[np.float32]]],
    results_lock: Lock,
    band_dict: dict[str, BandDict],
    download_dir: Path,
    cutout_size: int,
    cert_path: Path,
    max_retries: int,
    shutdown: Event,
    stats: StreamingStats,
) -> None:
    """
    Worker thread that downloads cutouts from the job queue.

    Uses HTTP with session reuse for efficiency. Results are stored
    in a shared dict keyed by object_id -> band -> array.

    Args:
        job_queue: Queue of CutoutJob or None (sentinel)
        results: Shared dict for storing results
        results_lock: Lock for synchronizing access to results
        band_dict: Band configuration dictionary
        download_dir: Base directory for downloads
        cutout_size: Size of square cutouts in pixels
        cert_path: Path to SSL certificate for requests
        max_retries: Max retries for failed downloads
        shutdown: Event to signal early shutdown
        stats: Shared stats object for recording successes/failures

    Returns:
        None
    """
    session = make_session(cert_path)

    try:
        while not shutdown.is_set():
            try:
                job = job_queue.get(timeout=0.5)
            except Empty:
                continue

            if job is None:  # Shutdown sentinel
                job_queue.task_done()
                break

            try:
                # Get base URL from existing tile_band_specs
                specs = tile_band_specs(
                    job.tile, band_dict, job.band, download_dir, cutout_mode='direct_only'
                )
                url = specs['http_url'] + job.cutout_query

                # Download with retries
                cutout_data = fetch_cutout(
                    url=url,
                    session=session,
                    fits_ext=specs['fits_ext'],
                    pad_x=job.pad_x,
                    pad_y=job.pad_y,
                    cutout_size=cutout_size,
                    max_retries=max_retries,
                )

                # Store result
                with results_lock:
                    if job.object_id not in results:
                        results[job.object_id] = {}
                    results[job.object_id][job.band] = cutout_data

                stats.record_success(job.band)

            except Exception as e:
                err_msg = str(e)
                if 'Server returned only FITS header' in err_msg:
                    logger.warning(
                        f'{job.object_id} ({job.band}) failed: Direct cutout not supported by server (received header only).'
                    )
                else:
                    logger.error(f'{job.object_id} ({job.band}) failed: {e}')
                stats.record_failure(job.band)

            finally:
                job_queue.task_done()

    finally:
        session.close()


def fetch_cutout(
    url: str,
    session: requests.Session,
    fits_ext: int,
    pad_x: int,
    pad_y: int,
    cutout_size: int,
    max_retries: int,
) -> NDArray[np.float32]:
    """Fetch a single cutout with retry logic."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return parse_cutout_bytes(resp.content, pad_x, pad_y, cutout_size, fits_ext)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise  # Don't retry missing files
            last_error = e
        except Exception as e:
            last_error = e

        # Exponential backoff before retrying
        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    raise RuntimeError(f'Failed after {max_retries} attempts: {last_error}')


def stream_direct_cutouts(
    catalog: pd.DataFrame,
    bands: list[str],
    band_dict: dict[str, BandDict],
    download_dir: Path,
    cutout_subdir: str,
    cutout_size: int,
    cert_path: Path,
    max_retries: int,
    n_workers: int,
    run_stats: RunStatistics,
    cutout_success_map: dict[str, set[str]],
    batch_size: int = 500,
) -> None:
    """
    Stream cutouts directly from VOSpace.

    Features:
    - Sorts by tile to maximize I/O efficiency.
    - Batches processing to keep memory usage low.
    - Serializes HDF5 writes to ensure thread safety.
    - Creates worker threads once for efficiency.

    Args:
        catalog: DataFrame with object info and pixel coordinates
        bands: List of band names to download
        band_dict: Band configuration dictionary
        download_dir: Base directory for downloads
        cutout_subdir: Subdirectory for cutout HDF5 files
        cutout_size: Size of square cutouts in pixels
        cert_path: Path to SSL certificate for requests
        max_retries: Max retries for failed downloads
        n_workers: Number of parallel download threads
        run_stats: RunStatistics object for per-band tracking
        cutout_success_map: Dict to update with successful cutout bands per object
        batch_size: Number of objects to process per batch

    Returns:
        None
    """
    if catalog.empty:
        return

    # Sort catalog by tile for efficiency
    catalog = catalog.sort_values(by='tile').reset_index(drop=True)

    bands = get_wavelength_order(bands, band_dict)
    catalog = build_cutout_queries(catalog, cutout_size, band_dict, bands)

    # Filter invalid
    query_cols = [f'cutout_query_{b}' for b in bands if f'cutout_query_{b}' in catalog.columns]

    if query_cols:
        # Keep row if at least one of the required bands has a valid query
        valid_mask = catalog[query_cols].notna().any(axis=1)
        catalog = catalog[valid_mask].reset_index(drop=True)
    else:
        # Should not happen if bands list is valid
        return

    if catalog.empty:
        logger.warning('No valid cutouts after filtering')
        return

    # Shared state for all batches
    job_queue: Queue[CutoutJob | None] = Queue()
    results: dict[str, dict[str, NDArray[np.float32]]] = {}
    results_lock = Lock()
    shutdown = Event()
    stats = StreamingStats()
    total_tasks = int(catalog[query_cols].notna().sum().sum())
    n_workers = min(n_workers, max(1, total_tasks))

    # Create workers once and reuse across all batches
    workers = [
        threading.Thread(
            target=download_worker,
            args=(
                job_queue,
                results,
                results_lock,
                band_dict,
                download_dir,
                cutout_size,
                cert_path,
                max_retries,
                shutdown,
                stats,
            ),
            daemon=True,
        )
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()

    jobs_queued = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn('[bold blue]{task.description}'),
            BarColumn(bar_width=None),
            TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
            '•',
            MofNCompleteColumn(),
            '•',
            TimeElapsedColumn(),
            '•',
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task('Streaming cutouts', total=total_tasks)

            for batch_start in range(0, len(catalog), batch_size):
                if shutdown.is_set():
                    break

                chunk = catalog.iloc[batch_start : batch_start + batch_size].copy()
                if chunk.empty:
                    continue

                # Queue jobs for this batch
                batch_job_count = 0
                for _, row in chunk.iterrows():
                    tile_parts = str(row['tile']).split('_')
                    tile = (int(tile_parts[0]), int(tile_parts[1]))

                    for band in bands:
                        col_name = f'cutout_query_{band}'

                        if col_name in row and pd.notna(row[col_name]):
                            job_queue.put(
                                CutoutJob(
                                    object_id=str(row['ID']),
                                    tile=tile,
                                    band=band,
                                    ra=float(row['ra']),
                                    dec=float(row['dec']),
                                    cutout_query=row[col_name],
                                    pad_x=int(row['pad_x']),
                                    pad_y=int(row['pad_y']),
                                )
                            )
                            batch_job_count += 1

                jobs_queued += batch_job_count

                # Wait for this batch to complete
                while (stats.succeeded + stats.failed) < jobs_queued:
                    progress.update(task, completed=stats.succeeded + stats.failed)
                    time.sleep(0.1)
                    if shutdown.is_set():
                        break

                progress.update(task, completed=stats.succeeded + stats.failed)

                # Save results for tiles in this batch
                for tile_key in chunk['tile'].unique():
                    tile_chunk = chunk[chunk['tile'] == tile_key]

                    save_results_by_tile(
                        results=results,
                        catalog=tile_chunk,
                        bands=bands,
                        band_dict=band_dict,
                        download_dir=download_dir,
                        cutout_subdir=cutout_subdir,
                        cutout_size=cutout_size,
                    )

                # Update success map
                for obj_id_str, bands_data in results.items():
                    if obj_id_str not in cutout_success_map:
                        cutout_success_map[obj_id_str] = set()
                    cutout_success_map[obj_id_str].update(bands_data.keys())

                # Clear results for next batch
                with results_lock:
                    results.clear()

    except KeyboardInterrupt:
        logger.warning('Interrupted, shutting down workers...')
        shutdown.set()

    # Shutdown workers
    for _ in range(n_workers):
        job_queue.put(None)

    for w in workers:
        w.join(timeout=5.0)

    # Update run_stats with per-band results
    for band in bands:
        succeeded = stats.band_succeeded.get(band, 0)
        failed = stats.band_failed.get(band, 0)
        run_stats.record_cutout_result(band, succeeded=succeeded, failed=failed)


def save_results_by_tile(
    results: dict[str, dict[str, NDArray[np.float32]]],
    catalog: pd.DataFrame,
    bands: list[str],
    band_dict: dict[str, BandDict],
    download_dir: Path,
    cutout_subdir: str,
    cutout_size: int,
) -> None:
    """
    Save cutouts to HDF5. Handles creating new files or updating existing ones.

    Args:
        results: Dict of cutout results from downloads
        catalog: DataFrame of objects for this tile
        bands: List of band names
        band_dict: Band configuration dictionary
        download_dir: Base directory for downloads
        cutout_subdir: Subdirectory for cutout HDF5 files
        cutout_size: Size of square cutouts in pixels

    Returns:
        None
    """
    # Filter catalog to objects we actually have results for
    valid_mask = catalog['ID'].astype(str).isin(results.keys())
    if not valid_mask.any():
        return

    tile_catalog = catalog[valid_mask].reset_index(drop=True)
    tile_key = tile_catalog['tile'].iloc[0]

    output_dir = download_dir / tile_key / cutout_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{tile_key}_cutouts_{cutout_size}.h5'

    # Build 4D array (n_objects, bands, cutout_size, cutout_size) from results dict
    def get_data_array(df: pd.DataFrame, target_bands: list[str]) -> NDArray[np.float32]:
        """Helper to build data array from results dict.
        Args:
            df: DataFrame of objects to include
            target_bands: Bands to include in output array

        Returns:
            4D array of shape (n_objects, n_bands, cutout_size, cutout_size)
        """
        arr = np.zeros((len(df), len(target_bands), cutout_size, cutout_size), dtype=np.float32)
        for i, row in enumerate(df.itertuples()):
            obj_res = results.get(str(row.ID), {})
            for j, b in enumerate(target_bands):
                if b in obj_res:
                    arr[i, j] = obj_res[b]
        return arr

    try:
        existing_info = analyze_existing_file(output_path)

        # Case 1: new file
        if not existing_info.exists:
            data = get_data_array(tile_catalog, bands)
            write_to_h5(output_path, data, tile_catalog, bands, tile_key, band_dict)
            return

        # Case 2: existing file (update/append)
        # Determine new vs existing bands
        new_bands = [b for b in bands if b not in existing_info.bands]

        # Satisfy type checker
        assert existing_info.object_coords is not None

        # Match coordinates to find existing objects
        cat_coords = np.column_stack([tile_catalog['ra'], tile_catalog['dec']]).astype(np.float32)
        is_match, match_idx = match_objects_by_coords(cat_coords, existing_info.object_coords)

        n_new = int((~is_match).sum())

        # A. Update matched objects (add new bands if available)
        if is_match.any():
            if new_bands:
                matched_df = tile_catalog[is_match].reset_index(drop=True)
                new_data = get_data_array(matched_df, new_bands)

                merge_bands_into_h5(
                    output_path, new_data, new_bands, existing_info, match_idx[is_match], band_dict
                )

        # B. Append completely new objects (write all bands)
        if n_new > 0:
            new_df = tile_catalog[~is_match].reset_index(drop=True)
            data = get_data_array(new_df, bands)
            write_to_h5(output_path, data, new_df, bands, tile_key, band_dict)

    except (OSError, ValueError, RuntimeError) as e:
        # Catch locking errors, disk full, or corrupt HDF5 files
        # We log and return 0 so the stream doesn't crash
        logger.error(f'Failed to save cutouts to {output_path}: {e}')
        return
