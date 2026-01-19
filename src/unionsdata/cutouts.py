import logging
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from numpy.typing import NDArray

from unionsdata.config import BandCfg
from unionsdata.logging_setup import setup_logger
from unionsdata.stats import RunStatistics
from unionsdata.utils import get_dataset, get_wavelength_order, open_fits, tile_str

logger = logging.getLogger(__name__)

# Default batch size for processing cutouts
DEFAULT_BATCH_SIZE = 1000


# ========== Helper Functions ==========


def worker_log_init(log_dir: Path, name: str, level: int) -> None:
    """Initialize logging in worker process."""
    setup_logger(log_dir, name, level, force=True)


@dataclass
class CutoutResult:
    """Result of cutout creation for a tile."""

    object_bands: dict[str, set[str]]  # object_id -> list of cutout bands
    band_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def add_band_stats(
        self, band: str, succeeded: int = 0, skipped: int = 0, failed: int = 0
    ) -> None:
        """Add statistics for a specific band."""
        if band not in self.band_stats:
            self.band_stats[band] = {'succeeded': 0, 'skipped': 0, 'failed': 0}
        self.band_stats[band]['succeeded'] += succeeded
        self.band_stats[band]['skipped'] += skipped
        self.band_stats[band]['failed'] += failed


@dataclass
class ExistingFileInfo:
    """Information about an existing HDF5 cutout file."""

    exists: bool
    bands: list[str]
    n_objects: int
    object_coords: NDArray[np.float32] | None  # shape (n, 2) for ra, dec
    cutout_shape: tuple[int, ...] | None  # (n_objects, n_bands, h, w)


@dataclass
class CutoutPlan:
    """Plan describing what cutout operations need to be performed."""

    create_new_file: bool
    bands_to_add_to_existing: list[str]
    matched_catalog: pd.DataFrame  # Objects matching existing HDF5, with 'h5_index' column
    new_objects_catalog: pd.DataFrame  # Objects not in existing HDF5
    bands_for_new_objects: list[str]  # Full band list for new objects
    existing_info: ExistingFileInfo


class CutoutResultBuilder:
    """Accumulates cutout results during processing."""

    def __init__(self, bands: list[str]) -> None:
        """Initialize tracking structures for the given bands.

        Args:
            bands: List of band names to track
        """
        self._object_bands: dict[str, set[str]] = {}
        self._band_stats: dict[str, dict[str, int]] = {
            band: {'succeeded': 0, 'skipped': 0, 'failed': 0} for band in bands
        }
        self._bands = bands

    def record_object_success(self, object_id: str, bands: set[str]) -> None:
        """Record successful cutout creation for an object.

        Args:
            object_id: The object identifier
            bands: Set of bands successfully created for this object
        """
        if object_id not in self._object_bands:
            self._object_bands[object_id] = set()
        self._object_bands[object_id].update(bands)

    def record_band_stats(
        self, band: str, succeeded: int = 0, skipped: int = 0, failed: int = 0
    ) -> None:
        """Increment per-band counters.

        Args:
            band: Band name
            succeeded: Number of successful cutouts to add
            skipped: Number of skipped cutouts to add
            failed: Number of failed cutouts to add
        """
        if band not in self._band_stats:
            self._band_stats[band] = {'succeeded': 0, 'skipped': 0, 'failed': 0}
        self._band_stats[band]['succeeded'] += succeeded
        self._band_stats[band]['skipped'] += skipped
        self._band_stats[band]['failed'] += failed

    def merge(self, other: 'CutoutResultBuilder') -> None:
        """Combine results from another builder into this one.

        Args:
            other: Another CutoutResultBuilder to merge in
        """
        for obj_id, bands in other._object_bands.items():
            self.record_object_success(obj_id, bands)
        for band, stats in other._band_stats.items():
            self.record_band_stats(
                band,
                succeeded=stats['succeeded'],
                skipped=stats['skipped'],
                failed=stats['failed'],
            )

    def build(self) -> CutoutResult:
        """Finalize and return immutable result.

        Returns:
            CutoutResult containing accumulated data
        """
        return CutoutResult(
            object_bands=dict(self._object_bands),
            band_stats=dict(self._band_stats),
        )


def analyze_existing_file(output_path: Path) -> ExistingFileInfo:
    """
    Analyze an existing HDF5 file to understand its current state.

    Args:
        output_path: Path to the HDF5 file

    Returns:
        ExistingFileInfo with bands, object count, and coordinates
    """
    if not output_path.exists():
        return ExistingFileInfo(
            exists=False,
            bands=[],
            n_objects=0,
            object_coords=None,
            cutout_shape=None,
        )

    try:
        with h5py.File(str(output_path), 'r') as f:
            # Read existing bands
            bands_raw = np.array(get_dataset(f, 'bands'))
            bands = [b.decode('utf-8') if isinstance(b, bytes) else b for b in bands_raw]

            # Read coordinates for matching
            ra = np.array(get_dataset(f, 'ra'), dtype=np.float32)
            dec = np.array(get_dataset(f, 'dec'), dtype=np.float32)
            coords = np.column_stack([ra, dec])

            # Get cutout shape
            cutout_shape = get_dataset(f, 'cutouts').shape

            return ExistingFileInfo(
                exists=True,
                bands=bands,
                n_objects=len(ra),
                object_coords=coords,
                cutout_shape=cutout_shape,
            )

    except Exception as e:
        logger.warning(f'Error analyzing existing file {output_path}: {e}')
        return ExistingFileInfo(
            exists=False,
            bands=[],
            n_objects=0,
            object_coords=None,
            cutout_shape=None,
        )


def match_objects_by_coords(
    new_coords: NDArray[np.float32],
    existing_coords: NDArray[np.float32],
    threshold_arcsec: float = 5.0,
) -> tuple[NDArray[np.bool_], NDArray[np.intp]]:
    """
    Match new coordinates against existing coordinates using on-sky distance.

    Args:
        new_coords: Array of (ra, dec) for new objects, shape (n_new, 2)
        existing_coords: Array of (ra, dec) for existing objects, shape (n_existing, 2)
        threshold_arcsec: Match threshold in arcseconds

    Returns:
        Tuple of:
            - is_match: Boolean array (n_new,) indicating which new objects have matches
            - match_indices: Integer array (n_new,) with index of match in existing
              (only valid where is_match is True)
    """
    existing_skycoord = SkyCoord(ra=existing_coords[:, 0], dec=existing_coords[:, 1], unit='deg')
    new_skycoord = SkyCoord(ra=new_coords[:, 0], dec=new_coords[:, 1], unit='deg')

    # Find nearest existing object for each new object
    idx, d2d, _ = new_skycoord.match_to_catalog_sky(existing_skycoord)

    threshold = threshold_arcsec * Unit('arcsec')
    is_match = d2d < threshold

    return np.array(is_match), np.array(idx, dtype=np.intp)


def match_catalog_to_existing(
    catalog: pd.DataFrame,
    existing_info: ExistingFileInfo,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Determine which catalog objects already exist in the HDF5 file.

    Args:
        catalog: Input catalog with 'ra', 'dec' columns
        existing_info: Information about existing HDF5 file

    Returns:
        Tuple of:
            - matched: DataFrame of rows that matched, with added 'h5_index' column
            - unmatched: DataFrame of rows that didn't match
    """
    if not existing_info.exists or existing_info.object_coords is None:
        # No existing file or no objects in it
        empty_matched = catalog.iloc[:0].copy()
        empty_matched['h5_index'] = pd.Series(dtype=np.intp)
        return empty_matched, catalog.copy()

    # Build coordinate arrays from catalog
    catalog_coords = np.column_stack([catalog['ra'].to_numpy(), catalog['dec'].to_numpy()]).astype(
        np.float32
    )

    # Match against existing
    is_match, match_indices = match_objects_by_coords(catalog_coords, existing_info.object_coords)

    # Split catalog
    matched = catalog[is_match].copy()
    matched['h5_index'] = match_indices[is_match]

    unmatched = catalog[~is_match].copy()

    return matched.reset_index(drop=True), unmatched.reset_index(drop=True)


def plan_cutout_operations(
    catalog: pd.DataFrame,
    requested_bands: list[str],
    existing_info: ExistingFileInfo,
    all_band_dictionary: dict[str, BandCfg],
) -> CutoutPlan:
    """Analyze the situation and produce a plan for cutout operations.

    This function makes no I/O calls - it only analyzes and computes.

    Args:
        catalog: Full input catalog for this tile
        requested_bands: Bands user wants this run
        existing_info: State of existing HDF5 file
        all_band_dictionary: For wavelength ordering

    Returns:
        CutoutPlan describing what operations need to be performed
    """
    # Case 1: No existing file
    if not existing_info.exists:
        empty_matched = catalog.iloc[:0].copy()
        empty_matched['h5_index'] = pd.Series(dtype=np.intp)

        return CutoutPlan(
            create_new_file=True,
            bands_to_add_to_existing=[],
            matched_catalog=empty_matched,
            new_objects_catalog=catalog.copy(),
            bands_for_new_objects=requested_bands,
            existing_info=existing_info,
        )

    # Case 2 & 3: File exists
    matched, unmatched = match_catalog_to_existing(catalog, existing_info)

    # Determine which bands are new
    existing_bands_set = set(existing_info.bands)
    bands_to_add = [b for b in requested_bands if b not in existing_bands_set]

    # New objects need all bands (existing + requested) for uniform coverage
    all_bands = list(existing_bands_set | set(requested_bands))
    bands_for_new_objects = get_wavelength_order(all_bands, all_band_dictionary)

    return CutoutPlan(
        create_new_file=False,
        bands_to_add_to_existing=bands_to_add,
        matched_catalog=matched,
        new_objects_catalog=unmatched,
        bands_for_new_objects=bands_for_new_objects,
        existing_info=existing_info,
    )


# ========= Core Functions ==========


def read_band_data(
    tile_dir: Path,
    tile: tuple[int, int],
    bands: list[str],
    in_dict: dict[str, BandCfg],
) -> tuple[NDArray[np.float32], list[str]]:
    """
    Read image data for specified bands and tile.

    Args:
        tile_dir: Directory for the specific tile
        tile: Tuple of (tile_x, tile_y)
        bands: List of band names to read
        in_dict: Dictionary with band-specific file info

    Returns:
        Tuple of (multiband data array, list of successfully loaded bands).
        The returned array only contains channels for successfully loaded bands,
        with indices matching the returned band list.
    """
    loaded_data: list[NDArray[np.float32]] = []
    loaded_bands: list[str] = []

    for band in bands:
        zfill = in_dict[band].zfill
        file_prefix = in_dict[band].name
        delimiter = in_dict[band].delimiter
        suffix = in_dict[band].suffix
        fits_ext = in_dict[band].fits_ext

        base_dir = tile_dir / band
        num1, num2 = str(tile[0]).zfill(zfill), str(tile[1]).zfill(zfill)
        filename = f'{file_prefix}{delimiter}{num1}{delimiter}{num2}{suffix}'
        data_path = base_dir / filename

        try:
            data, _ = open_fits(file_path=data_path, fits_ext=fits_ext)
            loaded_data.append(data)
            loaded_bands.append(band)
            logger.debug(f'Loaded {band} from {data_path.name}')
        except Exception as e:
            logger.warning(f'Failed to load {band} from {data_path}: {e}')
            continue

    if len(loaded_bands) == 0:
        raise ValueError(f'No bands could be loaded for tile {tile}')

    # Stack into single array - indices now match loaded_bands
    multiband_data = np.stack(loaded_data, axis=0).astype(np.float32)

    return multiband_data, loaded_bands


def make_multiband_cutouts(
    multiband_data: NDArray[np.float32],
    tile_str: str,
    df: pd.DataFrame,
    cutout_size: int,
) -> NDArray[np.float32]:
    """
    Create cutouts from multi-band image data.

    Args:
        multiband_data: Image array with shape (n_bands, y, x)
        tile_str: Tile identifier for logging
        df: DataFrame with 'X' and 'Y' columns (pixel coordinates)
        cutout_size: Square cutout size in pixels

    Returns:
        Array of cutouts with shape (n_objects, n_bands, cutout_size, cutout_size)
    """
    # Convert DataFrame coordinates to integer pixel positions (need to convert from 1-based to 0-based)
    xs = np.round(df['x'].to_numpy(dtype=np.float32) - 1.0).astype(np.int32)
    ys = np.round(df['y'].to_numpy(dtype=np.float32) - 1.0).astype(np.int32)

    n_bands = multiband_data.shape[0]
    n_objects = len(xs)

    # Pre-allocate output: (n_objects, n_bands, cutout_size, cutout_size)
    cutouts = np.zeros((n_objects, n_bands, cutout_size, cutout_size), dtype=multiband_data.dtype)

    size_half = cutout_size // 2
    y_max, x_max = multiband_data.shape[1], multiband_data.shape[2]

    # Vectorized bounds calculation for efficiency
    y_starts = np.maximum(0, ys - size_half)
    y_ends = np.minimum(y_max, ys + (cutout_size - size_half))
    x_starts = np.maximum(0, xs - size_half)
    x_ends = np.minimum(x_max, xs + (cutout_size - size_half))

    # Identify valid objects (partially inside image)
    valid_mask = (y_starts < y_ends) & (x_starts < x_ends)

    if not np.all(valid_mask):
        logger.warning(f'Tile {tile_str}: {np.sum(~valid_mask)} objects outside image bounds')

    # Fill cutouts
    for i in np.where(valid_mask)[0]:
        y, x = ys[i], xs[i]
        ys_i, ye_i = y_starts[i], y_ends[i]
        xs_i, xe_i = x_starts[i], x_ends[i]

        # Calculate target positions in the cutout array
        cys = ys_i - y + size_half
        cye = ye_i - y + size_half
        cxs = xs_i - x + size_half
        cxe = xe_i - x + size_half

        cutouts[i, :, cys:cye, cxs:cxe] = multiband_data[:, ys_i:ye_i, xs_i:xe_i]

    return cutouts


def merge_bands_into_h5(
    output_path: Path,
    new_cutouts: NDArray[np.float32],
    new_bands: list[str],
    existing_info: ExistingFileInfo,
    match_indices: NDArray[np.intp],
    band_dictionary: dict[str, BandCfg],
) -> None:
    """
    Merge new band data into existing HDF5 file for matched objects.

    Inserts new band columns at the correct wavelength-ordered positions,
    shifting existing data as needed. Uses chunked processing to avoid
    loading the entire dataset into memory.

    Args:
        output_path: Path to existing HDF5 file
        new_cutouts: New cutout data, shape (n_matched, n_new_bands, h, w)
        new_bands: Names of the new bands being added
        existing_info: Information about the existing file
        match_indices: Indices mapping new cutouts to existing HDF5 rows
        band_dictionary: Full band config for wavelength ordering
    """
    # Compute merged band list in wavelength order
    all_bands = set(existing_info.bands) | set(new_bands)
    merged_bands = get_wavelength_order(list(all_bands), band_dictionary)

    logger.debug(
        f'Merging bands: existing={existing_info.bands}, new={new_bands}, merged={merged_bands}'
    )

    # Compute index mappings: band name -> position in merged array
    old_band_to_merged_idx = {b: merged_bands.index(b) for b in existing_info.bands}
    new_band_to_merged_idx = {b: merged_bands.index(b) for b in new_bands}

    n_merged_bands = len(merged_bands)
    assert existing_info.cutout_shape is not None
    n_total_objects, n_old_bands, h, w = existing_info.cutout_shape

    with h5py.File(str(output_path), 'a') as f:
        cutouts_dset = get_dataset(f, 'cutouts')

        # 1. Resize band dimension to accommodate new bands
        new_shape = (n_total_objects, n_merged_bands, h, w)
        cutouts_dset.resize(new_shape)

        # 2. Reorder existing data in chunks to avoid memory issues
        batch_size = 100  # Process 100 objects at a time

        logger.debug(f'Reordering existing data in chunks of {batch_size}...')

        for i in range(0, n_total_objects, batch_size):
            end_i = min(i + batch_size, n_total_objects)
            n_in_batch = end_i - i

            # Read ONLY the columns corresponding to the old bands
            # Shape: (batch, n_old_bands, h, w)
            chunk_old = cutouts_dset[i:end_i, :n_old_bands, :, :]

            # Create buffer for merged bands
            chunk_new = np.zeros((n_in_batch, n_merged_bands, h, w), dtype=np.float32)

            # Place old bands into their new correct positions
            for old_idx, band_name in enumerate(existing_info.bands):
                merged_idx = old_band_to_merged_idx[band_name]
                chunk_new[:, merged_idx, :, :] = chunk_old[:, old_idx, :, :]

            # Write the merged chunk back to the file
            cutouts_dset[i:end_i] = chunk_new

        # 3. Insert new band data for matched objects
        logger.debug(f'Inserting {len(new_bands)} new bands for matched objects...')

        if len(match_indices) > 0:
            sort_order = np.argsort(match_indices)
            sorted_indices = match_indices[sort_order]

            for new_local_idx, new_band in enumerate(new_bands):
                merged_idx = new_band_to_merged_idx[new_band]
                data_to_write = new_cutouts[sort_order, new_local_idx, :, :]
                cutouts_dset[sorted_indices, merged_idx, :, :] = data_to_write

        # 4. Update bands metadata
        del f['bands']
        f.create_dataset('bands', data=np.array(merged_bands, dtype='S'))

    logger.debug(f'Merged {len(new_bands)} new bands into {output_path}')


# ========= Batch processing ==========


def process_catalog_in_batches(
    tile: tuple[int, int],
    tile_dir: Path,
    catalog: pd.DataFrame,
    bands: list[str],
    band_dictionary: dict[str, BandCfg],
    output_path: Path,
    cutout_size: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[set[str], list[str], int]:
    """Load tile FITS data and create cutouts for a catalog of objects.

    Processes objects in batches to manage memory, writing to HDF5 incrementally.

    Args:
        tile: Tile coordinates (x, y)
        tile_dir: Directory containing band subdirectories with FITS files
        catalog: Objects to process (must have x, y, ID, ra, dec columns)
        bands: Bands to create cutouts for
        band_dictionary: Band configuration dictionary
        output_path: HDF5 file to write to
        cutout_size: Size of cutouts in pixels
        batch_size: Number of objects per batch

    Returns:
        Tuple of (set of object IDs successfully processed, list of bands actually loaded, failed count)
    """
    if catalog.empty:
        return set(), [], 0

    tile_key = tile_str(tile)

    # Load FITS data for all requested bands
    try:
        multiband_data, loaded_bands = read_band_data(
            tile_dir=tile_dir,
            tile=tile,
            bands=bands,
            in_dict=band_dictionary,
        )
    except ValueError as e:
        logger.warning(f'Tile {tile_key}: {e}')
        return set(), [], 0

    if not loaded_bands:
        logger.warning(f'Tile {tile_key}: No bands could be loaded')
        return set(), [], 0

    # Process in batches
    processed_ids: set[str] = set()
    failed_count: int = 0

    for start_idx in range(0, len(catalog), batch_size):
        end_idx = min(start_idx + batch_size, len(catalog))
        batch_df = catalog.iloc[start_idx:end_idx].reset_index(drop=True)

        try:
            batch_cutouts = make_multiband_cutouts(
                multiband_data=multiband_data,
                tile_str=tile_key,
                df=batch_df,
                cutout_size=cutout_size,
            )

            write_to_h5(
                output_path=output_path,
                cutouts=batch_cutouts,
                catalog=batch_df,
                bands=loaded_bands,
                tile_key=tile_key,
                band_dictionary=band_dictionary,
            )

            # Record successful IDs
            processed_ids.update(batch_df['ID'].astype(str).tolist())

        except Exception as e:
            logger.error(f'Tile {tile_key}: Batch {start_idx}-{end_idx} failed: {e}')
            failed_count += len(batch_df)
            continue

    return processed_ids, loaded_bands, failed_count


# ========== Case Handlers ==========


def handle_new_file(
    plan: CutoutPlan,
    tile: tuple[int, int],
    tile_dir: Path,
    output_path: Path,
    band_dictionary: dict[str, BandCfg],
    cutout_size: int,
    results: CutoutResultBuilder,
) -> None:
    """Handle Case 1: Create a new HDF5 file with cutouts for all objects.

    Args:
        plan: The cutout plan
        tile: Tile coordinates
        tile_dir: Directory containing FITS files
        output_path: Path for new HDF5 file
        band_dictionary: Band configuration
        cutout_size: Size of cutouts in pixels
        results: Builder to accumulate results
    """
    tile_key = tile_str(tile)
    catalog = plan.new_objects_catalog

    logger.debug(f'Tile {tile_key}: Creating new file with {len(catalog)} objects')

    processed_ids, loaded_bands, failed_count = process_catalog_in_batches(
        tile=tile,
        tile_dir=tile_dir,
        catalog=catalog,
        bands=plan.bands_for_new_objects,
        band_dictionary=band_dictionary,
        output_path=output_path,
        cutout_size=cutout_size,
    )

    # Record results
    loaded_bands_set = set(loaded_bands)
    for obj_id in processed_ids:
        results.record_object_success(obj_id, loaded_bands_set)

    # Update band stats
    n_processed = len(processed_ids)
    for band in loaded_bands:
        results.record_band_stats(band, succeeded=n_processed, failed=failed_count)


def handle_add_bands(
    plan: CutoutPlan,
    tile: tuple[int, int],
    tile_dir: Path,
    output_path: Path,
    band_dictionary: dict[str, BandCfg],
    cutout_size: int,
    results: CutoutResultBuilder,
) -> ExistingFileInfo:
    """Handle Case 2: Add new bands to existing objects in HDF5.

    Args:
        plan: The cutout plan
        tile: Tile coordinates
        tile_dir: Directory containing FITS files
        output_path: Path to existing HDF5 file
        band_dictionary: Band configuration
        cutout_size: Size of cutouts in pixels
        results: Builder to accumulate results

    Returns:
        Updated ExistingFileInfo after merge (bands have changed)
    """
    tile_key = tile_str(tile)
    matched_catalog = plan.matched_catalog
    new_bands = plan.bands_to_add_to_existing

    if not new_bands or matched_catalog.empty:
        return plan.existing_info

    n_matched = len(matched_catalog)
    logger.debug(f'Tile {tile_key}: Adding bands {new_bands} to {n_matched} existing objects')

    # Load FITS data for new bands only
    try:
        multiband_data, loaded_new_bands = read_band_data(
            tile_dir=tile_dir,
            tile=tile,
            bands=new_bands,
            in_dict=band_dictionary,
        )
    except ValueError as e:
        logger.warning(f'Tile {tile_key}: Failed to load new bands: {e}')
        return plan.existing_info

    if not loaded_new_bands:
        return plan.existing_info

    # Create cutouts for matched objects in new bands
    new_band_cutouts = make_multiband_cutouts(
        multiband_data=multiband_data,
        tile_str=tile_key,
        df=matched_catalog,
        cutout_size=cutout_size,
    )

    # Get HDF5 indices for matched objects
    matched_h5_indices = matched_catalog['h5_index'].to_numpy(dtype=np.intp)

    # Merge into HDF5
    merge_bands_into_h5(
        output_path=output_path,
        new_cutouts=new_band_cutouts,
        new_bands=loaded_new_bands,
        existing_info=plan.existing_info,
        match_indices=matched_h5_indices,
        band_dictionary=band_dictionary,
    )

    # Re-read existing_info after merge - bands have changed
    updated_info = analyze_existing_file(output_path)

    # Record results - matched objects now have all merged bands
    merged_bands_set = set(updated_info.bands)
    for obj_id in matched_catalog['ID'].astype(str):
        results.record_object_success(obj_id, merged_bands_set)

    # Update band stats for new bands
    for band in loaded_new_bands:
        results.record_band_stats(band, succeeded=n_matched)

    return updated_info


def handle_new_objects(
    plan: CutoutPlan,
    tile: tuple[int, int],
    tile_dir: Path,
    output_path: Path,
    band_dictionary: dict[str, BandCfg],
    cutout_size: int,
    results: CutoutResultBuilder,
) -> None:
    """Handle Case 3: Append new objects to existing HDF5.

    Args:
        plan: The cutout plan
        tile: Tile coordinates
        tile_dir: Directory containing FITS files
        output_path: Path to existing HDF5 file
        band_dictionary: Band configuration
        cutout_size: Size of cutouts in pixels
        results: Builder to accumulate results
    """
    tile_key = tile_str(tile)
    new_catalog = plan.new_objects_catalog

    if new_catalog.empty:
        return

    n_new = len(new_catalog)
    logger.debug(f'Tile {tile_key}: Appending {n_new} new objects')

    processed_ids, loaded_bands, failed_count = process_catalog_in_batches(
        tile=tile,
        tile_dir=tile_dir,
        catalog=new_catalog,
        bands=plan.bands_for_new_objects,
        band_dictionary=band_dictionary,
        output_path=output_path,
        cutout_size=cutout_size,
    )

    # Record results
    loaded_bands_set = set(loaded_bands)
    for obj_id in processed_ids:
        results.record_object_success(obj_id, loaded_bands_set)

    # Update band stats
    n_processed = len(processed_ids)
    for band in loaded_bands:
        results.record_band_stats(band, succeeded=n_processed, failed=failed_count)


# ========== Main Entry Point ==========


def create_cutouts_for_tile(
    tile: tuple[int, int],
    tile_dir: Path,
    bands: list[str],
    catalog: pd.DataFrame,
    all_band_dictionary: dict[str, BandCfg],
    output_dir: Path,
    cutout_size: int,
) -> CutoutResult:
    """Create cutouts for objects in a tile.

    Handles three scenarios:
        1. New file: create cutouts for all catalog objects in requested bands
        2. Existing file, new bands requested: add new band cutouts to existing
           objects that match the catalog (by coordinate)
        3. New objects in catalog: create cutouts in ALL available bands
           (existing + requested) so all objects have uniform band coverage

    Args:
        tile: Tile coordinates (x, y)
        tile_dir: Directory containing tile data
        bands: Bands to create cutouts for (in this run)
        catalog: DataFrame with object coordinates (ra, dec, x, y, ID)
        all_band_dictionary: Full band configuration
        output_dir: Directory to write HDF5 files
        cutout_size: Square cutout size in pixels

    Returns:
        CutoutResult with statistics and object band mapping
    """
    # Setup
    tile_key = tile_str(tile)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{tile_key}_cutouts_{cutout_size}.h5'

    # Sort requested bands by wavelength
    bands = get_wavelength_order(bands, all_band_dictionary)

    # Initialize result builder
    results = CutoutResultBuilder(bands)

    # Analyze existing file state
    existing_info = analyze_existing_file(output_path)

    # Handle corruption: file exists on disk but analysis failed
    if output_path.exists() and not existing_info.exists:
        logger.warning(
            f'Tile {tile_key}: Output file exists but appears corrupt. '
            f'Deleting and recreating: {output_path}'
        )
        try:
            output_path.unlink()
        except OSError as e:
            logger.error(f'Failed to delete corrupt file {output_path}: {e}')
            return results.build()

        # Reset existing_info after deletion
        existing_info = ExistingFileInfo(
            exists=False, bands=[], n_objects=0, object_coords=None, cutout_shape=None
        )

    # Plan operations
    plan = plan_cutout_operations(
        catalog=catalog,
        requested_bands=bands,
        existing_info=existing_info,
        all_band_dictionary=all_band_dictionary,
    )

    # Execute plan
    if plan.create_new_file:
        handle_new_file(
            plan=plan,
            tile=tile,
            tile_dir=tile_dir,
            output_path=output_path,
            band_dictionary=all_band_dictionary,
            cutout_size=cutout_size,
            results=results,
        )
    else:
        # Case 2: Add bands to existing objects (if needed)
        if plan.bands_to_add_to_existing and not plan.matched_catalog.empty:
            updated_info = handle_add_bands(
                plan=plan,
                tile=tile,
                tile_dir=tile_dir,
                output_path=output_path,
                band_dictionary=all_band_dictionary,
                cutout_size=cutout_size,
                results=results,
            )
            # Update plan with new info for potential new objects handling
            plan.existing_info = updated_info

        # Case 3: Append new objects (if needed)
        if not plan.new_objects_catalog.empty:
            handle_new_objects(
                plan=plan,
                tile=tile,
                tile_dir=tile_dir,
                output_path=output_path,
                band_dictionary=all_band_dictionary,
                cutout_size=cutout_size,
                results=results,
            )

    return results.build()


# ========== Batch Processing for Multiple Tiles ==========


def create_cutouts_for_existing_tiles(
    catalog: pd.DataFrame,
    bands: list[str],
    download_dir: Path,
    all_band_dictionary: dict[str, BandCfg],
    cutout_size: int,
    cutout_subdir: str,
    num_workers: int,
    log_dir: Path,
    log_name: str,
    log_level: int,
    run_stats: RunStatistics,
    cutout_success_map: dict[str, set[str]],
) -> None:
    """
    Create cutouts for tiles that already exist on disk.

    This is used when tile FITS files were downloaded in a previous run
    but cutouts still need to be created.

    Args:
        catalog: DataFrame with 'tile' and 'bands_to_cutout' columns
        bands: List of bands to create cutouts for
        download_dir: Root directory where tiles are stored
        all_band_dictionary: Full band configuration dictionary
        cutout_size: Square cutout size in pixels
        cutout_subdir: Subdirectory name for cutout HDF5 files
        num_workers: Number of parallel worker processes
        log_dir: Directory for log files
        log_name: Log file name
        log_level: Logging level
        run_stats: RunStatistics object for per-band tracking
        cutout_success_map: Dictionary to track which objects have cutouts in which bands

    Returns:
        None
    """

    if catalog.empty:
        return None
    # Filter to objects that actually need cutouts
    needs_cutout = catalog[catalog['bands_to_cutout'].map(len) > 0].copy()

    if needs_cutout.empty:
        logger.info('No objects need cutout creation.')
        return None

    # Group by tile
    tiles_to_process: dict[str, pd.DataFrame] = {}
    for tile_key, group in needs_cutout.groupby('tile'):
        tiles_to_process[str(tile_key)] = group.reset_index(drop=True)

    # Verify tiles exist on disk and collect valid ones
    valid_tiles: list[tuple[tuple[int, int], Path, list[str], pd.DataFrame]] = []

    for tile_key, tile_catalog in tiles_to_process.items():
        tile_parts = tile_key.split('_')
        tile = (int(tile_parts[0]), int(tile_parts[1]))
        tile_dir = download_dir / tile_key

        if not tile_dir.exists():
            logger.warning(f'Tile directory does not exist, skipping: {tile_dir}')
            continue

        # Check which bands are available for this tile
        available_bands = []
        for band in bands:
            band_dir = tile_dir / band
            if band_dir.exists() and any(band_dir.glob('*.fits')):
                available_bands.append(band)

        if not available_bands:
            logger.warning(f'No band data found for tile {tile_key}, skipping')
            continue

        valid_tiles.append((tile, tile_dir, available_bands, tile_catalog))

    if not valid_tiles:
        logger.warning('No valid tiles found with existing data.')
        return None

    mp_context = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_context,
        initializer=worker_log_init,
        initargs=(log_dir, log_name, log_level),
    ) as executor:
        # Submit all jobs
        futures: dict[str, Future[CutoutResult]] = {}

        for tile, tile_dir, available_bands, tile_catalog in valid_tiles:
            tile_key = tile_str(tile)
            output_dir = tile_dir / cutout_subdir

            future = executor.submit(
                create_cutouts_for_tile,
                tile=tile,
                tile_dir=tile_dir,
                bands=available_bands,
                catalog=tile_catalog,
                all_band_dictionary=all_band_dictionary,
                output_dir=output_dir,
                cutout_size=cutout_size,
            )
            futures[tile_key] = future

        for tile_key, future in futures.items():
            try:
                result = future.result(timeout=300)

                # Update per-band stats
                for band, band_stat in result.band_stats.items():
                    run_stats.record_cutout_result(
                        band,
                        succeeded=band_stat.get('succeeded', 0),
                        skipped=band_stat.get('skipped', 0),
                        failed=band_stat.get('failed', 0),
                    )

                cutout_success_map.update(result.object_bands)

                n_objects_processed = len(result.object_bands)
                if n_objects_processed > 0:
                    logger.debug(
                        f'✅ Cutouts for tile {tile_key}: Processed {n_objects_processed} objects'
                    )
                else:
                    logger.warning(f' Tile {tile_key}: created 0 cutouts (objects outside bounds?)')

            except Exception as e:
                for band in bands:
                    run_stats.record_cutout_result(band, failed=1)
                logger.error(f'❌ Cutouts failed for tile {tile_key}: {e}')


def write_to_h5(
    output_path: Path,
    cutouts: NDArray[np.float32],
    catalog: pd.DataFrame,
    bands: list[str],
    tile_key: str,
    band_dictionary: dict[str, BandCfg],
) -> None:
    """
    Write cutouts to HDF5. Creates new file if missing, handles band merging or
    object appending if file exists.

    Args:
        output_path: Path to HDF5 file
        cutouts: Cutout array, shape (n_objects, n_bands, h, w)
        catalog: DataFrame with ID, ra, dec columns
        bands: Band names corresponding to cutout channels
        tile_key: Tile identifier for logging
        band_dictionary: Full band config for wavelength ordering
    """
    n_new = len(cutouts)
    _, n_bands, h, w = cutouts.shape

    # Sort bands by wavelength for consistent storage
    sorted_bands = get_wavelength_order(bands, band_dictionary)
    if sorted_bands != bands:
        # Reorder cutout array to match wavelength order
        band_order = [bands.index(b) for b in sorted_bands]
        cutouts = cutouts[:, band_order, :, :]
        bands = sorted_bands

    # Prepare metadata
    ids_encoded = catalog['ID'].astype(str).to_numpy().astype('S')
    tile_encoded = np.array([tile_key.encode('utf-8')] * n_new)
    ra = catalog['ra'].to_numpy(dtype=np.float32)
    dec = catalog['dec'].to_numpy(dtype=np.float32)

    if not output_path.exists():
        # Case 1: Create new file with resizable datasets
        with h5py.File(str(output_path), 'w') as f:
            f.create_dataset('bands', data=np.array(bands, dtype='S'))

            # Allow both object and band dimensions to grow
            f.create_dataset(
                'cutouts',
                data=cutouts,
                maxshape=(None, None, h, w),
                chunks=True,
            )
            f.create_dataset('object_id', data=ids_encoded, maxshape=(None,), chunks=True)
            f.create_dataset('ra', data=ra, maxshape=(None,), chunks=True)
            f.create_dataset('dec', data=dec, maxshape=(None,), chunks=True)
            f.create_dataset('tile', data=tile_encoded, maxshape=(None,), chunks=True)

        logger.debug(f'Created {output_path} with {n_new} cutouts in {len(bands)} bands')
        return

    # Case 2: File exists - check bands
    existing_info = analyze_existing_file(output_path)

    # Case 2a: Bands match - simple append
    if set(existing_info.bands) == set(bands):
        with h5py.File(str(output_path), 'a') as f:
            for name, data in [
                ('cutouts', cutouts),
                ('object_id', ids_encoded),
                ('ra', ra),
                ('dec', dec),
                ('tile', tile_encoded),
            ]:
                dset = get_dataset(f, name)
                dset.resize(dset.shape[0] + n_new, axis=0)
                dset[-n_new:] = data

        logger.debug(f'Appended {n_new} cutouts to {output_path}')

    # Case 2b: Bands differ - need to merge bands and append
    else:
        all_bands = set(existing_info.bands) | set(bands)
        merged_bands = get_wavelength_order(list(all_bands), band_dictionary)
        n_merged = len(merged_bands)

        # Index mapping
        old_band_to_new_idx = {b: merged_bands.index(b) for b in existing_info.bands}
        input_band_to_new_idx = {b: merged_bands.index(b) for b in bands}

        with h5py.File(str(output_path), 'a') as f:
            cutouts_dset = get_dataset(f, 'cutouts')
            n_existing = cutouts_dset.shape[0]
            n_old_bands = cutouts_dset.shape[1]

            # 1. Resize band dimension (width)
            # Old data is now at [:, 0:n_old_bands, ...], we must reorder it
            cutouts_dset.resize((n_existing, n_merged, h, w))

            # 2. Reorder existing data in chunks (Memory Safe)
            batch_size = 100
            logger.debug(f'Reordering existing data for {n_merged} bands in chunks...')

            for i in range(0, n_existing, batch_size):
                end_i = min(i + batch_size, n_existing)
                n_batch = end_i - i

                # Read the old block (valid data is in the first n_old_bands columns)
                block_raw = cutouts_dset[i:end_i, :n_old_bands, :, :]

                # Create new ordered block
                block_new = np.zeros((n_batch, n_merged, h, w), dtype=np.float32)

                for old_idx, b_name in enumerate(existing_info.bands):
                    new_idx = old_band_to_new_idx[b_name]
                    block_new[:, new_idx, :, :] = block_raw[:, old_idx, :, :]

                # Write back
                cutouts_dset[i:end_i] = block_new

            # 3. Update bands list
            del f['bands']
            f.create_dataset('bands', data=np.array(merged_bands, dtype='S'))

            # 4. Append NEW objects (Memory Safe)
            # Resize rows to fit new objects
            total_rows = n_existing + n_new
            cutouts_dset.resize((total_rows, n_merged, h, w))

            # Write new cutouts into the correct bands
            logger.debug(f'Appending {n_new} new cutouts...')

            # Optimization: Write all rows for one band at a time to avoid allocating
            # the huge (n_new, n_merged, ...) padded array.
            for input_idx, b_name in enumerate(bands):
                target_idx = input_band_to_new_idx[b_name]
                # input cutouts is (n_new, n_input_bands, h, w)
                cutouts_dset[n_existing:, target_idx, :, :] = cutouts[:, input_idx, :, :]

            # 5. Append other metadata
            for name, data in [
                ('object_id', ids_encoded),
                ('ra', ra),
                ('dec', dec),
                ('tile', tile_encoded),
            ]:
                dset = get_dataset(f, name)
                dset.resize(dset.shape[0] + n_new, axis=0)
                dset[-n_new:] = data

        logger.debug(
            f'Appended {n_new} cutouts with band expansion '
            f'({existing_info.bands} -> {merged_bands}) to {output_path}'
        )
