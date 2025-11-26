import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from numpy.typing import NDArray

from unionsdata.config import BandDict
from unionsdata.utils import get_dataset, open_fits, tile_str

logger = logging.getLogger(__name__)


# ========== Helper Functions ==========


def get_wavelength_order(bands: list[str], band_dictionary: dict[str, BandDict]) -> list[str]:
    """
    Sort bands by wavelength order using the band dictionary's key order.

    The band dictionary keys are ordered from shortest to longest wavelength
    (u -> g -> r -> i -> z), so we filter and sort by that canonical order.

    Args:
        bands: List of band names to sort
        band_dictionary: Full band configuration dictionary (defines wavelength order)

    Returns:
        Bands sorted by wavelength (shortest to longest)
    """
    canonical_order = list(band_dictionary.keys())
    return [b for b in canonical_order if b in bands]


@dataclass
class ExistingFileInfo:
    """Information about an existing HDF5 cutout file."""

    exists: bool
    bands: list[str]
    n_objects: int
    object_coords: NDArray[np.float32] | None  # shape (n, 2) for ra, dec
    cutout_shape: tuple[int, ...] | None  # (n_objects, n_bands, h, w)


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


# ========= Core Functions ==========


def read_band_data(
    tile_dir: Path,
    tile: tuple[int, int],
    bands: list[str],
    in_dict: dict[str, BandDict],
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
        zfill = in_dict[band]['zfill']
        file_prefix = in_dict[band]['name']
        delimiter = in_dict[band]['delimiter']
        suffix = in_dict[band]['suffix']
        fits_ext = in_dict[band]['fits_ext']

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
    # Convert DataFrame coordinates to integer pixel positions
    xs = np.floor(df['x'].to_numpy(dtype=np.float32) + 0.5).astype(np.int32)
    ys = np.floor(df['y'].to_numpy(dtype=np.float32) + 0.5).astype(np.int32)

    n_bands = multiband_data.shape[0]
    n_objects = len(xs)

    # Pre-allocate output: (n_objects, n_bands, cutout_size, cutout_size)
    cutouts = np.zeros((n_objects, n_bands, cutout_size, cutout_size), dtype=multiband_data.dtype)

    size_half = cutout_size // 2
    y_max, x_max = multiband_data.shape[1], multiband_data.shape[2]

    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        # Calculate bounds in image coordinates
        y_start = max(0, y - size_half)
        y_end = min(y_max, y + (cutout_size - size_half))
        x_start = max(0, x - size_half)
        x_end = min(x_max, x + (cutout_size - size_half))

        # Check if object is completely outside image
        if y_start >= y_end or x_start >= x_end:
            logger.warning(f'Tile {tile_str}: Object at ({x}, {y}) is outside image bounds')
            continue

        # Calculate corresponding positions in cutout array
        cy_start = y_start - y + size_half
        cy_end = y_end - y + size_half
        cx_start = x_start - x + size_half
        cx_end = x_end - x + size_half

        # Extract cutout from ALL bands at once
        # Shape: (n_bands, cutout_y, cutout_x)
        cutouts[i, :, cy_start:cy_end, cx_start:cx_end] = multiband_data[
            :, y_start:y_end, x_start:x_end
        ]

    logger.debug(f'Tile {tile_str}: Created {n_objects} cutouts in {n_bands} bands')
    return cutouts


def merge_bands_into_h5(
    output_path: Path,
    new_cutouts: NDArray[np.float32],
    new_bands: list[str],
    existing_info: ExistingFileInfo,
    match_indices: NDArray[np.intp],
    band_dictionary: dict[str, BandDict],
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


def create_cutouts_for_tile(
    tile: tuple[int, int],
    tile_dir: Path,
    bands: list[str],
    catalog: pd.DataFrame,
    all_band_dictionary: dict[str, BandDict],
    output_dir: Path,
    cutout_size: int,
) -> tuple[int, int, int]:
    """
    Create cutouts for objects in a tile.

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
        all_band_dictionary: Full band configuration (must contain ALL bands
            for correct wavelength ordering)
        output_dir: Directory to write HDF5 files
        cutout_size: Square cutout size in pixels

    Returns:
        Tuple of (n_new_objects, n_updated_objects, n_skipped_objects)
    """
    tile_key = tile_str(tile)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{tile_key}_cutouts_{cutout_size}.h5'

    # Sort requested bands by wavelength
    bands = get_wavelength_order(bands, all_band_dictionary)

    # Analyze existing file state
    existing_info = analyze_existing_file(output_path)

    if not existing_info.exists:
        # Case 1: No existing file - create cutouts for all objects
        logger.debug(f'Tile {tile_key}: Creating new file with {len(catalog)} objects')

        multiband_data, loaded_bands = read_band_data(
            tile_dir=tile_dir, tile=tile, bands=bands, in_dict=all_band_dictionary
        )

        cutouts = make_multiband_cutouts(
            multiband_data=multiband_data,
            tile_str=tile_key,
            df=catalog,
            cutout_size=cutout_size,
        )

        write_to_h5(output_path, cutouts, catalog, loaded_bands, tile_key, all_band_dictionary)
        return len(catalog), 0, 0

    # File exists - determine what's new
    existing_bands_set = set(existing_info.bands)
    new_bands = [b for b in bands if b not in existing_bands_set]

    # Match catalog objects to existing objects
    catalog_coords = np.column_stack([catalog['ra'].to_numpy(), catalog['dec'].to_numpy()]).astype(
        np.float32
    )

    assert existing_info.object_coords is not None
    is_match, match_indices = match_objects_by_coords(catalog_coords, existing_info.object_coords)

    n_matched = int(np.sum(is_match))
    n_new_objects = len(catalog) - n_matched

    # Initialize counts
    n_updated_objects = 0
    n_skipped_objects = 0

    logger.debug(
        f'Tile {tile_key}: {n_matched} existing objects, {n_new_objects} new objects, '
        f'{len(new_bands)} new bands ({new_bands})'
    )

    # Case 2: Add new bands to existing objects
    if new_bands and n_matched > 0:
        n_updated_objects = n_matched
        logger.debug(f'Tile {tile_key}: Adding bands {new_bands} to {n_matched} existing objects')

        # Load data for new bands only
        multiband_data, loaded_new_bands = read_band_data(
            tile_dir=tile_dir, tile=tile, bands=new_bands, in_dict=all_band_dictionary
        )

        if loaded_new_bands:
            # Create cutouts for matched objects in new bands
            matched_catalog = catalog[is_match].reset_index(drop=True)
            new_band_cutouts = make_multiband_cutouts(
                multiband_data=multiband_data,
                tile_str=tile_key,
                df=matched_catalog,
                cutout_size=cutout_size,
            )

            # Get the match indices for objects that were matched
            matched_h5_indices = match_indices[is_match]

            merge_bands_into_h5(
                output_path=output_path,
                new_cutouts=new_band_cutouts,
                new_bands=loaded_new_bands,
                existing_info=existing_info,
                match_indices=matched_h5_indices,
                band_dictionary=all_band_dictionary,
            )

            # Re-read existing_info after merge - bands have changed
            existing_info = analyze_existing_file(output_path)

    elif n_matched > 0:
        n_skipped_objects = n_matched
        logger.debug(f'Tile {tile_key}: No new bands to add, skipping {n_matched} existing objects')

    # Case 3: Append new objects with cutouts in ALL available bands
    if n_new_objects > 0:
        logger.debug(f'Tile {tile_key}: Appending {n_new_objects} new objects')

        new_object_catalog = catalog[~is_match].reset_index(drop=True)

        # New objects should have cutouts in all bands (existing + requested)
        # Since tile data exists for existing bands, we can create full cutouts
        all_bands = get_wavelength_order(
            list(set(existing_info.bands) | set(bands)),
            all_band_dictionary,
        )

        multiband_data, loaded_bands = read_band_data(
            tile_dir=tile_dir, tile=tile, bands=all_bands, in_dict=all_band_dictionary
        )

        new_object_cutouts = make_multiband_cutouts(
            multiband_data=multiband_data,
            tile_str=tile_key,
            df=new_object_catalog,
            cutout_size=cutout_size,
        )

        write_to_h5(
            output_path,
            new_object_cutouts,
            new_object_catalog,
            loaded_bands,
            tile_key,
            all_band_dictionary,
        )

    return n_new_objects, n_updated_objects, n_skipped_objects


def write_to_h5(
    output_path: Path,
    cutouts: NDArray[np.float32],
    catalog: pd.DataFrame,
    bands: list[str],
    tile_key: str,
    band_dictionary: dict[str, BandDict],
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
