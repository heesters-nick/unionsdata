import logging
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


def read_band_data(
    tile_dir: Path,
    tile: tuple[int, int],
    bands: list[str],
    in_dict: dict[str, BandDict],
) -> tuple[NDArray[np.float32], list[str]]:
    """
    Read image data for specified bands and tile.

    Args:
        parent_dir: Base directory containing tile directories
        tile_dir: Directory for the specific tile
        tile: Tuple of (tile_x, tile_y)
        bands: List of band names to read
        in_dict: Dictionary with band-specific file info

    Returns:
        Tuple of (multiband data array, list of successfully loaded bands)
    """
    multiband_data = np.zeros((len(bands), 10000, 10000), dtype=np.float32)
    loaded_bands = []

    for i, band in enumerate(bands):
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
            multiband_data[i, :, :] = data
            loaded_bands.append(band)
            logger.debug(f'Loaded {band} from {data_path.name}')
        except Exception as e:
            logger.warning(f'Failed to load {band} from {data_path}: {e}')
            continue

    if len(loaded_bands) == 0:
        raise ValueError(f'No bands could be loaded for tile {tile}')

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


def create_cutouts_for_tile(
    tile: tuple[int, int],
    tile_dir: Path,
    bands: list[str],
    catalog: pd.DataFrame,
    band_dictionary: dict[str, BandDict],
    output_dir: Path,
    cutout_size: int,
) -> tuple[int, int]:
    """
    Create cutouts for objects in a tile.
    Checks for existing objects via on-sky proximity and skips them.
    """
    tile_key = tile_str(tile)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{tile_key}_cutouts_{cutout_size}.h5'

    # 1. Filter out objects that are already processed
    catalog, skipped_count = filter_existing_objects(catalog, output_path, tile_key)

    if len(catalog) == 0:
        logger.debug(f'Tile {tile_key}: All objects already done. Skipping.')
        return 0, skipped_count

    # 2. Proceed with loading data only for the remaining objects
    n_objects = len(catalog)
    logger.debug(
        f'Tile {tile_key}: Skipping {skipped_count} existing objects. Creating {n_objects} NEW cutouts in {len(bands)} bands'
    )

    try:
        # Load image data (expensive step)
        multiband_data, loaded_bands = read_band_data(
            tile_dir=tile_dir,
            tile=tile,
            bands=bands,
            in_dict=band_dictionary,
        )

        # Create cutouts in memory
        cutouts = make_multiband_cutouts(
            multiband_data=multiband_data,
            tile_str=tile_key,
            df=catalog,
            cutout_size=cutout_size,
        )

        # Write to HDF5 (append or create)
        write_to_h5(output_path, cutouts, catalog, loaded_bands, tile_key)

        return n_objects, skipped_count

    except Exception as e:
        logger.error(f'Failed to create cutouts for tile {tile_key}: {e}')
        raise


def write_to_h5(
    output_path: Path,
    cutouts: NDArray[np.float32],
    catalog: pd.DataFrame,
    bands: list[str],
    tile_key: str,
) -> None:
    """
    Write cutouts to HDF5. Creates new file if missing, appends if exists.
    Uses maxshape to allow datasets to grow indefinitely.
    """
    n_new = len(cutouts)

    # Prepare data for writing
    # Convert string data to fixed-size bytes (S) for HDF5 compatibility
    ids_encoded = catalog['ID'].astype(str).to_numpy().astype('S')
    tile_encoded = np.array([tile_key.encode('utf-8')] * n_new)

    data_map = {
        'cutouts': cutouts,
        'object_id': ids_encoded,
        'ra': catalog['ra'].to_numpy(dtype=np.float32),
        'dec': catalog['dec'].to_numpy(dtype=np.float32),
        'tile': tile_encoded,
    }

    if not output_path.exists():
        # Case 1: Create new file with resizeable datasets
        with h5py.File(str(output_path), 'w') as f:
            f.create_dataset('bands', data=np.array(bands, dtype='S'))

            for name, data in data_map.items():
                # maxshape=(None, ...) makes the first dimension unlimited/resizeable
                maxshape = (None,) + data.shape[1:]
                f.create_dataset(name, data=data, maxshape=maxshape, chunks=True)

        logger.debug(f'Created {output_path} with {n_new} cutouts')

    else:
        # Case 2: Append to existing file
        with h5py.File(str(output_path), 'a') as f:
            # Quick sanity check to ensure we aren't mixing different band data
            if 'bands' in f:
                saved_bands = [b.decode('utf-8') for b in np.array(get_dataset(f, 'bands'))]
                if saved_bands != bands:
                    raise ValueError(
                        f'Band mismatch for {tile_key}. '
                        f'File has {saved_bands}, new data has {bands}.'
                    )

            # Resize and append
            for name, data in data_map.items():
                if name in f:
                    dset = get_dataset(f, name)
                    dset.resize(dset.shape[0] + n_new, axis=0)
                    dset[-n_new:] = data

        logger.debug(f'Appended {n_new} cutouts to {output_path}')


def filter_existing_objects(
    catalog: pd.DataFrame,
    output_path: Path,
    tile_key: str,
    threshold_arcsec: float = 5.0,
) -> tuple[pd.DataFrame, int]:
    """
    Filter out objects from the catalog that are already present in the HDF5 file
    based on on-sky proximity.
    """
    if not output_path.exists():
        return catalog, 0

    n_objects = len(catalog)

    try:
        with h5py.File(str(output_path), 'r') as f:
            # Check if we have coordinates to match against
            if 'ra' not in f or 'dec' not in f:
                return catalog, 0

            # Load existing coordinates
            existing_ra = get_dataset(f, 'ra')[:]
            existing_dec = get_dataset(f, 'dec')[:]

            # Create SkyCoord objects
            existing_coords = SkyCoord(ra=existing_ra, dec=existing_dec, unit='deg')
            new_coords = SkyCoord(
                ra=catalog['ra'].to_numpy(),
                dec=catalog['dec'].to_numpy(),
                unit='deg',
            )

            # Find nearest existing object for each new object
            # idx is the index in existing_coords, d2d is the separation
            idx, d2d, _ = new_coords.match_to_catalog_sky(existing_coords)

            # Define match threshold
            threshold = threshold_arcsec * Unit('arcsec')
            is_match = d2d < threshold

            # Filter catalog: keep objects that DO NOT have a match
            skipped_count = np.sum(is_match)
            if skipped_count > 0:
                logger.debug(
                    f'Tile {tile_key}: {skipped_count}/{n_objects} objects already cut out.'
                )
                return catalog[~is_match].copy(), skipped_count

    except Exception as e:
        logger.warning(f'Error checking existing file {output_path} for matches: {e}')

    return catalog, 0
