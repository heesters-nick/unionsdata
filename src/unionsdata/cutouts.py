import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from unionsdata.config import BandDict
from unionsdata.utils import open_fits, tile_str

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
    bands: list[str],  # Ordered by wavelength
    catalog: pd.DataFrame,
    band_dictionary: dict[str, BandDict],
    output_dir: Path,
    cutout_size: int,
) -> int:
    """
    Create cutouts for all objects in a completed tile.

    Args:
        tile: Tile numbers (x, y)
        tile_dir: Directory containing band subdirectories with FITS files
        bands: List of bands in wavelength order
        catalog: DataFrame with columns 'X', 'Y', 'ID', 'ra', 'dec'
        band_dictionary: Band configurations (has 'zfill', 'name', etc.)
        output_dir: Directory to save HDF5 files
        cutout_size: Size of square cutouts in pixels

    Returns:
        Number of cutouts created
    """
    tile_key = tile_str(tile)
    n_objects = len(catalog)

    logger.info(f'Creating {n_objects} cutouts for tile {tile_key} in {len(bands)} bands: {bands}')

    try:
        # Load all bands into single array: (n_bands, 10000, 10000)
        multiband_data, loaded_bands = read_band_data(
            tile_dir=tile_dir,
            tile=tile,
            bands=bands,
            in_dict=band_dictionary,
        )

        logger.debug(f'Loaded {len(loaded_bands)} bands: {loaded_bands}')

        # Create cutouts: (n_objects, n_bands, cutout_size, cutout_size)
        cutouts = make_multiband_cutouts(
            multiband_data=multiband_data,
            tile_str=tile_key,
            df=catalog,
            cutout_size=cutout_size,
        )

        logger.debug(f'Created cutout array with shape {cutouts.shape}')

        # Save to HDF5
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{tile_key}_cutouts.h5'

        with h5py.File(str(output_path), 'w') as f:
            # Main data
            f.create_dataset('cutouts', data=cutouts)

            # Metadata
            f.create_dataset('bands', data=np.array(loaded_bands, dtype='S'))
            f.create_dataset(
                'object_id',
                data=catalog['ID'].astype(str).to_numpy(),
                dtype=h5py.string_dtype(encoding='utf-8'),
            )
            f.create_dataset('ra', data=catalog['ra'].to_numpy(), dtype=np.float32)
            f.create_dataset('dec', data=catalog['dec'].to_numpy(), dtype=np.float32)
            f.create_dataset('tile', data=tile_key, dtype=h5py.string_dtype(encoding='utf-8'))

        logger.info(f'Saved {n_objects} cutouts to {output_path}')
        return n_objects

    except Exception as e:
        logger.error(f'Failed to create cutouts for tile {tile_key}: {e}')
        raise
