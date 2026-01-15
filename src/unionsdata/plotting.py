import logging
from itertools import chain
from pathlib import Path
from typing import Literal, TypedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from unionsdata.config import BandDict
from unionsdata.make_rgb import generate_rgb, normalize_mono, preprocess_cutout
from unionsdata.utils import filter_catalog_all_bands, get_bands_short_string, get_dataset

logger = logging.getLogger(__name__)


class CutoutData(TypedDict):
    """Dictionary containing loaded cutout data and metadata."""

    cutouts: NDArray[np.float32]  # Raw: (n, n_bands, h, w), Processed: (n, h, w, 3) or (n, h, w)
    ra: NDArray[np.float32]
    dec: NDArray[np.float32]
    tile: NDArray[np.object_]
    object_id: NDArray[np.object_]
    bands: list[str]  # list of 1 or 3 band names


def load_cutouts(
    catalog_path: Path,
    bands_to_plot: list[str],
    data_dir: Path,
    cutout_subdir: str,
    cutout_size: int,
) -> CutoutData:
    """
    Load cutouts from HDF5 files for plotting, using augmented catalog as index.

    Args:
        catalog_path: Path to augmented catalog CSV with cutout_created column
        bands_to_plot: List of exactly 1 or 3 band names for RGB visualization
        data_dir: Root data directory containing tile subdirectories
        cutout_subdir: Subdirectory name within each tile dir containing HDF5 files
        cutout_size: Square size of cutouts in pixels

    Returns:
        Dictionary with:
            - 'cutouts': array(n_objects, 3, cutout_size, cutout_size)
            - 'ra': array(n_objects,)
            - 'dec': array(n_objects,)
            - 'tile': array(n_objects,) dtype object
            - 'object_id': array(n_objects,) dtype object
            - 'bands': list of 3 band names

    Raises:
        ValueError: If bands_to_plot is not exactly 1 or 3 bands
        FileNotFoundError: If catalog doesn't exist
        RuntimeError: If no cutouts can be loaded with requested bands
    """
    # Validate inputs
    if len(bands_to_plot) not in (1, 3):
        raise ValueError(f'Expected 1 or 3 bands for plotting, got {len(bands_to_plot)}')

    if not catalog_path.exists():
        raise FileNotFoundError(f'Catalog not found: {catalog_path}')

    logger.debug(f'Loading cutouts for bands: {bands_to_plot}')

    # Load and filter catalog
    catalog = pd.read_csv(catalog_path)
    target = set(bands_to_plot)

    catalog_success = filter_catalog_all_bands(
        catalog=catalog,
        required_bands=target,
        col_name='cutout_bands',
    )

    logger.info(f'Found {len(catalog_success)}/{len(catalog)} objects with successful cutouts')

    if len(catalog_success) == 0:
        raise RuntimeError('No objects with successful cutouts found in catalog')

    # Group by tile
    unique_tiles = catalog_success['tile'].unique()
    logger.debug(f'Objects span {len(unique_tiles)} tiles')

    # Accumulate data from each tile
    cutouts_list: list[NDArray[np.float32]] = []
    ra_list: list[NDArray[np.float32]] = []
    dec_list: list[NDArray[np.float32]] = []
    tile_list: list[list[str]] = []
    object_id_list: list[list[str]] = []

    tiles_loaded = 0
    tiles_skipped = 0
    objects_loaded = 0

    for tile in unique_tiles:
        # Get catalog entries for this tile
        catalog_tile = catalog_success[catalog_success['tile'] == tile]
        catalog_ids = catalog_tile['ID'].astype(str).values

        # Build HDF5 path
        h5_path = data_dir / tile / cutout_subdir / f'{tile}_cutouts_{cutout_size}.h5'

        if not h5_path.exists():
            logger.warning(f'Skipping tile {tile}: HDF5 file not found at {h5_path}')
            tiles_skipped += 1
            continue

        try:
            with h5py.File(h5_path, 'r') as f:
                # Read bands available in this tile
                bands_bytes = np.array(get_dataset(f, 'bands'))
                bands_in_file = [
                    b.decode('utf-8') if isinstance(b, bytes) else b for b in bands_bytes
                ]

                # Check if all requested bands are present
                missing_bands = [b for b in bands_to_plot if b not in bands_in_file]
                if missing_bands:
                    logger.warning(
                        f'Skipping tile {tile}: missing bands {missing_bands}. '
                        f'Available: {bands_in_file}'
                    )
                    tiles_skipped += 1
                    continue

                # Read object IDs and find matching indices
                h5_ids = np.array(get_dataset(f, 'object_id')).astype(str)
                matches = [(cat_id, np.where(h5_ids == cat_id)[0]) for cat_id in catalog_ids]
                valid_matches = [(cat_id, idx[0]) for cat_id, idx in matches if len(idx) > 0]

                if len(valid_matches) == 0:
                    logger.warning(f'Skipping tile {tile}: no matching objects found in HDF5')
                    tiles_skipped += 1
                    continue

                matched_catalog_ids = [cat_id for cat_id, _ in valid_matches]
                indices_array = np.array([idx for _, idx in valid_matches], dtype=int)

                # HDF5 requires indices in increasing order for fancy indexing
                # Sort indices and keep track of original order to restore later
                sort_order = np.argsort(indices_array)
                sorted_indices = indices_array[sort_order]

                # Load cutouts and extract requested bands
                band_indices = [bands_in_file.index(b) for b in bands_to_plot]
                cutouts_raw = np.array(get_dataset(f, 'cutouts')[sorted_indices])

                # Restore original order
                restore_order = np.argsort(sort_order)
                cutouts_raw = cutouts_raw[restore_order]

                cutouts_3band = cutouts_raw[:, band_indices, :, :]
                cutouts_3band = np.nan_to_num(cutouts_3band, nan=0.0).astype(np.float32)

                # Load metadata (also need sorted indices for HDF5)
                ra_sorted = np.array(get_dataset(f, 'ra')[sorted_indices], dtype=np.float32)
                dec_sorted = np.array(get_dataset(f, 'dec')[sorted_indices], dtype=np.float32)

                # Restore original order for metadata
                ra = ra_sorted[restore_order]
                dec = dec_sorted[restore_order]

                # Append to lists
                cutouts_list.append(cutouts_3band)
                ra_list.append(ra)
                dec_list.append(dec)
                tile_list.append([tile] * len(indices_array))
                object_id_list.append(matched_catalog_ids)

                tiles_loaded += 1
                objects_loaded += len(indices_array)

                logger.debug(
                    f'Loaded {len(indices_array)} objects from tile {tile} (bands: {bands_to_plot})'
                )

        except Exception as e:
            logger.error(f'Error loading tile {tile}: {e}')
            tiles_skipped += 1
            continue

    # Check if any data was loaded
    if len(cutouts_list) == 0:
        raise RuntimeError(
            f'No cutouts could be loaded with bands {bands_to_plot}. Checked {len(unique_tiles)} tiles, all were skipped.'
        )

    # Concatenate all data
    logger.debug('Concatenating data from all tiles...')
    result: CutoutData = {
        'cutouts': np.concatenate(cutouts_list, axis=0),
        'ra': np.concatenate(ra_list, axis=0),
        'dec': np.concatenate(dec_list, axis=0),
        'tile': np.array(list(chain.from_iterable(tile_list)), dtype=object),
        'object_id': np.array(list(chain.from_iterable(object_id_list)), dtype=object),
        'bands': bands_to_plot,
    }

    # Validate output
    n_objects = len(result['cutouts'])
    assert len(result['ra']) == n_objects
    assert len(result['dec']) == n_objects
    assert len(result['tile']) == n_objects
    assert len(result['object_id']) == n_objects

    logger.debug('=' * 70)
    logger.debug('CUTOUT LOADING SUMMARY')
    logger.debug(f'  Tiles loaded: {tiles_loaded}')
    logger.debug(f'  Tiles skipped: {tiles_skipped}')
    logger.debug(f'  Total objects loaded: {objects_loaded}')
    logger.debug(f'  Bands: {bands_to_plot}')
    logger.debug(f'  Cutout shape: {result["cutouts"].shape}')
    logger.debug('=' * 70)

    return result


def cutouts_to_rgb(
    cutout_data: CutoutData,
    band_config: dict[str, BandDict],
    scaling_type: Literal['asinh', 'linear'] = 'asinh',
    stretch: float = 125,
    Q: float = 7.0,
    gamma: float = 0.25,
    standard_zp: float = 30.0,
) -> CutoutData:
    """
    Process raw cutouts to images ready for plotting.

    For 3 bands: applies flux adjustment, anomaly detection, and RGB scaling.
    For 1 band: applies anomaly detection and scaling, returns grayscale.

    Args:
        cutout_data: Dictionary with raw cutout data from load_cutouts
        band_config: Band configuration dictionary for flux adjustments
        scaling_type: Type of scaling ('asinh' or 'linear')
        stretch: Scaling factor controlling overall brightness
        Q: Softening parameter for asinh scaling
        gamma: Gamma correction factor
        standard_zp: Standard zero-point for flux normalization

    Returns:
        CutoutData dictionary with processed images:
            - RGB mode: (n_objects, h, w, 3)
            - Mono mode: (n_objects, h, w)

    Raises:
        ValueError: If cutout_data doesn't have 1 or 3 bands
    """
    n_objects = len(cutout_data['cutouts'])
    bands = cutout_data['bands']
    n_bands = len(bands)

    if n_bands not in (1, 3):
        raise ValueError(f'Expected 1 or 3 bands in cutout_data, got {n_bands}')

    is_mono = n_bands == 1
    mode_str = 'monochromatic' if is_mono else 'RGB'
    logger.debug(f'Processing {n_objects} cutouts to {mode_str} with bands: {bands}')

    cutout_size = cutout_data['cutouts'].shape[2]  # Square cutouts

    if is_mono:
        # Mono: output shape (n, h, w)
        images = np.zeros((n_objects, cutout_size, cutout_size), dtype=np.float32)

        for i in range(n_objects):
            images[i] = normalize_mono(
                cutout=cutout_data['cutouts'][i, 0],
                scaling_type=scaling_type,
                stretch=stretch,
                Q=Q,
                gamma=gamma,
            )
    else:
        # RGB: output shape (n, h, w, 3)
        images = np.zeros((n_objects, cutout_size, cutout_size, 3), dtype=np.float32)

        for i in range(n_objects):
            cutout_prep = preprocess_cutout(
                cutout=cutout_data['cutouts'][i],
                bands=bands,
                in_dict=band_config,
                standard_zp=standard_zp,
            )
            images[i] = generate_rgb(
                cutout=cutout_prep,
                scaling_type=scaling_type,
                stretch=stretch,
                Q=Q,
                gamma=gamma,
            )

    logger.debug(f'Successfully processed {n_objects} cutouts to {mode_str}')

    return {
        'cutouts': images,
        'ra': cutout_data['ra'],
        'dec': cutout_data['dec'],
        'tile': cutout_data['tile'],
        'object_id': cutout_data['object_id'],
        'bands': cutout_data['bands'],
    }


def find_most_recent_catalog(tables_dir: Path) -> Path | None:
    """
    Find the most recently modified augmented catalog in the tables directory.

    Args:
        tables_dir: Directory containing augmented catalog files

    Returns:
        Path to the most recent *_augmented.csv file, or None if none found
    """
    augmented_files = list(tables_dir.glob('*_augmented.csv'))

    if not augmented_files:
        return None

    # Sort by modification time, most recent first
    augmented_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if len(augmented_files) > 1:
        logger.info(
            f'Found {len(augmented_files)} augmented catalogs, '
            f'using most recent: {augmented_files[0].name}'
        )

    return augmented_files[0]


def build_plot_filename(
    catalog_name: str,
    size_pix: int,
    bands: list[str],
    band_dict: dict[str, BandDict],
    extension: str,
) -> str:
    """Build filename for saving cutout plots.

    Args:
        catalog_name: Name of the catalog
        size_pix: Size of cutouts in pixels
        bands: List of band names used
        band_dict: Band configuration dictionary
        extension: File extension (e.g., 'png', 'pdf')

    Returns:
        Filename string
    """
    band_str = get_bands_short_string(bands, band_dict)
    return f'{catalog_name}_cutouts_{size_pix}_{band_str}.{extension}'


def plot_cutouts(
    cutout_data: CutoutData,
    mode: Literal['grid', 'channel'],
    max_cols: int,
    figsize: tuple[int, int] | None,
    save_path: Path | None,
    show_plot: bool,
) -> None:
    """
    Display galaxy cutouts in a chosen format.

    For RGB data: supports 'grid' or 'channel' mode.
    For monochromatic data: always uses 'grid' mode.

    Args:
        cutout_data: Dictionary with cutout data and metadata
        mode: 'grid' for grid display, 'channel' for individual channels plus RGB
        max_cols: Maximum number of columns in grid mode
        figsize: Figure size in inches
        save_path: Path to save the figure, or None to skip saving
        show_plot: Whether to display the plot interactively

    Returns:
        None
    """
    n_total = len(cutout_data['cutouts'])
    if n_total == 0:
        logger.info('Nothing to display.')
        return

    cutouts = cutout_data['cutouts']
    all_coords = list(zip(cutout_data['ra'], cutout_data['dec'], strict=True))
    ids = cutout_data['object_id']

    # Detect mono vs RGB from array shape
    is_mono = cutouts.ndim == 3  # (n, h, w) for mono, (n, h, w, 3) for RGB

    if not is_mono and (cutouts.ndim != 4 or cutouts.shape[-1] != 3):
        raise ValueError(
            f'Expected shape (n, h, w) for mono or (n, h, w, 3) for RGB, got {cutouts.shape}. '
            f'Did you run cutouts_to_rgb()?'
        )

    # Force grid mode for mono
    if is_mono and mode == 'channel':
        logger.debug('Channel mode not supported for monochromatic data, using grid mode.')
        mode = 'grid'

    # Display cutouts in regular grid
    if mode == 'grid':
        mode_str = 'monochromatic' if is_mono else 'RGB'
        logger.debug(f'Plotting {mode_str} cutouts in a grid..')

        n_cols = min(max_cols, n_total)
        n_rows = (n_total + n_cols - 1) // n_cols  # Ceiling division

        if figsize is None:
            figsize = (3 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)

        for idx in range(n_total):
            i, j = divmod(idx, n_cols)
            ax = axes[i, j]

            if is_mono:
                ax.imshow(cutouts[idx], cmap='gray', origin='lower', aspect='equal')
            else:
                ax.imshow(cutouts[idx], origin='lower', aspect='equal')

            coord_text = f'{all_coords[idx][0]:.4f}, {all_coords[idx][1]:.4f}'
            label_text = f'{ids[idx]}\n{cutout_data["tile"][idx]}'

            ax.text(
                0.03,
                0.97,
                label_text,
                color='orange',
                fontweight='bold',
                bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 2},
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
            )

            ax.text(
                0.97,
                0.03,
                coord_text,
                color='orange',
                fontweight='bold',
                bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 2},
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
            )

            ax.set_xticks([])
            ax.set_yticks([])

        # Hide empty subplots
        for idx in range(n_total, n_rows * n_cols):
            i, j = divmod(idx, n_cols)
            axes[i, j].axis('off')

        plt.tight_layout(pad=0.5)

    # Display cutouts row-wise with individual channels and RGB image (RGB only)
    elif mode == 'channel':
        logger.debug('Plotting cutouts with individual channels plus RGB..')

        if figsize is None:
            figsize = (12, 3 * n_total)

        fig, axes = plt.subplots(n_total, 4, figsize=figsize, constrained_layout=True)
        axes = np.atleast_2d(axes)

        col_titles = ['Red', 'Green', 'Blue', 'RGB']
        for j, title in enumerate(col_titles):
            if n_total > 0:
                axes[0, j].set_title(title, fontsize=18, fontweight='bold', pad=10)

        for i in range(n_total):
            img = cutouts[i]

            coord_text = f'{all_coords[i][0]:.4f}, {all_coords[i][1]:.4f}'
            label_text = f'{ids[i]}\n{cutout_data["tile"][i]}'
            status_color = 'orange'

            # Display individual channels
            for j in range(3):
                axes[i, j].imshow(img[:, :, j], cmap='gray', origin='lower', aspect='equal')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

            # Display RGB image
            axes[i, 3].imshow(img, origin='lower', aspect='equal')
            axes[i, 3].text(
                0.97,
                0.03,
                coord_text,
                color=status_color,
                fontweight='bold',
                bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 2},
                transform=axes[i, 3].transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
            )

            axes[i, 3].text(
                0.03,
                0.97,
                label_text,
                color=status_color,
                fontweight='bold',
                bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 2},
                transform=axes[i, 3].transAxes,
                verticalalignment='top',
                horizontalalignment='left',
            )

            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'grid' or 'channel'.")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()
