import logging
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header, ImageHDU, PrimaryHDU
from astropy.wcs.utils import skycoord_to_pixel
from numpy.typing import NDArray
from vos import Client

from unionsdata.config import BandCfg, InputsCfg
from unionsdata.kd_tree import TileWCS, build_tree, query_tree, relate_coord_tile

logger = logging.getLogger(__name__)


class TileAvailability:
    """Track availability of tiles across different bands."""

    def __init__(
        self,
        tile_nums: list[list[tuple[int, int]]],
        in_dict: dict[str, BandCfg],
        at_least: bool = False,
    ) -> None:
        """Initialize tile availability tracker.

        Args:
            tile_nums: List of tile lists per band, where each tile is (x, y)
            in_dict: Band configuration dictionary
            at_least: If True, count "at least N bands" instead of "exactly N bands"
        """

        self.all_tiles: list[list[tuple[int, int]]] = tile_nums
        self.tile_num_sets: list[set[tuple[int, int]]] = [
            cast(set[tuple[int, int]], set(tile_array)) for tile_array in self.all_tiles
        ]
        self.unique_tiles: list[tuple[int, int]] = cast(
            list[tuple[int, int]], sorted(set.union(*self.tile_num_sets))
        )
        self.availability_matrix: NDArray[np.int_] = self._create_availability_matrix()
        self.counts: dict[int, int] = self._calculate_counts(at_least)
        self.band_dict: dict[str, BandCfg] = in_dict

    def _create_availability_matrix(self) -> NDArray[np.int_]:
        """Create binary matrix of tile availability per band."""

        array_shape = (len(self.unique_tiles), len(self.all_tiles))
        availability_matrix = np.zeros(array_shape, dtype=int)

        for i, tile in enumerate(self.unique_tiles):
            for j, tile_num_set in enumerate(self.tile_num_sets):
                availability_matrix[i, j] = int(tile in tile_num_set)

        return availability_matrix

    def _calculate_counts(self, at_least: bool) -> dict[int, int]:
        """Calculate how many tiles are available in N bands."""

        counts = np.sum(self.availability_matrix, axis=1)
        bands_available, tile_counts = np.unique(counts, return_counts=True)

        counts_dict = dict(zip(bands_available, tile_counts, strict=True))

        if at_least:
            at_least_counts = np.zeros_like(bands_available)
            for i, _ in enumerate(bands_available):
                at_least_counts[i] = np.sum(tile_counts[i:])
            counts_dict = dict(zip(bands_available, at_least_counts, strict=True))

        return counts_dict

    def get_availability(self, tile_nums: tuple[int, int]) -> tuple[list[str], NDArray[np.intp]]:
        """Get bands available for a given tile.
        Args:
            tile_nums: Tile numbers (x, y)

        Returns:
            Tuple of (list of band names, numpy array of band indices)
        """

        try:
            index = self.unique_tiles.index(tile_nums)
        except ValueError:
            logger.warning(f'Tile number {tile_nums} not available in any band.')
            return [], np.array([], dtype=np.intp)
        except TypeError:
            return [], np.array([], dtype=np.intp)
        bands_available = np.where(self.availability_matrix[index] == 1)[0]
        return [list(self.band_dict.keys())[i] for i in bands_available], bands_available

    def band_tiles(self, band: str) -> list[tuple[int, int]]:
        """
        Get all tiles available in a specific band.

        Args:
            band: Band name

        Returns:
            List of tiles available in that band
        """

        tile_array: NDArray[np.object_] = np.array(self.unique_tiles)[
            self.availability_matrix[:, list(self.band_dict.keys()).index(band)] == 1
        ]
        return cast(list[tuple[int, int]], [tuple(tile) for tile in tile_array])

    def get_tiles_for_bands(self, bands: str | list[str] | None = None) -> list[tuple[int, int]]:
        """
        Get all tiles that are available in specified bands.
        If no bands are specified, return all unique tiles.

        Args:
            bands: Band name(s) to check for availability.
                                 Can be a single band name or a list of band names.

        Returns:
            list: List of tuples representing the tiles available in all specified bands.
        """
        if bands is None:
            return self.unique_tiles

        if isinstance(bands, str):
            bands = [bands]

        try:
            band_indices = [list(self.band_dict.keys()).index(band) for band in bands]
        except ValueError as e:
            logger.error(f'Invalid band name: {e}')
            return []

        # Get tiles available in all specified bands
        available_tiles: NDArray[np.intp] = np.where(
            self.availability_matrix[:, band_indices].all(axis=1)
        )[0]

        return [self.unique_tiles[i] for i in available_tiles]

    def stats(self, band: str | None = None) -> None:
        logger.info('Number of currently available tiles per band:')
        max_band_name_length = max(map(len, self.band_dict.keys()))  # for output format
        for band_name, count in zip(
            self.band_dict.keys(), np.sum(self.availability_matrix, axis=0), strict=False
        ):
            logger.info(f'{band_name.ljust(max_band_name_length)}: {count}')

        logger.info('Number of tiles available in different bands:')
        for bands_available, count in sorted(self.counts.items(), reverse=True):
            logger.info(f'In {bands_available} bands: {count}')

        logger.info(f'Number of unique tiles available: {len(self.unique_tiles)}')

        if band:
            logger.info(f'Number of tiles available in combinations containing the {band}-band:\n')

            all_bands = list(self.band_dict.keys())
            all_combinations: list[tuple[str, ...]] = []
            for r in range(1, len(all_bands) + 1):
                all_combinations.extend(combinations(all_bands, r))
            combinations_w_r = [x for x in all_combinations if band in x]

            for band_combination in combinations_w_r:
                band_combination_str = ''.join([str(x).split('-')[-1] for x in band_combination])
                band_indices = [
                    list(self.band_dict.keys()).index(band_c) for band_c in band_combination
                ]
                common_tiles = np.sum(self.availability_matrix[:, band_indices].all(axis=1))
                logger.info(f'{band_combination_str}: {common_tiles}')


def tile_finder(
    avail_all: TileAvailability,
    avail_sel: TileAvailability,
    all_unique_tiles: list[tuple[int, int]],
    catalog: pd.DataFrame | None,
    coord_c: SkyCoord | None,
    tile_info_dir: Path,
    band_constr: int = 5,
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None, pd.DataFrame]:
    """
    Finds tiles a list of objects are in.

    Args:
        avail_all: object to retrieve all available tiles and their bands for all bands
        avail_sel: object to retrieve all available tiles and their bands for selected bands
        all_unique_tiles: list of all unique tiles available
        catalog: object catalog
        coord_c: astropy SkyCoord object of the coordinates
        tile_info_dir: tile information directory
        band_constr: minimum number of bands that should be available. Defaults to 5.

    Returns:
        unique_tiles: unique tiles the objects are in
        tiles_x_bands: unique tiles with at least band_constr bands available
        catalog: updated catalog with tile information
    """
    if catalog is None or coord_c is None:
        return None, None, pd.DataFrame()

    # This helps the type checker
    assert coord_c is not None
    assert catalog is not None

    tiles_matching_catalog = np.empty(len(catalog), dtype=object)
    pix_coords = np.empty((len(catalog), 2), dtype=np.float64)
    bands_available = np.empty(len(catalog), dtype=object)
    n_bands_available = np.empty(len(catalog), dtype=np.int32)

    for i, obj_coord in enumerate(coord_c):
        assert obj_coord is not None
        tile_numbers = query_tree(
            all_unique_tiles,
            np.array([obj_coord.ra.deg, obj_coord.dec.deg]),
            tile_info_dir,
        )
        tiles_matching_catalog[i] = tile_numbers
        # check how many bands are available for this tile
        bands_tile, band_idx_tile = avail_all.get_availability(tile_numbers)
        bands_available[i], n_bands_available[i] = ','.join(bands_tile), len(band_idx_tile)
        if len(bands_tile) == 0:
            logger.warning(
                f'Object at ({obj_coord.ra.deg:.4f}, {obj_coord.dec.deg:.4f}) '
                f'is in tile {tile_numbers} which has no requested bands'
            )
            bands_available[i] = []
            pix_coords[i] = np.nan, np.nan
            continue
        wcs = TileWCS()
        wcs.set_coords(relate_coord_tile(nums=tile_numbers))
        pix_coord = skycoord_to_pixel(obj_coord, wcs.wcs_tile, origin=1)
        pix_coords[i] = pix_coord

    # add tile numbers and pixel coordinates to catalog
    catalog['tile'] = [tile_str(tile) for tile in tiles_matching_catalog]
    catalog['x'] = np.round(pix_coords[:, 0], 4)
    catalog['y'] = np.round(pix_coords[:, 1], 4)
    catalog['bands_available'] = bands_available
    catalog['n_bands_available'] = n_bands_available
    unique_tiles = list(set(tiles_matching_catalog.tolist()))
    tiles_x_bands = [
        tile for tile in unique_tiles if len(avail_sel.get_availability(tile)[1]) >= band_constr
    ]

    return unique_tiles, tiles_x_bands, catalog


def get_tile_numbers(name: str) -> tuple[int, int]:
    """
    Extract tile numbers from tile name
    Args:
        name: .fits file name of a given tile

    Returns:
        two three digit tile numbers
    """

    if name.startswith('calexp'):
        pattern = re.compile(r'(?<=[_-])(\d+)(?=[_.])')
    else:
        pattern = re.compile(r'(?<=\.)(\d+)(?=\.)')

    matches = pattern.findall(name)

    if len(matches) < 2:
        raise ValueError(f'Could not extract tile numbers from filename: {name}')

    return int(matches[0]), int(matches[1])


def extract_tile_numbers(
    tile_dict: dict[str, NDArray[np.str_]], in_dict: dict[str, BandCfg]
) -> list[list[tuple[int, int]]]:
    """
    Extract tile numbers from .fits file names.

    Args:
        tile_dict: lists of file names from the different bands
        in_dict: band dictionary

    Returns:
        num_lists: list of lists containing available tile numbers in the different bands
    """

    num_lists = []
    for band in list(in_dict.keys()):
        # Convert to tuples of ints
        tile_numbers = [get_tile_numbers(name) for name in tile_dict[band]]
        num_lists.append(tile_numbers)

    return num_lists


def load_available_tiles(path: Path, in_dict: dict[str, BandCfg]) -> dict[str, NDArray[np.str_]]:
    """
    Load tile lists from disk.
    Args:
        path: path to files
        in_dict: band dictionary

    Returns:
        dictionary of available tiles for the selected bands
    """

    band_tiles = {}
    for band in in_dict.keys():
        tiles = np.loadtxt(path / f'{band}_tiles.txt', dtype=str)
        band_tiles[band] = tiles

    return band_tiles


def update_available_tiles(path: Path, in_dict: dict[str, BandCfg], save: bool = True) -> None:
    """
    Update available tile lists from the VOSpace. Takes a few mins to run.

    Args:
        path: path to save tile lists.
        in_dict: band dictionary
        save: save new lists to disk, default is True.

    Returns:
        None
    """

    for band in np.array(list(in_dict.keys())):
        vos_dir = in_dict[band].vos
        band_filter = in_dict[band].band
        prefix = in_dict[band].name
        delimiter = in_dict[band].delimiter
        suffix = in_dict[band].suffix
        zfill = in_dict[band].zfill

        start_fetch = time.time()
        try:
            logger.info(f'Retrieving {band_filter}-band tiles...')
            # filter files based on their names
            if zfill == 0:
                band_tiles = Client().glob1(
                    vos_dir, f'{prefix}{delimiter}[0-9]*{delimiter}[0-9]*{suffix}'
                )
            else:
                digit_pattern = '[0-9]' * zfill
                band_tiles = Client().glob1(
                    vos_dir,
                    f'{prefix}{delimiter}{digit_pattern}{delimiter}{digit_pattern}{suffix}',
                )
            # filter out problematic files
            band_tiles = [tile for tile in band_tiles if '!' not in tile]
            end_fetch = time.time()
            logger.info(
                f'Retrieving {band_filter}-band tiles completed. Took {np.round((end_fetch - start_fetch) / 60, 3)} minutes.'
            )
            logger.info(f'Number of {band_filter}-band tiles: {len(band_tiles)}')
            if save:
                np.savetxt(os.path.join(path, f'{band}_tiles.txt'), band_tiles, fmt='%s')
        except Exception as e:
            logger.error(f'Error fetching {band_filter}-band tiles: {e}')


def query_availability(
    update: bool,
    in_dict: dict[str, BandCfg],
    show_stats: bool,
    tile_info_dir: Path,
) -> tuple[TileAvailability, list[list[tuple[int, int]]]]:
    """
    Gather information on the currently available tiles.

    Args:
        update: update the available tiles
        in_dict: band dictionary
        show_stats: show stats on the currently available tiles
        tile_info_dir: path to save the tile information

    Returns:
        A pair with the availability object and the band-by-tile listings.
    """
    # update information on the currently available tiles
    if update:
        update_available_tiles(tile_info_dir, in_dict)
    # extract the tile numbers from the available tiles
    all_bands = extract_tile_numbers(load_available_tiles(tile_info_dir, in_dict), in_dict)
    # create the tile availability object
    availability = TileAvailability(all_bands, in_dict)
    # build the kd tree
    if update:
        build_tree(availability.unique_tiles, tile_info_dir)
    # show stats on the currently available tiles
    if show_stats:
        availability.stats()
    return availability, all_bands


def import_coordinates(
    coordinates: list[tuple[float, float]],
    ra_key: str,
    dec_key: str,
    id_key: str,
) -> tuple[pd.DataFrame, SkyCoord]:
    """
    Process coordinates provided in the config file.

    Args:
        coordinates: ra, dec coordinates
        ra_key: right ascention key
        dec_key: declination key
        id_key: ID key

    Raises:
        ValueError: error if the number of coordinates is not even

    Returns:
        tuple: DataFrame, SkyCoord object of the coordinates
    """
    catalog = pd.DataFrame(coordinates, columns=[ra_key, dec_key], dtype=np.float32)
    # assign IDs to the coordinates
    catalog[id_key] = pd.RangeIndex(start=1, stop=len(catalog) + 1, step=1)
    logger.info('Coordinates received from config: %s', coordinates)
    coord_c = SkyCoord(
        ra=catalog[ra_key].to_numpy(),
        dec=catalog[dec_key].to_numpy(),
        unit='deg',
        frame='icrs',
    )
    return catalog, coord_c


def import_table(
    table_path: Path,
    ra_key: str,
    dec_key: str,
    id_key: str,
) -> tuple[pd.DataFrame, SkyCoord | None]:
    """
    Process a table provided in the config file.

    Args:
        table_path: path to the table
        ra_key: right ascention key
        dec_key: declination key
        id_key: ID key

    Returns:
        tuple: DataFrame, SkyCoord object of the coordinates
    """
    logger.info('Table read from config file.')
    catalog = pd.read_csv(table_path)

    # Add ID column if not present
    if id_key not in catalog.columns:
        catalog[id_key] = pd.RangeIndex(start=1, stop=len(catalog) + 1, step=1)

    if ra_key not in catalog.columns or dec_key not in catalog.columns:
        logger.error(
            'One or more keys not found in the table. Please provide the correct keys '
            'for right ascention and declination \n'
            'if they are not equal to the default keys: ra, dec.'
        )
        return pd.DataFrame(), None

    coord_c = SkyCoord(
        catalog[ra_key].to_numpy(),
        catalog[dec_key].to_numpy(),
        unit='deg',
        frame='icrs',
    )

    return catalog, coord_c


def import_tiles(
    tiles: list[tuple[int, int]], availability: TileAvailability, band_constr: int
) -> list[tuple[int, int]]:
    """
    Process tiles provided in the config file.

    Args:
        tiles: tile numbers
        availability: instance of the TileAvailability class
        band_constr: minimum number of bands that should be available

    Raises:
        ValueError: provide two three digit numbers for each tile

    Returns:
        list: list of tiles that are available in at least band_constr bands
    """
    logger.info(f'Tiles read from config file: {tiles}')

    return [tile for tile in tiles if len(availability.get_availability(tile)[1]) >= band_constr]


def input_to_tile_list(
    avail_all: TileAvailability,
    avail_sel: TileAvailability,
    all_unique_tiles: list[tuple[int, int]],
    band_constr: int,
    inputs: InputsCfg,
    tile_info_dir: Path,
    ra_key_default: str = 'ra',
    dec_key_default: str = 'dec',
    id_key_default: str = 'ID',
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None, pd.DataFrame]:
    """
    Process the input to get a list of tiles that are available in r and at least two other bands.

    Args:
        avail_all: instance of the TileAvailability class for all bands
        avail_sel: instance of the TileAvailability class for selected bands
        all_unique_tiles: list of all unique tiles available
        band_constr: minimum number of bands that should be available
        inputs: input dictionary with coordinates, a table, or tiles
        tile_info_dir: path to tile information.
        ra_key_default: default right ascention key. Defaults to 'ra'.
        dec_key_default: default declination key. Defaults to 'dec'.
        id_key_default: default ID key. Defaults to 'ID'.

    Returns:
        list: list of tiles that are available in r and at least two other bands
        catalog: updated catalog with tile information
    """
    source = inputs.source
    if source == 'coordinates':
        catalog, coord_c = import_coordinates(
            inputs.coordinates, ra_key_default, dec_key_default, id_key_default
        )
        if coord_c is None:
            logger.error('Failed to load coordinates.')
            return None, None, pd.DataFrame()
    elif source == 'table':
        catalog, coord_c = import_table(
            inputs.table.path,
            inputs.table.columns.ra,
            inputs.table.columns.dec,
            inputs.table.columns.id,
        )
        if coord_c is None:
            logger.error('Failed to load coordinates from table')
            return None, None, pd.DataFrame()
    elif source == 'tiles':
        return (
            None,
            import_tiles(inputs.tiles, avail_sel, band_constr),
            pd.DataFrame(),
        )
    else:
        logger.info('No coordinates, table or tiles provided. Processing all available tiles..')
        return None, None, pd.DataFrame()

    unique_tiles, tiles_x_bands, catalog = tile_finder(
        avail_all, avail_sel, all_unique_tiles, catalog, coord_c, tile_info_dir, band_constr
    )

    return unique_tiles, tiles_x_bands, catalog


def tile_str(tile: tuple[int, int]) -> str:
    return f'{tile[0]:03d}_{tile[1]:03d}'


# def decompress_fits(in_path: Path) -> None:
#     """
#     Decompress a tile-compressed FITS using funpack.
#     """

#     try:
#         subprocess.run(
#             ['funpack', '-F', str(in_path)],
#             check=True,
#             capture_output=True,
#             text=True,
#         )
#         logger.debug(f'Decompressed {in_path.name}')
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f'funpack failed for {in_path}: {e.stderr.strip()}') from e


def decompress_fits(file_path: Path, fits_ext: int = 1) -> None:
    """
    Decompress fits file by reading and saving again.
    """
    temp_path = file_path.with_name(file_path.name + '.tmp')

    try:
        # Read the data and header from the compressed file
        data, header = open_fits(file_path, fits_ext=fits_ext)
        # Create a new PrimaryHDU with the data and header
        new_hdu = fits.PrimaryHDU(data=data, header=header)
        # Overwrite the existing file
        new_hdu.writeto(temp_path, overwrite=True, checksum=True)
        # Atomically replace the original file with the decompressed version
        temp_path.replace(file_path)
    except Exception as e:
        # Clean up partial temp file if it exists
        if temp_path.exists():
            temp_path.unlink()

        # Raise as RuntimeError so we catch it correctly
        raise RuntimeError(f'Astropy decompression failed: {e}') from e


def open_fits(file_path: Path, fits_ext: int) -> tuple[NDArray[np.float32], Header]:
    """
    Open fits file and return data and header.
    Args:
        file_path: name of the fits file
        fits_ext: extension of the fits file
    Returns:
        data: image data
        header: header of the fits file
    """
    logger.debug(f'Opening fits file {file_path.name}..')
    start_opening = time.time()

    with fits.open(file_path, memmap=True) as hdul:
        if fits_ext > len(hdul) - 1:
            fits_ext = 0

        # Type assertion - tell the type checker this is an ImageHDU or PrimaryHDU
        hdu = cast(ImageHDU | PrimaryHDU, hdul[fits_ext])

        if hdu.data is None:
            raise ValueError(f'HDU extension {fits_ext} contains no data')

        data = hdu.data.astype(np.float32)
        header = hdu.header

    logger.debug(f'Fits file {file_path.name} opened in {time.time() - start_opening:.2f} seconds.')
    return data, header


def split_by_tile(catalog: pd.DataFrame, tiles: list[str]) -> dict[str, pd.DataFrame]:
    """
    Split catalog by tile numbers.

    Args:
        catalog: input catalog with a 'tile' column
        tiles: list of tile numbers as strings

    Returns:
        Dictionary mapping tile numbers to DataFrames containing entries for that tile
    """
    if catalog.empty:
        return {}
    tile_catalogs: dict[str, pd.DataFrame] = {}
    for tile in tiles:
        tile_catalog = catalog[catalog['tile'] == tile].reset_index(drop=True)
        if len(tile_catalog) > 0:
            tile_catalogs[tile] = tile_catalog
    return tile_catalogs


def read_h5(
    file_path: Path,
    needed_datasets: list[str] | None = None,
) -> dict[str, NDArray[np.float32]]:
    """Reads cutout data from HDF5 file with optimized dataset selection.

    Args:
        file_path: path to HDF5 file
        needed_datasets: list of datasets to read (None = read all)

    Returns:
        cutout_data (dict): dictionary with requested datasets
    """
    cutout_data = {}

    with h5py.File(str(file_path), 'r') as f:
        # Determine which datasets to load
        if needed_datasets is None:
            datasets_to_read = list(f.keys())
        else:
            datasets_to_read = [d for d in needed_datasets if d in f]

        # Loop through and load only needed datasets
        for dataset_name in datasets_to_read:
            if dataset_name == 'cutouts':
                data = np.nan_to_num(np.array(f[dataset_name]), nan=0.0)
            else:
                data = np.array(f[dataset_name])
            cutout_data[dataset_name] = data

    return cutout_data


def get_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """Get a dataset from an HDF5 file, with runtime validation."""
    obj = f[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected Dataset at '{key}', got {type(obj).__name__}")
    return obj


def get_bands_short_string(bands: list[str], band_dict: dict[str, BandCfg]) -> str:
    """
    Get short string representation of bands for filename.

    Args:
        bands: List of band keys (e.g., ['whigs-g', 'cfis-r', 'ps-i'])
        band_dict: Band configuration dictionary

    Returns:
        Concatenated band letters (e.g., 'gri')
    """
    return ''.join(band_dict[b].band for b in bands)


def load_existing_catalog(inputs: InputsCfg, tables_dir: Path) -> tuple[pd.DataFrame | None, Path]:
    """
    Load catalog from previous run if it exists.

    Args:
        inputs: Input configuration
        tables_dir: Directory where tables are stored

    Returns:
         Tuple containing DataFrame with existing catalog (or None if not found) and the path to the catalog file
    """
    if inputs.source == 'table':
        input_name = inputs.table.path.stem
    elif inputs.source == 'coordinates':
        input_name = 'input_coordinates'
    else:
        return None, Path()

    existing_path = tables_dir / f'{input_name}_augmented.csv'

    if not existing_path.exists():
        return None, existing_path

    try:
        catalog_existing = pd.read_csv(
            existing_path, dtype={'cutout_bands': str, 'bands_downloaded': str}
        )

        catalog_existing['cutout_bands'] = catalog_existing['cutout_bands'].fillna('')
        catalog_existing['bands_downloaded'] = catalog_existing['bands_downloaded'].fillna('')
        # drop these columns since they will be re-computed
        cols_to_drop = ['bands_available', 'n_bands_available', 'x', 'y', 'tile']
        catalog_existing.drop(
            columns=[c for c in cols_to_drop if c in catalog_existing.columns], inplace=True
        )

        logger.debug(
            f'Loaded existing augmented catalog history for {len(catalog_existing)} objects'
        )
        return catalog_existing, existing_path

    except Exception as e:
        logger.warning(f'Failed to load existing catalog: {e}. Starting fresh.')
        return None, existing_path


def merge_w_existing_catalog(
    fresh_catalog: pd.DataFrame,
    existing_catalog: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge fresh catalog data with existing augmented catalog.

    Preserves cutout_bands from existing catalog for matching IDs.
    New objects get empty cutout_bands.
    Objects removed from fresh catalog are dropped.

    Args:
        fresh_catalog: Newly processed catalog with tile/coord info
        existing_catalog: Previously saved augmented catalog (or None)

    Returns:
        Merged catalog with preserved cutout history
    """
    if existing_catalog is not None and not fresh_catalog.empty:
        fresh_catalog = fresh_catalog.merge(
            existing_catalog[['ID', 'bands_downloaded', 'cutout_bands']], on='ID', how='left'
        )
        logger.debug(f'Merged input with previous run; now {len(fresh_catalog)} total objects')
    for col in ['cutout_bands', 'bands_downloaded']:
        if col not in fresh_catalog.columns:
            fresh_catalog[col] = ''  # Create missing column
        else:
            fresh_catalog[col] = fresh_catalog[col].fillna('')  # Fix merge artifacts

    return fresh_catalog


def compute_bands_to_process(
    catalog: pd.DataFrame,
    bands: list[str],
    existing_col: str,
    output_col: str,
    require_all: bool = False,
) -> pd.DataFrame:
    """
    Compute which bands need processing for each object.

    Args:
        catalog: Augmented catalog DataFrame
        bands: List of bands to check for processing
        existing_col: Column name indicating already processed bands
        output_col: Column name to store bands needing processing
        require_all: If True, skip objects missing required bands in 'bands_available'

    Returns:
        Updated catalog with new column for bands needing processing
    """
    catalog = catalog.copy()
    req_bands_set = set(bands)

    def get_bands_to_process(row: pd.Series) -> list[str]:
        # Parse available bands
        val_avail = row['bands_available']
        available = set(str(val_avail).split(',')) if pd.notna(val_avail) and val_avail else set()
        available.discard('')

        # Check require_all constraint
        if require_all and not req_bands_set.issubset(available):
            potential = req_bands_set & available
            return []

        # Parse existing bands
        val_existing = row[existing_col]
        existing = (
            set(str(val_existing).split(',')) if pd.notna(val_existing) and val_existing else set()
        )
        existing.discard('')

        # Compute what's needed
        potential = req_bands_set & available

        return list(potential - existing)

    catalog[output_col] = catalog.apply(get_bands_to_process, axis=1)
    return catalog


def filter_for_processing(
    catalog: pd.DataFrame,
    bands: list[str],
    download_col: str,
    cutout_col: str,
    require_all: bool,
    cutouts_enabled: bool,
) -> pd.DataFrame | None:
    """
    Filter catalog to objects needing either downloads or cutouts.

    Creates two columns:
        - 'bands_to_download': bands needing tile download
        - 'bands_to_cutout': bands needing cutout creation

    Args:
        catalog: Augmented catalog DataFrame
        bands: List of requested bands
        download_col: Column tracking downloaded bands
        cutout_col: Column tracking cutout bands
        require_all: If True, skip objects missing required bands
        cutouts_enabled: Whether to compute cutout requirements

    Returns:
        Filtered catalog
    """
    # Compute download requirements
    catalog = compute_bands_to_process(
        catalog=catalog,
        bands=bands,
        existing_col=download_col,
        output_col='bands_to_download',
        require_all=require_all,
    )

    # Compute cutout requirements (if enabled)
    if cutouts_enabled:
        catalog = compute_bands_to_process(
            catalog=catalog,
            bands=bands,
            existing_col=cutout_col,
            output_col='bands_to_cutout',
            require_all=require_all,
        )

    else:
        catalog['bands_to_cutout'] = [[] for _ in range(len(catalog))]

    # Filter: keep rows that need EITHER downloads OR cutouts
    needs_work = (catalog['bands_to_download'].map(len) > 0) | (
        catalog['bands_to_cutout'].map(len) > 0
    )
    filtered_catalog = catalog[needs_work].reset_index(drop=True)

    # Logging
    n_need_download = (catalog['bands_to_download'].map(len) > 0).sum()
    n_need_cutout = (catalog['bands_to_cutout'].map(len) > 0).sum()

    logger.debug(f'Processing requirements for bands {bands}:')
    logger.debug(f'  Objects needing downloads: {n_need_download}/{len(catalog)}')
    logger.debug(f'  Objects needing cutouts:   {n_need_cutout}/{len(catalog)}')

    if filtered_catalog.empty:
        return None

    return filtered_catalog


def update_catalog(
    catalog: pd.DataFrame,
    successful_bands_map: dict[str, set[str]],
    band_dict: dict[str, BandCfg],
    col_name: str = 'cutout_bands',
) -> pd.DataFrame:
    """
    Update catalog for successfully processed objects.

    Args:
        catalog: Augmented catalog DataFrame
        successful_bands_map: Dictionary mapping object IDs to sets of successfully processed bands
        band_dict: Dictionary of band information

    Returns:
        Updated catalog with merged cutout_bands
    """
    catalog = catalog.copy()

    id_to_update = set(successful_bands_map.keys())
    # create boolean mask for rows to update
    mask = catalog['ID'].astype(str).isin(id_to_update)

    # early return if no rows to update
    if not mask.any():
        return catalog

    def merge_bands(row: pd.Series) -> str:
        obj_id = str(row['ID'])

        # get existing bands
        existing_str = row[col_name] if pd.notna(row[col_name]) else ''
        existing = set(existing_str.split(','))
        existing.discard('')

        # get new bands
        new_bands = successful_bands_map[obj_id]

        # merge
        updated_bands = existing | new_bands

        # sort by wavelength
        updated_bands_sorted = get_wavelength_order(
            bands=list(updated_bands), band_dictionary=band_dict
        )

        return ','.join(updated_bands_sorted)

    # apply merge only to rows needing update
    catalog.loc[mask, col_name] = catalog.loc[mask].apply(merge_bands, axis=1)

    return catalog


def update_bands_downloaded(
    catalog: pd.DataFrame,
    tile_progress: dict[str, set[str]],
    band_dict: dict[str, BandCfg],
) -> pd.DataFrame:
    """
    Update 'bands_downloaded' column based on tile-level success.

    Since FITS files are per-tile, if a tile is downloaded, we mark
    ALL objects in that tile as having those bands.

    Args:
        catalog: Augmented catalog DataFrame
        tile_progress: Dictionary mapping tile keys to sets of successfully downloaded bands
        band_dict: Dictionary of band information

    Returns:
        Updated catalog DataFrame
    """
    if catalog.empty or not tile_progress:
        return catalog

    # Filter to only successful tiles
    successful_tiles = {t: b for t, b in tile_progress.items() if b}

    if not successful_tiles:
        return catalog

    logger.debug(f"Updating 'bands_downloaded' for {len(successful_tiles)} tiles...")

    for tile_key, new_bands_set in successful_tiles.items():
        # Identify rows belonging to this tile
        mask = catalog['tile'] == tile_key

        if not mask.any():
            continue

        # Define the update logic for this specific tile
        def _update_row(existing_val: str, bands: set[str] = new_bands_set) -> str:
            existing_str = existing_val if pd.notna(existing_val) else ''
            existing_set = set(existing_str.split(','))
            existing_set.discard('')

            # Merge & Sort
            updated_set = existing_set | bands
            return ','.join(get_wavelength_order(list(updated_set), band_dict))

        # Apply update to just this slice
        catalog.loc[mask, 'bands_downloaded'] = catalog.loc[mask, 'bands_downloaded'].apply(
            _update_row
        )

    return catalog


def process_download_results(
    catalog: pd.DataFrame,
    tile_progress: dict[str, set[str]],
    cutout_success_map: dict[str, set[str]],
    cutouts_enabled: bool,
    band_dict: dict[str, BandCfg],
) -> pd.DataFrame:
    """
    Process results from download_tiles and update catalog columns.
    Delegates to specialized functions for different column types.

    Args:
        catalog: Augmented catalog DataFrame
        tile_progress: Dictionary mapping tile keys to sets of successfully downloaded bands
        cutout_success_map: Dictionary mapping object IDs to sets of successfully cutout bands
        cutouts_enabled: Whether cutouts were enabled
        band_dict: Dictionary of band information

    Returns:
        Updated catalog DataFrame
    """
    if catalog.empty:
        return catalog

    logger.debug('Updating catalog state...')

    # Update 'bands_downloaded'
    if tile_progress:
        catalog = update_bands_downloaded(
            catalog=catalog, tile_progress=tile_progress, band_dict=band_dict
        )

    # Update 'cutout_bands'
    if cutouts_enabled and cutout_success_map:
        logger.debug(f"Updating 'cutout_bands' for {len(cutout_success_map)} objects...")

        catalog = update_catalog(
            catalog=catalog,
            successful_bands_map=cutout_success_map,
            band_dict=band_dict,
            col_name='cutout_bands',
        )

    return catalog


def get_wavelength_order(bands: list[str], band_dictionary: dict[str, BandCfg]) -> list[str]:
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


def filter_catalog_all_bands(
    catalog: pd.DataFrame,
    required_bands: set[str],
    col_name: str = 'cutout_bands',
) -> pd.DataFrame:
    """
    Filter catalog to only include objects that have all required bands available.

    Args:
        catalog: Augmented catalog DataFrame
        required_bands: Set of bands that must be available for an object to be included
        col_name: Column name in catalog indicating available bands. Defaults to 'cutout_bands'.

    Returns:
        Filtered catalog DataFrame
    """
    if catalog.empty:
        return catalog

    def has_all_bands(row: pd.Series) -> bool:
        val = row[col_name]
        available = set(str(val).split(',')) if pd.notna(val) and val else set()
        available.discard('')
        return required_bands.issubset(available)

    filtered_catalog = catalog[catalog.apply(has_all_bands, axis=1)].reset_index(drop=True)

    logger.debug(
        f'Filtered catalog for objects with all required bands {required_bands}: '
        f'{len(filtered_catalog)}/{len(catalog)} objects remain.'
    )

    return filtered_catalog


def jobs_from_catalog(catalog: pd.DataFrame) -> set[tuple[tuple[int, int], str]]:
    """Get (tile, band) combinations from augmented input catalog that should be downloaded.

    Args:
        catalog: Augmented input catalog with 'tile' and 'bands_to_download' columns

    Returns:
        Set of (tile, band) tuples that need to be downloaded
    """
    needed_jobs = set()
    for tile_str, group in catalog.groupby('tile'):
        t_parts = str(tile_str).split('_')
        t_tuple = (int(t_parts[0]), int(t_parts[1]))

        unique_bands_for_tile = set()
        for bands_list in group['bands_to_download']:
            unique_bands_for_tile.update(bands_list)

        for b in unique_bands_for_tile:
            needed_jobs.add((t_tuple, b))

    return needed_jobs


def jobs_from_tiles(
    tiles: list[tuple[int, int]], bands: list[str], avail: TileAvailability, require_all: bool
) -> set[tuple[tuple[int, int], str]]:
    """Get (tile, band) combinations that should be downloaded.

    Args:
        tiles: List of tile numbers
        bands: List of requested bands
        avail: TileAvailability object
        require_all: Whether all bands are required for processing

    Returns:
        Set of (tile, band) tuples that need to be downloaded
    """

    needed_jobs = set()
    req_tiles = set(tiles)

    if require_all:
        logger.info(f'Requiring all specified bands: {bands}')
        tiles_to_process = set(avail.get_tiles_for_bands(bands)) & req_tiles

        for band in bands:
            for tile in tiles_to_process:
                needed_jobs.add((tile, band))
    else:
        logger.info(f'Requiring any of the specified bands: {bands}')
        for band in bands:
            tiles_to_process = set(avail.band_tiles(band)) & req_tiles

            for tile in tiles_to_process:
                needed_jobs.add((tile, band))

    return needed_jobs


def clean_process_columns(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Remove temporary processing columns from the catalog.

    Args:
        catalog: Augmented catalog DataFrame

    Returns:
        Cleaned catalog DataFrame
    """
    temp_cols = ['bands_to_download', 'bands_to_cutout']
    catalog_cleaned = catalog.drop(columns=[col for col in temp_cols if col in catalog.columns])

    return catalog_cleaned
