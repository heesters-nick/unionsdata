import logging
import os
import re
import subprocess
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

from unionsdata.config import BandDict, InputsCfg
from unionsdata.kd_tree import TileWCS, build_tree, query_tree, relate_coord_tile

client = Client()
logger = logging.getLogger(__name__)


class TileAvailability:
    """Track availability of tiles across different bands."""

    def __init__(
        self,
        tile_nums: list[list[tuple[int, int]]],
        in_dict: dict[str, BandDict],
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
        self.band_dict: dict[str, BandDict] = in_dict

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
    avail: TileAvailability,
    all_unique_tiles: list[tuple[int, int]],
    catalog: pd.DataFrame | None,
    coord_c: SkyCoord | None,
    tile_info_dir: Path,
    band_constr: int = 5,
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None, pd.DataFrame]:
    """
    Finds tiles a list of objects are in.

    Args:
        avail: object to retrieve available tiles
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

    # available_tiles = avail.unique_tiles

    tiles_matching_catalog = np.empty(len(catalog), dtype=object)
    pix_coords = np.empty((len(catalog), 2), dtype=np.float64)
    bands = np.empty(len(catalog), dtype=object)
    n_bands = np.empty(len(catalog), dtype=np.int32)
    for i, obj_coord in enumerate(coord_c):
        assert obj_coord is not None
        tile_numbers, _ = query_tree(
            all_unique_tiles,
            np.array([obj_coord.ra.deg, obj_coord.dec.deg]),
            tile_info_dir,
        )
        tiles_matching_catalog[i] = tile_numbers
        # check how many bands are available for this tile
        bands_tile, band_idx_tile = avail.get_availability(tile_numbers)
        bands[i], n_bands[i] = bands_tile, len(band_idx_tile)
        if len(bands_tile) == 0:
            logger.warning(
                f'Object at ({obj_coord.ra.deg:.4f}, {obj_coord.dec.deg:.4f}) '
                f'is in tile {tile_numbers} which has no requested bands'
            )
            bands[i] = []
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
    catalog['bands'] = bands
    catalog['n_bands'] = n_bands
    unique_tiles = list(set(tiles_matching_catalog.tolist()))
    tiles_x_bands = [
        tile for tile in unique_tiles if len(avail.get_availability(tile)[1]) >= band_constr
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
    tile_dict: dict[str, NDArray[np.str_]], in_dict: dict[str, BandDict]
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


def load_available_tiles(path: Path, in_dict: dict[str, BandDict]) -> dict[str, NDArray[np.str_]]:
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


def update_available_tiles(path: Path, in_dict: dict[str, BandDict], save: bool = True) -> None:
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
        vos_dir = in_dict[band]['vos']
        band_filter = in_dict[band]['band']
        prefix = in_dict[band]['name']
        delimiter = in_dict[band]['delimiter']
        suffix = in_dict[band]['suffix']
        zfill = in_dict[band]['zfill']

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
    in_dict: dict[str, BandDict],
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
        tuple: dataframe, SkyCoord object of the coordinates
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


def import_dataframe(
    dataframe_path: Path,
    ra_key: str,
    dec_key: str,
    id_key: str,
) -> tuple[pd.DataFrame, SkyCoord | None]:
    """
    Process a DataFrame provided in the config file.

    Args:
        dataframe_path: path to the DataFrame
        ra_key: right ascention key
        dec_key: declination key
        id_key: ID key

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    logger.info('Dataframe read from config file.')
    catalog = pd.read_csv(dataframe_path)

    # Add ID column if not present
    if id_key not in catalog.columns:
        catalog[id_key] = pd.RangeIndex(start=1, stop=len(catalog) + 1, step=1)

    if ra_key not in catalog.columns or dec_key not in catalog.columns:
        logger.error(
            'One or more keys not found in the DataFrame. Please provide the correct keys '
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
        list: list of tiles that are available in r and at least two other bands
    """
    logger.info(f'Tiles read from config file: {tiles}')

    return [tile for tile in tiles if len(availability.get_availability(tile)[1]) >= band_constr]


def input_to_tile_list(
    availability: TileAvailability,
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
        availability: instance of the TileAvailability class
        all_unique_tiles: list of all unique tiles available
        band_constr: minimum number of bands that should be available
        inputs: input dictionary with coordinates, a dataframe, or tiles
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
    elif source == 'dataframe':
        catalog, coord_c = import_dataframe(
            inputs.dataframe.path,
            inputs.dataframe.columns.ra,
            inputs.dataframe.columns.dec,
            inputs.dataframe.columns.id,
        )
        if coord_c is None:
            logger.error('Failed to load coordinates from dataframe')
            return None, None, pd.DataFrame()
    elif source == 'tiles':
        return (
            None,
            import_tiles(inputs.tiles, availability, band_constr),
            pd.DataFrame(),
        )
    else:
        logger.info('No coordinates, DataFrame or tiles provided. Processing all available tiles..')
        return None, None, pd.DataFrame()

    unique_tiles, tiles_x_bands, catalog = tile_finder(
        availability, all_unique_tiles, catalog, coord_c, tile_info_dir, band_constr
    )

    return unique_tiles, tiles_x_bands, catalog


def tile_str(tile: tuple[int, int]) -> str:
    return f'{tile[0]:03d}_{tile[1]:03d}'


def decompress_fits(in_path: Path) -> None:
    """
    Decompress a tile-compressed FITS using funpack.
    """

    try:
        subprocess.run(
            ['funpack', '-F', str(in_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(f'Decompressed {in_path.name}')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'funpack failed for {in_path}: {e.stderr.strip()}') from e


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
