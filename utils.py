import os
import re
import time
from itertools import combinations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from vos import Client

from kd_tree import TileWCS, build_tree, query_tree, relate_coord_tile
from logging_setup import get_logger

client = Client()
logger = get_logger()


def tile_finder(avail, catalog, coord_c, tile_info_dir, band_constr=5):
    """
    Finds tiles a list of objects are in.

    Args:
        avail (object): object to retrieve available tiles
        catalog (dataframe): object catalog
        coord_c (astropy SkyCoord): astropy SkyCoord object of the coordinates
        tile_info_dir (str): tile information directory
        band_constr (int, optional): minimum number of bands that should be available. Defaults to 5.

    Returns:
        unique_tiles (list): unique tiles the objects are in
        tiles_x_bands (list): unique tiles with at least band_constr bands available
        catalog (dataframe): updated catalog with tile information
    """
    available_tiles = avail.unique_tiles
    tiles_matching_catalog = np.empty(len(catalog), dtype=tuple)
    pix_coords = np.empty((len(catalog), 2), dtype=np.float64)
    bands = np.empty(len(catalog), dtype=object)
    n_bands = np.empty(len(catalog), dtype=np.int32)
    for i, obj_coord in enumerate(coord_c):
        tile_numbers, _ = query_tree(
            available_tiles,
            np.array([obj_coord.ra.deg, obj_coord.dec.deg]),
            tile_info_dir,
        )
        tiles_matching_catalog[i] = tile_numbers
        # check how many bands are available for this tile
        bands_tile, band_idx_tile = avail.get_availability(tile_numbers)
        bands[i], n_bands[i] = bands_tile, len(band_idx_tile)
        if not bands_tile:
            bands[i] = np.nan
            pix_coords[i] = np.nan, np.nan
            continue
        wcs = TileWCS()
        wcs.set_coords(relate_coord_tile(nums=tile_numbers))
        pix_coord = skycoord_to_pixel(obj_coord, wcs.wcs_tile, origin=1)
        pix_coords[i] = pix_coord

    # add tile numbers and pixel coordinates to catalog
    catalog["tile"] = tiles_matching_catalog
    catalog["x"] = pix_coords[:, 0]
    catalog["y"] = pix_coords[:, 1]
    catalog["bands"] = bands
    catalog["n_bands"] = n_bands
    unique_tiles = list(set(tiles_matching_catalog))
    tiles_x_bands = [
        tile
        for tile in unique_tiles
        if len(avail.get_availability(tile)[1]) >= band_constr
    ]

    return unique_tiles, tiles_x_bands, catalog


def get_tile_numbers(name):
    """
    Extract tile numbers from tile name
    :param name: .fits file name of a given tile
    :return two three digit tile numbers
    """

    if name.startswith("calexp"):
        pattern = re.compile(r"(?<=[_-])(\d+)(?=[_.])")
    else:
        pattern = re.compile(r"(?<=\.)(\d+)(?=\.)")

    matches = pattern.findall(name)

    return tuple(map(int, matches))


def extract_tile_numbers(tile_dict, in_dict):
    """
    Extract tile numbers from .fits file names.

    Args:
        tile_dict: lists of file names from the different bands
        in_dict: band dictionary

    Returns:
        num_lists (list): list of lists containing available tile numbers in the different bands
    """

    num_lists = []
    for band in list(in_dict.keys()):  # Remove np.array wrapper
        # Convert to regular Python tuples with regular ints
        tile_numbers = [get_tile_numbers(name) for name in tile_dict[band]]
        num_lists.append(tile_numbers)  # Remove np.array wrapper

    return num_lists


def load_available_tiles(path, in_dict):
    """
    Load tile lists from disk.
    Args:
        path (str): path to files
        in_dict (dict): band dictionary

    Returns:
        dictionary of available tiles for the selected bands
    """

    band_tiles = {}
    for band in np.array(list(in_dict.keys())):
        tiles = np.loadtxt(os.path.join(path, f"{band}_tiles.txt"), dtype=str)
        band_tiles[band] = tiles

    return band_tiles


def update_available_tiles(path, in_dict, save=True):
    """
    Update available tile lists from the VOSpace. Takes a few mins to run.

    Args:
        path (str): path to save tile lists.
        in_dict (dict): band dictionary
        save (bool): save new lists to disk, default is True.

    Returns:
        None
    """

    for band in np.array(list(in_dict.keys())):
        vos_dir = in_dict[band]["vos"]
        band_filter = in_dict[band]["band"]
        prefix = in_dict[band]["name"]
        delimiter = in_dict[band]["delimiter"]
        suffix = in_dict[band]["suffix"]
        zfill = in_dict[band]["zfill"]

        start_fetch = time.time()
        try:
            logger.info(f"Retrieving {band_filter}-band tiles...")
            # filter files based on their names
            if zfill == 0:
                band_tiles = Client().glob1(
                    vos_dir, f"{prefix}{delimiter}[0-9]*{delimiter}[0-9]*{suffix}"
                )
            else:
                digit_pattern = "[0-9]" * zfill
                band_tiles = Client().glob1(
                    vos_dir,
                    f"{prefix}{delimiter}{digit_pattern}{delimiter}{digit_pattern}{suffix}",
                )
            # filter out problematic files
            band_tiles = [tile for tile in band_tiles if "!" not in tile]
            end_fetch = time.time()
            logger.info(
                f"Retrieving {band_filter}-band tiles completed. Took {np.round((end_fetch - start_fetch) / 60, 3)} minutes."
            )
            logger.info(f"Number of {band_filter}-band tiles: {len(band_tiles)}")
            if save:
                np.savetxt(
                    os.path.join(path, f"{band}_tiles.txt"), band_tiles, fmt="%s"
                )
        except Exception as e:
            logger.error(f"Error fetching {band_filter}-band tiles: {e}")


class TileAvailability:
    def __init__(self, tile_nums, in_dict, at_least=False, band=None):
        self.all_tiles = tile_nums
        self.tile_num_sets = [
            set(map(tuple, tile_array)) for tile_array in self.all_tiles
        ]
        self.unique_tiles = sorted(set.union(*self.tile_num_sets))
        self.availability_matrix = self._create_availability_matrix()
        self.counts = self._calculate_counts(at_least)
        self.band_dict = in_dict

    def _create_availability_matrix(self):
        array_shape = (len(self.unique_tiles), len(self.all_tiles))
        availability_matrix = np.zeros(array_shape, dtype=int)

        for i, tile in enumerate(self.unique_tiles):
            for j, tile_num_set in enumerate(self.tile_num_sets):
                availability_matrix[i, j] = int(tile in tile_num_set)

        return availability_matrix

    def _calculate_counts(self, at_least):
        counts = np.sum(self.availability_matrix, axis=1)
        bands_available, tile_counts = np.unique(counts, return_counts=True)

        counts_dict = dict(zip(bands_available, tile_counts))

        if at_least:
            at_least_counts = np.zeros_like(bands_available)
            for i, count in enumerate(bands_available):
                at_least_counts[i] = np.sum(tile_counts[i:])
            counts_dict = dict(zip(bands_available, at_least_counts))

        return counts_dict

    def get_availability(self, tile_nums):
        try:
            index = self.unique_tiles.index(tuple(tile_nums))
        except ValueError:
            logger.warning(f"Tile number {tile_nums} not available in any band.")
            return [], []
        except TypeError:
            return [], []
        bands_available = np.where(self.availability_matrix[index] == 1)[0]
        return [
            list(self.band_dict.keys())[i] for i in bands_available
        ], bands_available

    def band_tiles(self, band=None):
        tile_array = np.array(self.unique_tiles)[
            self.availability_matrix[:, list(self.band_dict.keys()).index(band)] == 1
        ]
        return [tuple(tile) for tile in tile_array]

    def get_tiles_for_bands(self, bands=None):
        """
        Get all tiles that are available in specified bands.
        If no bands are specified, return all unique tiles.

        Args:
            bands (str or list): Band name(s) to check for availability.
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
            logger.error(f"Invalid band name: {e}")
            return []

        # Get tiles available in all specified bands
        available_tiles = np.where(
            self.availability_matrix[:, band_indices].all(axis=1)
        )[0]

        return [self.unique_tiles[i] for i in available_tiles]

    def stats(self, band=None):
        logger.info("Number of currently available tiles per band:")
        max_band_name_length = max(map(len, self.band_dict.keys()))  # for output format
        for band_name, count in zip(
            self.band_dict.keys(), np.sum(self.availability_matrix, axis=0)
        ):
            logger.info(f"{band_name.ljust(max_band_name_length)}: {count}")

        logger.info("Number of tiles available in different bands:")
        for bands_available, count in sorted(self.counts.items(), reverse=True):
            logger.info(f"In {bands_available} bands: {count}")

        logger.info(f"Number of unique tiles available: {len(self.unique_tiles)}")

        if band:
            logger.info(
                f"Number of tiles available in combinations containing the {band}-band:\n"
            )

            all_bands = list(self.band_dict.keys())
            all_combinations = []
            for r in range(1, len(all_bands) + 1):
                all_combinations.extend(combinations(all_bands, r))
            combinations_w_r = [x for x in all_combinations if band in x]

            for band_combination in combinations_w_r:
                band_combination_str = "".join(
                    [str(x).split("-")[-1] for x in band_combination]
                )
                band_indices = [
                    list(self.band_dict.keys()).index(band_c)
                    for band_c in band_combination
                ]
                common_tiles = np.sum(
                    self.availability_matrix[:, band_indices].all(axis=1)
                )
                logger.info(f"{band_combination_str}: {common_tiles}")


def import_coordinates(coordinates, ra_key_default, dec_key_default, id_key_default):
    """
    Process coordinates provided from the command line.

    Args:
        coordinates (nested list): ra, dec coordinates
        ra_key_default (str): default right ascention key
        dec_key_default (str): default declination key
        id_key_default (str): default ID key

    Raises:
        ValueError: error if the number of coordinates is not even

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    coordinates = coordinates[0]
    if (len(coordinates) == 0) or len(coordinates) % 2 != 0:
        raise ValueError("Provide even number of coordinates.")

    ras, decs, ids = (
        coordinates[::2],
        coordinates[1::2],
        list(np.arange(1, len(coordinates) // 2 + 1)),
    )
    ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
    df_coordinates = pd.DataFrame({id_key: ids, ra_key: ras, dec_key: decs})

    formatted_coordinates = " ".join([f"({ra}, {dec})" for ra, dec in zip(ras, decs)])
    logger.info(f"Coordinates received from the command line: {formatted_coordinates}")
    catalog = df_coordinates
    coord_c = SkyCoord(
        catalog[ra_key].values, catalog[dec_key].values, unit="deg", frame="icrs"
    )
    return catalog, coord_c


def import_dataframe(
    dataframe_path,
    ra_key,
    dec_key,
    id_key,
    ra_key_default,
    dec_key_default,
    id_key_default,
):
    """
    Process a DataFrame provided from the command line.

    Args:
        dataframe_path (str): path to the DataFrame
        ra_key (str): right ascention key
        dec_key (str): declination key
        id_key (str): ID key
        ra_key_default (str): default right ascention key
        dec_key_default (str): default declination key
        id_key_default (str): default ID key

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    logger.info("Dataframe received from command line.")
    catalog = pd.read_csv(dataframe_path)

    if ra_key is None or dec_key is None or id_key is None:
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default

    if (
        ra_key not in catalog.columns
        or dec_key not in catalog.columns
        or id_key not in catalog.columns
    ):
        logger.error(
            "One or more keys not found in the DataFrame. Please provide the correct keys "
            "for right ascention, declination and object ID \n"
            "if they are not equal to the default keys: ra, dec, ID."
        )
        return None, None

    coord_c = SkyCoord(
        catalog[ra_key].values, catalog[dec_key].values, unit="deg", frame="icrs"
    )

    return catalog, coord_c


def import_tiles(tiles, availability, band_constr):
    """
    Process tiles provided from the command line.

    Args:
        tiles (nested list): tile numbers
        availability (TileAvailability): instance of the TileAvailability class
        band_constr (int): minimum number of bands that should be available

    Raises:
        ValueError: provide two three digit numbers for each tile

    Returns:
        list: list of tiles that are available in r and at least two other bands
    """
    tiles = tiles[0]
    if (len(tiles) == 0) or len(tiles) % 2 != 0:
        raise ValueError("Provide two three digit numbers for each tile.")

    tile_list = [tuple(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    logger.info(f"Tiles received from command line: {tiles}")

    return [
        tile
        for tile in tile_list
        if "r" in availability.get_availability(tile)[0]
        and len(availability.get_availability(tile)[1]) >= band_constr
    ]


def query_availability(
    update, in_dict, at_least_key, show_stats, build_kdtree, tile_info_dir
):
    """
    Gather information on the currently available tiles.

    Args:
        update (bool): update the available tiles
        in_dict (dict): band dictionary
        at_least_key (bool): print the number of tiles in at least (not exactly) 5, 4, ... bands
        show_stats (bool): show stats on the currently available tiles
        build_kdtree (bool): build a kd tree from the currently available tiles
        tile_info_dir (str): path to save the tile information

    Returns:
        TileAvailability: availability of the tiles
    """
    # update information on the currently available tiles
    if update:
        update_available_tiles(tile_info_dir, in_dict)
    # extract the tile numbers from the available tiles
    all_bands = extract_tile_numbers(
        load_available_tiles(tile_info_dir, in_dict), in_dict
    )
    # create the tile availability object
    availability = TileAvailability(all_bands, in_dict, at_least_key)
    # build the kd tree
    if build_kdtree:
        build_tree(availability.unique_tiles, tile_info_dir)
    # show stats on the currently available tiles
    if show_stats:
        availability.stats()
    return availability, all_bands


def tile_str(tile):
    return f"({tile[0]}, {tile[1]})"
