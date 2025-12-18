import logging
import pickle
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from numpy.typing import NDArray
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def build_tree(tiles: list[tuple[int, int]], tile_info_dir: Path, save: bool = True) -> None:
    """
    Build a kd tree from the input tiles to efficiently query object positions.

    Args:
        tiles: list of unique tile number pairs
        tile_info_dir: path to save the tree
        save: save the built tree to file

    Returns:
        None
    """
    if not tiles:
        raise ValueError('Cannot build tree from empty tile list')

    logger.debug('Building kd tree..')
    tile_coords = np.array([relate_coord_tile(nums=num) for num in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])  # type: ignore
    tree = cKDTree(tile_coords_xyz)
    if save:
        with open(tile_info_dir / 'kdtree_xyz.pkl', 'wb') as f:
            pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug('KD tree built successfully.')


def query_tree(
    tiles: list[tuple[int, int]], coords: NDArray[np.float64], tile_info_dir: Path
) -> tuple[int, int]:
    """
    Query the kd tree to find what tile an object is in.

    Args:
        tiles: list of tile numbers as tuples
        coords: ra, dec of object to query
        tile_info_dir: path to save and load the tree

    Returns:
        tile name and distance object - nearest tile center
    """
    tree_path = tile_info_dir / 'kdtree_xyz.pkl'

    with open(tree_path, 'rb') as f:
        loaded_tree = pickle.load(f)
    logger.debug(f'Loaded kd tree from {tree_path}')
    try:
        tile_name, _ = find_tile(loaded_tree, tiles, coords)
        return tile_name
    except ValueError as e:
        raise e


class TileWCS:
    """
    Class to create a WCS object for a tile.
    """

    def __init__(self, wcs_keywords: dict[str, float | int | str] | None = None):
        if wcs_keywords is None:
            wcs_keywords = {}
        wcs_keywords.update(
            {
                'NAXIS': 2,
                'CTYPE1': 'RA---TAN',
                'CTYPE2': 'DEC--TAN',
                'CRVAL1': 0,
                'CRVAL2': 0,
                'CRPIX1': 5000.0,
                'CRPIX2': 5000.0,
                'CD1_1': -5.160234650248e-05,
                'CD1_2': 0.0,
                'CD2_1': 0.0,
                'CD2_2': 5.160234650248e-05,
                'NAXIS1': 10000,
                'NAXIS2': 10000,
            }
        )

        self.wcs_tile = WCS(wcs_keywords)

    def set_coords(self, coords: tuple[float, float]) -> None:
        self.wcs_tile.wcs.crval = [coords[0], coords[1]]


def find_tile(
    tree: cKDTree, tiles: list[tuple[int, int]], object_coord: NDArray[np.float64]
) -> tuple[tuple[int, int], float]:
    """
    Query the tree and find the tile the object is in.

    Args:
        tree: kd tree
        tiles: list of unique tiles in the survey
        object_coord: coordinates of the object we want to find a tile for

    Returns:
        tile numbers of the matching tile
    """
    ra_deg, dec_deg = object_coord
    coord_c = SkyCoord(ra_deg, dec_deg, unit='deg', frame='icrs')
    coord_xyz = np.asarray(coord_c.cartesian.xyz.value, dtype=float)  # type: ignore

    k = min(4, len(tiles))
    dists, indices = tree.query(coord_xyz, k=k)

    # Convert to Python scalars for clean typing
    dists_list: list[float] = [float(x) for x in np.atleast_1d(dists).tolist()]
    idx_list: list[int] = [int(i) for i in np.atleast_1d(indices).tolist()]

    wcs = TileWCS()
    for dist, idx in zip(dists_list, idx_list, strict=True):
        wcs.set_coords(relate_coord_tile(nums=tiles[idx]))
        if wcs.wcs_tile.footprint_contains(coord_c):
            return tiles[idx], dist
    raise ValueError(f'Object {ra_deg} {dec_deg} could not be assigned to a tile.')


def relate_coord_tile(
    coords: tuple[float, float] | None = None, nums: tuple[int, int] | None = None
) -> tuple[int, int] | tuple[float, float]:
    """
    Conversion between tile numbers and coordinates.

    Args:
        right ascention, declination: ra and dec coordinates
        nums: first and second tile numbers

    Returns:
        tuple: depending on the input, return the tile numbers or the ra and dec coordinates

    Raises:
        ValueError: if neither coords nor nums are provided
        TypeError: if coords is not a tuple of two floats
        TypeError: if nums is not a tuple of two ints
    """
    if coords:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise TypeError('coords must be a tuple of (ra, dec)')
        ra, dec = coords
        xxx = ra * 2 * np.cos(np.radians(dec))
        yyy = (dec + 90) * 2
        return int(np.round(xxx)), int(np.round(yyy))
    elif nums:
        if not isinstance(nums, tuple) or len(nums) != 2:
            raise TypeError('nums must be a tuple of (first_tile_num, second_tile_num)')
        xxx, yyy = nums
        dec = yyy / 2 - 90
        ra = xxx / 2 / np.cos(np.radians(dec))
        return float(np.round(ra, 12)), float(np.round(dec, 12))
    else:
        raise ValueError('Either coords or nums must be provided.')
