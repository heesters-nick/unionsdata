import pickle
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree

from unionsdata.kd_tree import TileWCS, build_tree, find_tile, query_tree, relate_coord_tile


@pytest.mark.parametrize(
    'coords, expected_nums',
    [
        ((194.0, 56.0), (217, 292)),
        ((217.7556, 57.5), (234, 295)),
        ((211.8287, 55.0), (243, 290)),
        ((2.758445, 25.0), (5, 230)),
    ],
)
def test_relate_coord_tile_coords_to_nums(coords, expected_nums):
    """Test conversion from coordinates to tile numbers."""
    nums = relate_coord_tile(coords=coords)
    assert nums == expected_nums


@pytest.mark.parametrize(
    'nums, expected_coords',
    [
        ((217, 292), (194.0296440200, 56.0)),
        ((234, 295), (217.7556, 57.5)),
        ((243, 290), (211.8287, 55.0)),
        ((5, 230), (2.758445, 25.0)),
    ],
)
def test_relate_coord_tile_nums_to_coords(nums, expected_coords):
    """Test conversion from tile numbers to coordinates."""
    coords = relate_coord_tile(nums=nums)
    ra, dec = coords
    exp_ra, exp_dec = expected_coords
    assert pytest.approx(ra, rel=1e-4) == exp_ra
    assert pytest.approx(dec, rel=1e-4) == exp_dec


def test_relate_coord_tile_no_args_raises_error():
    """Test that providing neither coords nor nums raises ValueError."""

    with pytest.raises(ValueError, match='Either coords or nums must be provided'):
        relate_coord_tile()


def test_relate_coord_tile_both_args_uses_coords():
    """Test that coords takes precedence when both are provided."""

    # Provide both - coords should be used
    result = relate_coord_tile(coords=(180.0, 0.0), nums=(999, 999))

    # Should use coords (180, 0) -> (360, 180)
    assert result == (360, 180)


# ========== TileWCS Tests ==========


def test_tile_wcs_initialization_default():
    """Test TileWCS initialization with default keywords."""

    wcs = TileWCS()

    assert wcs.wcs_tile is not None
    assert wcs.wcs_tile.wcs.crval[0] == 0
    assert wcs.wcs_tile.wcs.crval[1] == 0
    assert wcs.wcs_tile.wcs.crpix[0] == 5000.0
    assert wcs.wcs_tile.wcs.crpix[1] == 5000.0


def test_tile_wcs_set_coords():
    """Test setting coordinates on TileWCS."""

    wcs = TileWCS()
    new_coords = (227.3042, 52.5285)

    wcs.set_coords(new_coords)

    assert wcs.wcs_tile.wcs.crval[0] == new_coords[0]
    assert wcs.wcs_tile.wcs.crval[1] == new_coords[1]


def test_tile_wcs_footprint_contains():
    """Test that WCS footprint_contains works."""

    wcs = TileWCS()
    wcs.set_coords((180.0, 0.0))

    # Coordinate at tile center should be contained
    coord = SkyCoord(180.0, 0.0, unit='deg', frame='icrs')
    assert wcs.wcs_tile.footprint_contains(coord)

    # Coordinate far away should not be contained
    coord_far = SkyCoord(0.0, 0.0, unit='deg', frame='icrs')
    assert not wcs.wcs_tile.footprint_contains(coord_far)


# ========== build_tree Tests ==========


def test_build_tree_creates_file(tmp_path: Path):
    """Test that build_tree creates a kdtree file."""

    tiles = [(217, 292), (234, 295), (240, 300)]
    tile_info_dir = tmp_path

    build_tree(tiles, tile_info_dir, save=True)

    # Check that file was created
    tree_file = tile_info_dir / 'kdtree_xyz.pkl'
    assert tree_file.exists()

    # Verify the tree is valid
    with open(tree_file, 'rb') as f:
        tree = pickle.load(f)

    assert isinstance(tree, cKDTree)
    assert tree.n == 3  # Should have 3 points (one per tile)


def test_build_tree_no_save(tmp_path: Path):
    """Test that build_tree with save=False doesn't create file."""

    tiles = [(217, 292), (234, 295)]
    tile_info_dir = tmp_path

    build_tree(tiles, tile_info_dir, save=False)

    # File should not exist
    tree_file = tile_info_dir / 'kdtree_xyz.pkl'
    assert not tree_file.exists()


def test_build_tree_empty_tiles(tmp_path: Path):
    """Test build_tree with empty tile list."""

    tiles = []
    tile_info_dir = tmp_path

    # Should handle empty list gracefully (or raise appropriate error)
    with pytest.raises(ValueError, match='Cannot build tree from empty tile list'):
        build_tree(tiles, tile_info_dir, save=True)


# ========== find_tile Tests ==========


def test_find_tile_exact_match():
    """Test finding tile when coordinate is exactly at tile center."""

    tiles = [(217, 292), (234, 295), (240, 300)]

    # Build the tree properly using actual tile coordinates
    tile_coords = np.array([relate_coord_tile(nums=tile) for tile in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])  # type: ignore
    tree = cKDTree(tile_coords_xyz)

    # Query for exact tile center (convert tile numbers to coordinates)
    ra, dec = relate_coord_tile(nums=(217, 292))
    object_coord = np.array([ra, dec])

    result_tile, distance = find_tile(tree, tiles, object_coord)

    assert result_tile == (217, 292)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_find_tile_coordinate_not_in_footprint():
    """Test that find_tile raises ValueError when coordinate is not in any tile."""

    tiles = [(217, 292), (218, 292), (217, 293)]

    # Build tree properly
    tile_coords = np.array([relate_coord_tile(nums=tile) for tile in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])  # type: ignore
    tree = cKDTree(tile_coords_xyz)

    # Query for coordinate very far away (actual sky coordinates)
    object_coord = np.array([0.0, 0.0])  # This is fine - real RA/Dec coordinates

    with pytest.raises(
        ValueError,
        match=f'Object {object_coord[0]} {object_coord[1]} could not be assigned to a tile.',
    ):
        find_tile(tree, tiles, object_coord)


def test_find_tile_multiple_candidates():
    """Test find_tile when multiple tiles are close but only one contains the coordinate."""

    # Three tiles close together
    tiles = [(217, 292), (218, 292), (217, 293)]

    # Build tree properly
    tile_coords = np.array([relate_coord_tile(nums=tile) for tile in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])  # type: ignore
    tree = cKDTree(tile_coords_xyz)

    # Get coordinates of first tile and offset slightly
    ra, dec = relate_coord_tile(nums=(217, 292))
    object_coord = np.array([ra + 0.1, dec + 0.1])

    result_tile, distance = find_tile(tree, tiles, object_coord)

    # Should find one of the tiles (exact result depends on footprint)
    assert result_tile in tiles
    assert distance >= 0


# ========== query_tree Tests ==========


def test_query_tree_success(tmp_path: Path, mocker):
    """Test successful query_tree call."""

    tiles = [(217, 292), (234, 295)]

    # Get actual coordinates for the first tile
    ra, dec = relate_coord_tile(nums=(217, 292))
    coords = np.array([ra, dec])
    tile_info_dir = tmp_path

    # Build tree properly
    tile_coords = np.array([relate_coord_tile(nums=tile) for tile in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])  # type: ignore
    tree = cKDTree(tile_coords_xyz)

    # Save the tree
    with open(tile_info_dir / 'kdtree_xyz.pkl', 'wb') as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Mock find_tile to return predictable result
    mocker.patch('unionsdata.kd_tree.find_tile', return_value=((217, 292), 0.5))

    result_tile, distance = query_tree(tiles, coords, tile_info_dir)

    assert result_tile == (217, 292)
    assert distance == 0.5


def test_query_tree_propagates_value_error(tmp_path: Path, mocker):
    """Test that query_tree propagates ValueError from find_tile."""

    tiles = [(217, 292)]
    coords = np.array([0.0, 0.0])  # Far away coordinates
    tile_info_dir = tmp_path

    # Build tree properly
    tile_coords = np.array([relate_coord_tile(nums=tile) for tile in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])  # type: ignore
    tree = cKDTree(tile_coords_xyz)

    with open(tile_info_dir / 'kdtree_xyz.pkl', 'wb') as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Mock find_tile to raise ValueError
    mocker.patch(
        'unionsdata.kd_tree.find_tile',
        side_effect=ValueError('Object could not be assigned to a tile.'),
    )

    with pytest.raises(ValueError, match='could not be assigned to a tile'):
        query_tree(tiles, coords, tile_info_dir)
