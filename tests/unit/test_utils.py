import pytest

from unionsdata.config import BandDict
from unionsdata.utils import TileAvailability, get_tile_numbers

# Mock band dicts for testing purposes
MOCK_WHIGS_G: BandDict = {
    'name': 'calexp-CFIS',
    'band': 'g',
    'vos': 'vos:cfis/whigs/stack_images_CFIS_scheme',
    'suffix': '.fits',
    'delimiter': '_',
    'fits_ext': 1,
    'zfill': 0,
    'zp': 27.0,
}
MOCK_CFIS_LSB_R: BandDict = {
    'name': 'CFIS_LSB',
    'band': 'r',
    'vos': 'vos:cfis/tiles_LSB_DR5',
    'suffix': '.r.fits',
    'delimiter': '.',
    'fits_ext': 0,
    'zfill': 3,
    'zp': 30.0,
}
MOCK_PS_I: BandDict = {
    'name': 'PSS.DR4',
    'band': 'i',
    'vos': 'vos:cfis/panstarrs/DR4/resamp',
    'suffix': '.i.fits',
    'delimiter': '.',
    'fits_ext': 0,
    'zfill': 3,
    'zp': 30.0,
}

# A mock bands dictionary for testing purposes
MOCK_BANDS_DICT: dict[str, BandDict] = {
    'whigs-g': MOCK_WHIGS_G,
    'cfis_lsb-r': MOCK_CFIS_LSB_R,
    'ps-i': MOCK_PS_I,
}

# Mock tile lists.
# (1,1) is whigs-g only
# (1,2) is whigs-g, cfis_lsb-r
# (1,3) is cfis_lsb-r, ps-i
# (1,4) is all three
MOCK_TILE_LISTS: list[list[tuple[int, int]]] = [
    [(1, 1), (1, 2), (1, 4)],  # whigs-g tiles
    [(1, 2), (1, 3), (1, 4)],  # cfis_lsb-r tiles
    [(1, 3), (1, 4)],  # ps-i tiles
]


def test_get_tile_numbers_calexp():
    """Test get_tile_numbers for files with 'calexp' in the name."""
    filename = 'calexp-CFIS_234_295.fits'
    tile_numbers = get_tile_numbers(filename)
    assert tile_numbers == (234, 295)


@pytest.mark.parametrize(
    'filename, tile_numbers',
    [
        ('CFIS_LSB.234.295.r.fits', (234, 295)),
        ('CFIS.234.295.u.fits', (234, 295)),
        ('PSS.DR4.234.295.i.fits', (234, 295)),
        ('PS-DR3.234.295.i.fits', (234, 295)),
        ('WISHES.234.295.z.fits', (234, 295)),
    ],
)
def test_get_tile_numbers_no_calexp(filename, tile_numbers):
    """Test get_tile_numbers for files without 'calexp' in the name."""
    assert get_tile_numbers(filename) == tile_numbers


@pytest.mark.parametrize(
    'filename, tile_numbers',
    [
        ('CFIS_LSB.003.017.r.fits', (3, 17)),
        ('CFIS.020.024.u.fits', (20, 24)),
        ('PSS.DR4.011.005.i.fits', (11, 5)),
        ('PS-DR3.014.001.i.fits', (14, 1)),
    ],
)
def test_get_tile_numbers_with_zfill(filename, tile_numbers):
    """Test get_tile_numbers for zero-padded bands."""
    assert get_tile_numbers(filename) == tile_numbers


@pytest.mark.parametrize(
    'filename, tile_numbers',
    [
        ('WISHES.9.99.z.fits', (9, 99)),
        ('calexp-CFIS_7_12.fits', (7, 12)),
    ],
)
def test_get_tile_numbers_without_zfill(filename, tile_numbers):
    """Test get_tile_numbers for non-zero-padded bands."""
    assert get_tile_numbers(filename) == tile_numbers


@pytest.fixture
def availability() -> TileAvailability:
    """A reusable TileAvailability instance for tests."""
    return TileAvailability(MOCK_TILE_LISTS, MOCK_BANDS_DICT)


def test_availability_get_availability(availability: TileAvailability):
    """Test getting the availability for a single tile."""
    bands, _ = availability.get_availability((1, 1))
    assert sorted(bands) == ['whigs-g']

    bands, _ = availability.get_availability((1, 2))
    assert sorted(bands) == ['cfis_lsb-r', 'whigs-g']

    bands, _ = availability.get_availability((1, 3))
    assert sorted(bands) == ['cfis_lsb-r', 'ps-i']

    bands, _ = availability.get_availability((1, 4))
    assert sorted(bands) == ['cfis_lsb-r', 'ps-i', 'whigs-g']

    # Test a tile that doesn't exist
    bands, _ = availability.get_availability((9, 9))
    assert bands == []


def test_availability_get_tiles_for_bands(availability: TileAvailability):
    """Test getting all tiles that exist in a specific band or combination."""

    # Test for all unique tiles (bands=None)
    tiles = availability.get_tiles_for_bands(None)
    assert sorted(tiles) == [(1, 1), (1, 2), (1, 3), (1, 4)]

    # Test for a single band (string)
    tiles = availability.get_tiles_for_bands('whigs-g')
    assert sorted(tiles) == [(1, 1), (1, 2), (1, 4)]

    # Test for a single band (list)
    tiles = availability.get_tiles_for_bands(['ps-i'])
    assert sorted(tiles) == [(1, 3), (1, 4)]

    # Test for tiles in whigs-g AND ps-i
    tiles = availability.get_tiles_for_bands(['whigs-g', 'ps-i'])
    assert sorted(tiles) == [(1, 4)]

    # Test for tiles in all three bands
    tiles = availability.get_tiles_for_bands(['whigs-g', 'cfis_lsb-r', 'ps-i'])
    assert sorted(tiles) == [(1, 4)]

    # Test for tiles in cfis_lsb-r AND ps-i
    tiles = availability.get_tiles_for_bands(['cfis_lsb-r', 'ps-i'])
    assert sorted(tiles) == [(1, 3), (1, 4)]
