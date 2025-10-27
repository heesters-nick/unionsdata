import pytest

from unionsdata.utils import get_tile_numbers


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
