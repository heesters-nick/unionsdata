import logging
import math
import re
import warnings
from pathlib import Path
from typing import cast

from astropy.io import fits
from astropy.io.fits import BinTableHDU, ImageHDU, PrimaryHDU

# FITS format constants
FITS_BLOCK_SIZE = 2880  # bytes per FITS block
CARDS_PER_BLOCK = 36
BYTES_PER_CARD = 80

logger = logging.getLogger(__name__)


def is_fits_valid(file_path: Path, fits_ext: int) -> bool:
    """
    Check if a FITS file is valid. Fails if Astropy detects truncation.
    """
    try:
        # Catch warnings so we can fail on "File truncated"
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to be recorded
            warnings.simplefilter('always')

            with fits.open(file_path, checksum=False) as hdul:
                # Check 1: Did astropy warn us about truncation?
                for warning in w:
                    if 'truncated' in str(warning.message).lower():
                        logger.warning(f'File corrupted (truncated): {file_path.name}')
                        return False

                # Check 2: Access Primary Header
                primary = cast(PrimaryHDU, hdul[0])
                _ = primary.header

                # Check 3: Access Extension Header (if required)
                if fits_ext > 0:
                    if len(hdul) <= fits_ext:
                        logger.warning(f'File {file_path.name} missing extension {fits_ext}')
                        return False

                    ext_hdu = cast(ImageHDU | PrimaryHDU | BinTableHDU, hdul[fits_ext])
                    _ = ext_hdu.header

        return True

    except Exception as e:
        logger.warning(f'Integrity check failed for {file_path.name}: {e}')
        return False


def verify_download(
    file_path: Path,
    expected_file_size: int | None,
    tolerance: int = 28800,  # +- 10 FITS blocks
) -> bool:
    """
    Verify that downloaded file matches expected size from server.

    Args:
        file_path: Path to downloaded file
        expected_file_size: Expected file size in bytes inferred from header
        tolerance: Allowed size difference in bytes (default +- 10 FITS blocks)

    Returns:
        True if file size matches or size check unavailable, False if mismatch
    """
    if not file_path.exists():
        logger.warning(f'File does not exist: {file_path}')
        return False

    if expected_file_size is None:
        logger.warning(f'Could not verify size for {file_path.name} (no header info)')
        return True

    # Size of the downloaded file
    actual_file_size = file_path.stat().st_size

    if abs(actual_file_size - expected_file_size) <= tolerance:
        logger.debug(f'✓ Size verified for {file_path.name}: {actual_file_size:,} bytes')
        return True
    else:
        logger.warning(
            f'✗ Size mismatch for {file_path.name}: '
            f'expected {expected_file_size:,} bytes, got {actual_file_size:,} bytes '
            f'({actual_file_size / expected_file_size * 100:.1f}% complete)'
        )
        return False


def get_file_size(header_content: str) -> int:
    """
    Compute the decompressed on-disk FITS size (bytes) from header content.

    Handles three cases:
      1. Uncompressed primary HDU: size = header + data
      2. Tile-compressed HDU 1: size = primary_header + header + decompressed_data
      3. First IMAGE extension: size = primary_header + header + data

    Args:
        header_content: FITS header content as string (META=true format)

    Returns:
        Total decompressed size in bytes

    Raises:
        ValueError: If no HDUs found or no science image HDU can be located
    """
    lines = header_content.splitlines()
    hdus = _split_into_hdus(lines)

    if not hdus:
        raise ValueError('No HDUs found in header dump')

    primary = hdus[0]

    # Case 1: Uncompressed science in primary HDU
    if _is_primary_image(primary):
        return _calculate_hdu_size(primary, 'BITPIX', 'NAXIS', 'NAXIS{}')

    # Case 2: Tile-compressed science in HDU 1
    if len(hdus) > 1 and _is_tile_compressed(hdus[1]):
        return FITS_BLOCK_SIZE + _calculate_hdu_size(hdus[1], 'ZBITPIX', 'ZNAXIS', 'ZNAXIS{}')

    # Case 3: First uncompressed IMAGE extension
    for hdu in hdus[1:]:
        if _is_image_extension(hdu):
            return FITS_BLOCK_SIZE + _calculate_hdu_size(hdu, 'BITPIX', 'NAXIS', 'NAXIS{}')

    raise ValueError('Could not locate science image HDU')


def _split_into_hdus(lines: list[str]) -> list[list[str]]:
    """Split header lines into separate HDUs (split on END cards)."""
    hdus = []
    current_hdu = []

    for line in lines:
        current_hdu.append(line.rstrip())
        if line.strip().startswith('END'):
            hdus.append(current_hdu)
            current_hdu = []

    return hdus


def _get_card_value(cards: list[str], keyword: str, default: int = 0) -> int:
    """Extract integer value from FITS header cards."""
    keyword_prefix = keyword.ljust(8)

    for card in cards:
        if card.startswith(keyword_prefix):
            value_part = card.split('=', 1)[-1]
            match = re.search(r'[+\-]?\d+', value_part)
            return int(match.group()) if match else default

    return default


def _calculate_hdu_size(
    cards: list[str], bitpix_key: str, naxis_key: str, axis_template: str
) -> int:
    """Calculate total HDU size: header + padded data."""
    header_size = _header_size(cards)
    data_size = _data_size(cards, bitpix_key, naxis_key, axis_template)
    return header_size + data_size


def _header_size(cards: list[str]) -> int:
    """Calculate header size in bytes (always multiple of FITS_BLOCK_SIZE)."""
    num_blocks = math.ceil(len(cards) / CARDS_PER_BLOCK)
    return num_blocks * FITS_BLOCK_SIZE


def _data_size(cards: list[str], bitpix_key: str, naxis_key: str, axis_template: str) -> int:
    """Calculate data size in bytes, padded to FITS_BLOCK_SIZE."""
    bitpix = abs(_get_card_value(cards, bitpix_key))
    naxis = _get_card_value(cards, naxis_key)

    # Calculate total number of pixels
    num_pixels = 1
    for axis_num in range(1, naxis + 1):
        axis_key = axis_template.format(axis_num)
        num_pixels *= _get_card_value(cards, axis_key)

    bytes_per_pixel = bitpix // 8
    total_bytes = bytes_per_pixel * num_pixels

    return _pad_to_block_size(total_bytes)


def _pad_to_block_size(num_bytes: int) -> int:
    """Pad byte count to next FITS_BLOCK_SIZE boundary."""
    return math.ceil(num_bytes / FITS_BLOCK_SIZE) * FITS_BLOCK_SIZE


def _is_primary_image(cards: list[str]) -> bool:
    """Check if HDU is a primary image with data."""
    has_simple = any(card.startswith('SIMPLE  =') for card in cards)
    has_data = _get_card_value(cards, 'NAXIS', 0) > 0
    return has_simple and has_data


def _is_tile_compressed(cards: list[str]) -> bool:
    """Check if HDU is a tile-compressed BINTABLE."""
    is_bintable = any(card.startswith("XTENSION= 'BINTABLE") for card in cards)
    has_zbitpix = any(card.startswith('ZBITPIX') for card in cards)
    return is_bintable and has_zbitpix


def _is_image_extension(cards: list[str]) -> bool:
    """Check if HDU is an IMAGE extension."""
    return any(card.startswith("XTENSION= 'IMAGE") for card in cards)
