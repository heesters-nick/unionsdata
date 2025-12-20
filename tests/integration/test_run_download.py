import argparse
import logging
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml
from _pytest.logging import LogCaptureFixture
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.spatial import cKDTree

from unionsdata.kd_tree import relate_coord_tile
from unionsdata.main import run_download


@pytest.fixture
def mock_vcp(mocker, tmp_path: Path):
    """Mock the vcp subprocess to prevent actual downloads AND create temp files."""

    def mock_popen_side_effect(*args, **kwargs):
        # Extract the output path from vcp command: ['vcp', '-v', 'vos:...', '/path/to/temp.fits']
        if len(args) > 0 and isinstance(args[0], list) and args[0][0] == 'vcp':
            output_path = Path(args[0][3])  # 4th argument is the output path

            # Create the temp file to simulate vcp download
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a VALID minimal FITS file (Primary + 1 Extension)
            # This passes is_fits_valid check for both fits_ext=0 and fits_ext=1
            hdul = fits.HDUList(
                [fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros((10, 10), dtype=np.float32))]
            )
            hdul.writeto(output_path, overwrite=True)

        # Return a mock process that indicates success
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_process.communicate.return_value = ('', '')
        return mock_process

    mock_popen = mocker.patch(
        'unionsdata.download.subprocess.Popen', side_effect=mock_popen_side_effect
    )
    return mock_popen


@pytest.fixture
def mock_setup_logger(mocker):
    """Mock setup_logger to prevent it from reconfiguring logging during tests."""
    return mocker.patch('unionsdata.main.setup_logger')


@pytest.fixture
def mock_cert_file(tmp_path: Path) -> Path:
    """Create a mock certificate file."""
    cert_file = tmp_path / 'cadcproxy.pem'
    cert_file.write_text('mock certificate')
    return cert_file


@pytest.fixture
def mock_session(mocker):
    """Mock requests session creation."""
    mock_session_obj = mocker.MagicMock()
    return mocker.patch('unionsdata.download.make_session', return_value=mock_session_obj)


@pytest.fixture
def mock_verify_download(mocker):
    """Mock download verification to always succeed."""
    return mocker.patch('unionsdata.download.verify_download', return_value=True)


@pytest.fixture
def mock_header_fetch(mocker):
    """Mock FITS header fetching and size calculation."""
    # Mock fetch_fits_header to return a minimal valid header
    mock_header = """SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -32 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                 2048
NAXIS2  =                 2048
END"""
    mocker.patch('unionsdata.download.fetch_fits_header', return_value=mock_header)

    # Mock get_file_size to return a plausible size
    mocker.patch('unionsdata.download.get_file_size', return_value=16_777_216)  # 16 MB

    return mocker


@pytest.fixture
def test_config(tmp_path: Path, mock_cert_file: Path) -> Path:
    """Create a minimal test config file."""
    config_content = f"""
machine: local

logging:
  name: integration_test
  level: INFO

runtime:
  n_download_threads: 2
  n_cutout_processes: 2
  bands: ["whigs-g", "cfis_lsb-r", "ps-i"]
  resume: false
  max_retries: 3

tiles:
  update_tiles: false
  show_tile_statistics: false
  band_constraint: 1
  require_all_specified_bands: false

cutouts:
    mode: "after_download"
    size_pix: 256
    output_subdir: "cutouts"

plotting:
  catalog_name: 'test_catalog'
  bands: ["whigs-g", 'cfis_lsb-r', 'ps-i']
  size_pix: 256
  mode: "grid"
  max_cols: 7
  figsize: null
  save_plot: true
  show_plot: false
  save_format: "pdf"

  rgb:
    scaling_type: "asinh"
    stretch: 125.0
    Q: 7.0
    gamma: 0.25
    standard_zp: 30.0

inputs:
  source: "tiles"
  tiles:
    - [217, 292]
    - [234, 295]
  coordinates: []
  table:
    path: ""
    columns:
      ra: "ra"
      dec: "dec"
      id: "ID"

paths_database:
  tile_info_dirname: "tile_info"
  logs_dirname: "logs"

paths_by_machine:
  local:
    root_dir_main: "{tmp_path}"
    root_dir_data: "{tmp_path / 'data'}"
    dir_tables: "{tmp_path / 'tables'}"
    dir_figures: "{tmp_path / 'figures'}"
    cert_path: "{mock_cert_file}"

bands:
  cfis-u:
    name: "CFIS"
    band: "u"
    vos: "vos:cfis/tiles_DR6"
    suffix: ".u.fits"
    delimiter: "."
    fits_ext: 0
    zfill: 3
    zp: 30.0

  whigs-g:
    name: "calexp-CFIS"
    band: "g"
    vos: "vos:cfis/whigs/stack_images_CFIS_scheme"
    suffix: ".fits"
    delimiter: "_"
    fits_ext: 1
    zfill: 0
    zp: 27.0

  cfis-r:
    name: "CFIS"
    band: "r"
    vos: "vos:cfis/tiles_DR6"
    suffix: ".r.fits"
    delimiter: "."
    fits_ext: 0
    zfill: 3
    zp: 30.0

  cfis_lsb-r:
    name: "CFIS_LSB"
    band: "r"
    vos: "vos:cfis/tiles_LSB_DR6"
    suffix: ".r.fits"
    delimiter: "."
    fits_ext: 0
    zfill: 3
    zp: 30.0

  ps-i:
    name: "PSS.DR4"
    band: "i"
    vos: "vos:cfis/panstarrs/DR4/resamp"
    suffix: ".i.fits"
    delimiter: "."
    fits_ext: 0
    zfill: 3
    zp: 30.0
"""
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_config_loading(mocker, test_config: Path):
    """Mock config loading functions for all integration tests."""
    mocker.patch('unionsdata.config.determine_install_mode', return_value=True)
    mocker.patch('unionsdata.config.get_config_path', return_value=test_config)
    mocker.patch('unionsdata.config.is_first_run', return_value=False)


@pytest.fixture
def setup_tile_info(test_config: Path) -> None:
    """Set up fake tile info files and kdtree."""

    with open(test_config) as f:
        config_data = yaml.safe_load(f)

    root_dir = Path(config_data['paths_by_machine']['local']['root_dir_main'])
    tile_info_dirname = config_data['paths_database']['tile_info_dirname']
    tile_info_dir = root_dir / tile_info_dirname
    tile_info_dir.mkdir(parents=True, exist_ok=True)

    # Create fake tile list files
    (tile_info_dir / 'cfis-u_tiles.txt').write_text('CFIS.217.292.u.fits\nCFIS.234.295.u.fits\n')
    (tile_info_dir / 'whigs-g_tiles.txt').write_text(
        'calexp-CFIS_217_292.fits\ncalexp-CFIS_234_295.fits\n'
    )
    (tile_info_dir / 'cfis-r_tiles.txt').write_text('CFIS.217.292.r.fits\nCFIS.234.295.r.fits\n')
    (tile_info_dir / 'cfis_lsb-r_tiles.txt').write_text(
        'CFIS_LSB.217.292.r.fits\nCFIS_LSB.234.295.r.fits\n'
    )
    (tile_info_dir / 'ps-i_tiles.txt').write_text(
        'PSS.DR4.217.292.i.fits\nPSS.DR4.234.295.i.fits\n'
    )

    # Create fake kdtree
    tiles = [(217, 292), (234, 295)]
    tile_coords = np.array([relate_coord_tile(nums=tile) for tile in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])  # type: ignore
    tree = cKDTree(tile_coords_xyz)
    with open(tile_info_dir / 'kdtree_xyz.pkl', 'wb') as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.fixture
def mock_decompress(mocker):
    """Mock FITS decompression to prevent funpack calls."""
    return mocker.patch('unionsdata.download.decompress_fits')


@pytest.fixture
def mock_cutout_creation(mocker):
    """Mock cutout creation with proper Future simulation."""
    from concurrent.futures import Future

    def mock_submit(fn, *args, **kwargs):
        """Mock submit that returns a completed Future with realistic value."""
        future = Future()

        # Extract catalog from kwargs to return realistic count
        catalog = kwargs.get('catalog')
        if catalog is not None and not catalog.empty:
            n_cutouts = len(catalog)
        else:
            n_cutouts = 0

        future.set_result((n_cutouts, 0, 0))  # Return actual catalog size
        return future

    mock_executor = mocker.MagicMock()
    mock_executor.submit.side_effect = mock_submit
    mock_executor.shutdown = mocker.MagicMock()

    mocker.patch('unionsdata.download.ProcessPoolExecutor', return_value=mock_executor)

    return mock_executor


@pytest.fixture
def setup_table_dir(tmp_path: Path, test_config: Path) -> Path:
    """Create tables directory for augmented catalog output."""
    with open(test_config) as f:
        config_data = yaml.safe_load(f)

    tables_dir = Path(config_data['paths_by_machine']['local']['dir_tables'])
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


@pytest.fixture
def test_table(tmp_path: Path) -> Path:
    """Create a test CSV file with coordinates."""
    csv_path = tmp_path / 'test_coords.csv'

    # Get coordinates for tiles we have in our mock setup
    coords_data = []
    for tile in [(217, 292), (234, 295)]:
        ra, dec = relate_coord_tile(nums=tile)
        coords_data.append({'ra': ra, 'dec': dec, 'ID': f'obj_{tile[0]}_{tile[1]}'})

    df = pd.DataFrame(coords_data)
    df.to_csv(csv_path, index=False)
    return csv_path


def test_run_download_integration_basic(
    tmp_path: Path,
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test basic run_download workflow with tiles input."""

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Check that expected steps occurred
    assert 'Total download jobs:' in caplog.text

    # Verify directory structure was created
    data_dir = tmp_path / 'data'
    assert data_dir.exists()
    assert (tmp_path / 'tile_info').exists()
    assert (tmp_path / 'logs').exists()

    # Verify download directories were created
    tile_217_292 = data_dir / '217_292'
    tile_234_295 = data_dir / '234_295'
    assert tile_217_292.exists()
    assert tile_234_295.exists()

    # Verify band subdirectories
    for tile_dir in [tile_217_292, tile_234_295]:
        assert (tile_dir / 'whigs-g').exists()
        assert (tile_dir / 'cfis_lsb-r').exists()
        assert (tile_dir / 'ps-i').exists()

    # Verify vcp was called for downloads (2 tiles × 3 bands = 6 calls)
    assert mock_vcp.call_count == 6


def test_run_download_integration_cli_override_bands(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with band override from CLI."""

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=['whigs-g'],
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should only download whigs-g band (2 tiles × 1 band = 2 calls)
    assert mock_vcp.call_count == 2
    assert 'whigs-g: 2 tiles' in caplog.text


def test_run_download_integration_cli_override_tiles(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with tile override from CLI."""

    args = argparse.Namespace(
        config=test_config,
        tiles=[217, 292],
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should only download 1 tile × 3 bands = 3 calls
    assert mock_vcp.call_count == 3


def test_run_download_integration_coordinates(
    test_config: Path,
    setup_tile_info: None,
    setup_table_dir: Path,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with coordinates input."""

    ra, dec = relate_coord_tile(nums=(217, 292))

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=[ra, dec],
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should download 1 tile × 3 bands = 3 calls
    assert mock_vcp.call_count == 3

    # Check that augmented catalog was saved
    catalog_path = setup_table_dir / 'input_coordinates_augmented.csv'
    assert catalog_path.exists(), 'Augmented catalog should be saved'

    # Validate catalog contents
    catalog = pd.read_csv(catalog_path)
    assert 'tile' in catalog.columns
    assert 'x' in catalog.columns
    assert 'y' in catalog.columns
    assert 'bands' in catalog.columns
    assert 'n_bands' in catalog.columns
    assert 'cutout_created' in catalog.columns
    assert len(catalog) == 1  # One coordinate
    assert catalog['cutout_created'].iloc[0] == 1  # Cutout was created


def test_run_download_integration_no_tiles_to_download(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download when no tiles match the criteria."""

    args = argparse.Namespace(
        config=test_config,
        tiles=[999, 999],
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.WARNING):
        run_download(args)

    # Assert
    assert 'No tiles found to download!' in caplog.text
    assert mock_vcp.call_count == 0


def test_run_download_integration_resume_mode(
    tmp_path: Path,
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    mocker,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with resume mode enabled."""

    # Modify config to enable resume
    config_content = test_config.read_text()
    config_content = config_content.replace('resume: false', 'resume: true')
    test_config.write_text(config_content)

    # Pre-create one of the files to simulate partial download
    data_dir = tmp_path / 'data'
    tile_dir = data_dir / '217_292' / 'whigs-g'
    tile_dir.mkdir(parents=True, exist_ok=True)
    existing_file = tile_dir / 'calexp-CFIS_217_292.fits'

    # Mock verify_download with updated signature (file_path, expected_size)
    def mock_verify(file_path: Path, expected_size: int | None) -> bool:
        # Pre-existing file: already verified
        if file_path == existing_file and file_path.exists():
            return True
        # Newly downloaded files: verify if they exist
        return file_path.exists()

    mocker.patch('unionsdata.download.verify_download', side_effect=mock_verify)

    # CHANGE: Create a VALID FITS file for the existing one
    # The previous version just wrote b'0' bytes, which would fail the new integrity check
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros((10, 10), dtype=np.float32))]
    )
    hdul.writeto(existing_file)

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should skip the existing file
    # assert 'already downloaded and verified' in caplog.text
    # Should download the remaining 5 files (2 tiles × 3 bands - 1 existing)
    assert mock_vcp.call_count == 5
    # Decompress should only be called for whigs-g downloads (1 tile)
    assert mock_decompress.call_count == 1


def test_run_download_integration_band_constraint(
    tmp_path: Path,
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with band_constraint filtering."""

    # Set band_constraint to 3
    config_content = test_config.read_text()
    config_content = config_content.replace('band_constraint: 1', 'band_constraint: 3')
    test_config.write_text(config_content)

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Both tiles should still qualify
    assert mock_vcp.call_count == 6
    assert 'band_constraint: 3' in test_config.read_text()


def test_run_download_integration_error_handling(
    test_config: Path,
    setup_tile_info: None,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    mocker,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download error handling when download fails."""

    # Mock vcp to fail
    mock_popen = mocker.patch('unionsdata.download.subprocess.Popen')
    mock_process = MagicMock()
    mock_process.poll.return_value = 1
    mock_process.returncode = 1
    mock_process.communicate.return_value = ('', 'Error: file not found')
    mock_popen.return_value = mock_process

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=['whigs-g'],
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should log failures
    assert 'Failed downloading tile' in caplog.text
    assert mock_popen.call_count == 6

    # Check that retries were logged
    assert 'Retrying' in caplog.text


def test_run_download_integration_timing(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test that run_download reports timing information."""

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should log execution time
    assert 'Done! Execution took' in caplog.text
    assert mock_vcp.call_count == 6


def test_run_download_integration_decompression_failure_retry(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_cutout_creation: MagicMock,
    mocker,
    caplog: LogCaptureFixture,
) -> None:
    """Test that decompression failures trigger download retry."""

    # Mock decompress_fits to fail on first attempt, succeed on second
    decompress_call_count = 0

    def mock_decompress_side_effect(path: Path) -> None:
        nonlocal decompress_call_count
        decompress_call_count += 1
        # Fail on 1st and 3rd calls (one for each tile's first attempt)
        if decompress_call_count in [1, 3]:
            raise RuntimeError('funpack failed: FITSIO status = 415')
        # Succeed on 2nd and 4th calls

    mocker.patch('unionsdata.download.decompress_fits', side_effect=mock_decompress_side_effect)

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=['whigs-g'],  # Compressed band
        update_tiles=False,
    )

    with caplog.at_level(logging.ERROR):
        run_download(args)

    # Assert - Should show retry behavior
    assert 'Failed to decompress' in caplog.text

    # Should have 2 download attempts per tile (first fails, second succeeds)
    # 2 tiles × 2 attempts = 4 vcp calls
    assert mock_vcp.call_count == 4

    # Decompression should have been called twice per tile (4 total)
    assert decompress_call_count == 4


def test_run_download_integration_table_with_cutouts(
    test_config: Path,
    setup_tile_info: None,
    setup_table_dir: Path,
    test_table: Path,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    mock_config_loading: None,
    mock_header_fetch: MagicMock,
    mock_verify_download: MagicMock,
    mock_session: MagicMock,
    mock_decompress: MagicMock,
    mock_cutout_creation: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with table input and cutout creation."""

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        table=str(test_table),
        all_tiles=False,
        bands=None,
        update_tiles=False,
        cutouts=True,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Should download 2 tiles × 3 bands = 6 calls
    assert mock_vcp.call_count == 6

    # Check that augmented catalog was saved with correct name
    catalog_path = setup_table_dir / 'test_coords_augmented.csv'
    assert catalog_path.exists(), 'Augmented catalog should be saved'

    # Validate catalog contents
    catalog = pd.read_csv(catalog_path)
    assert len(catalog) == 2  # Two objects
    assert 'cutout_created' in catalog.columns
    assert catalog['cutout_created'].sum() == 2  # Both cutouts created

    # Check augmented columns
    required_cols = ['ra', 'dec', 'ID', 'tile', 'x', 'y', 'bands', 'n_bands', 'cutout_created']
    for col in required_cols:
        assert col in catalog.columns, f'Missing column: {col}'

    # Log summary
    assert 'Saved augmented catalog' in caplog.text
