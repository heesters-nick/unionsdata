import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
import yaml
from _pytest.logging import LogCaptureFixture
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree

from unionsdata.kd_tree import relate_coord_tile
from unionsdata.main import run_download


@pytest.fixture
def mock_vcp(mocker):
    """Mock the vcp subprocess to prevent actual downloads."""
    mock_popen = mocker.patch('unionsdata.download.subprocess.Popen')
    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Simulate successful completion
    mock_process.returncode = 0
    mock_process.communicate.return_value = ('', '')
    mock_popen.return_value = mock_process
    return mock_popen


@pytest.fixture
def mock_setup_logger(mocker):
    """Mock setup_logger to prevent it from reconfiguring logging during tests."""
    return mocker.patch('unionsdata.main.setup_logger')


@pytest.fixture
def test_config(tmp_path: Path) -> Path:
    """Create a minimal test config file."""
    config_content = f"""
machine: local

logging:
  name: integration_test
  level: INFO

runtime:
  n_download_threads: 2
  bands: ["whigs-g", "cfis_lsb-r", "ps-i"]
  resume: false

tiles:
  update_tiles: false
  show_tile_statistics: false
  band_constraint: 1

inputs:
  source: "tiles"
  tiles:
    - [217, 292]
    - [234, 295]
  coordinates: []
  dataframe:
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

bands:
  whigs-g:
    name: "calexp-CFIS"
    band: "g"
    vos: "vos:cfis/whigs/stack_images_CFIS_scheme"
    suffix: ".fits"
    delimiter: "_"
    fits_ext: 1
    zfill: 0
    zp: 27.0

  cfis_lsb-r:
    name: "CFIS_LSB"
    band: "r"
    vos: "vos:cfis/tiles_LSB_DR5"
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
def setup_tile_info(test_config: Path) -> None:
    """Set up fake tile info files and kdtree."""

    with open(test_config) as f:
        config_data = yaml.safe_load(f)

    root_dir = Path(config_data['paths_by_machine']['local']['root_dir_main'])
    tile_info_dirname = config_data['paths_database']['tile_info_dirname']
    tile_info_dir = root_dir / tile_info_dirname
    tile_info_dir.mkdir(parents=True, exist_ok=True)

    # Create fake tile list files
    (tile_info_dir / 'whigs-g_tiles.txt').write_text(
        'calexp-CFIS_217_292.fits\ncalexp-CFIS_234_295.fits\n'
    )
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
    joblib.dump(tree, tile_info_dir / 'kdtree_xyz.joblib')


def test_run_download_integration_basic(
    tmp_path: Path,
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test basic run_download workflow with tiles input."""

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        dataframe=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Check that expected steps occurred
    assert 'Starting UNIONS data download' in caplog.text
    assert 'Querying tile availability' in caplog.text
    assert 'Processing input to determine tiles to download' in caplog.text
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
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with band override from CLI."""

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        dataframe=None,
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
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with tile override from CLI."""

    args = argparse.Namespace(
        config=test_config,
        tiles=[217, 292],
        coordinates=None,
        dataframe=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should only download 1 tile × 3 bands = 3 calls
    assert mock_vcp.call_count == 3
    assert 'Filtering to 1 tiles based on input criteria' in caplog.text


def test_run_download_integration_coordinates(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download with coordinates input."""

    ra, dec = relate_coord_tile(nums=(217, 292))

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=[ra, dec],
        dataframe=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should download 1 tile × 3 bands = 3 calls
    assert mock_vcp.call_count == 3
    assert 'Filtering to 1 tiles based on input criteria' in caplog.text


def test_run_download_integration_no_tiles_to_download(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test run_download when no tiles match the criteria."""

    args = argparse.Namespace(
        config=test_config,
        tiles=[999, 999],
        coordinates=None,
        dataframe=None,
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
    (tile_dir / 'calexp-CFIS_217_292.fits').write_text('existing file')

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        dataframe=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should skip the existing file
    assert 'File calexp-CFIS_217_292.fits was already downloaded' in caplog.text
    assert mock_vcp.call_count == 5


def test_run_download_integration_band_constraint(
    tmp_path: Path,
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
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
        dataframe=None,
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
        dataframe=None,
        all_tiles=False,
        bands=['whigs-g'],
        update_tiles=False,
    )

    with caplog.at_level(logging.ERROR):
        run_download(args)

    # Assert - Should log failures
    assert 'Failed downloading tile' in caplog.text
    assert mock_popen.call_count == 2


def test_run_download_integration_timing(
    test_config: Path,
    setup_tile_info: None,
    mock_vcp: MagicMock,
    mock_setup_logger: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test that run_download reports timing information."""

    args = argparse.Namespace(
        config=test_config,
        tiles=None,
        coordinates=None,
        dataframe=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    with caplog.at_level(logging.INFO):
        run_download(args)

    # Assert - Should log execution time
    assert 'Done! Execution took' in caplog.text
    assert mock_vcp.call_count == 6
