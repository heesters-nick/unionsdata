import argparse
from pathlib import Path

import pytest

from unionsdata.main import build_cli_overrides, run_init


def test_run_init_creates_file(tmp_path: Path, mocker):
    """Test that 'run_init' creates a new config file."""

    # ARRANGE
    # Mock get_user_config_dir to use our temp directory
    mocker.patch('unionsdata.main.get_user_config_dir', return_value=tmp_path)

    # Mock the files() chain
    mock_template_path = mocker.MagicMock()
    mock_template_path.read_text.return_value = 'fake config content'

    mock_files_result = mocker.MagicMock()
    mock_files_result.joinpath.return_value = mock_template_path

    mocker.patch('unionsdata.main.files', return_value=mock_files_result)

    # Create the fake arguments
    args = argparse.Namespace(force=False)

    # Define the path where we expect the file to be created
    config_path = tmp_path / 'config.yaml'

    # Prove the file doesn't exist yet
    assert not config_path.exists()

    # ACT
    run_init(args)

    # ASSERT
    # Did it create the file?
    assert config_path.exists()

    # Did it write the content from our fake template?
    assert config_path.read_text() == 'fake config content'


def test_build_cli_overrides_tiles():
    """Test that tile arguments are correctly converted to override format."""

    # ARRANGE
    args = argparse.Namespace(
        tiles=[217, 292, 234, 295],
        bands=['whigs-g', 'cfis_lsb-r'],
        update_tiles=False,
        coordinates=None,
        table=None,
        all_tiles=False,
    )

    # ACT
    overrides = build_cli_overrides(args)

    # ASSERT
    assert overrides['tiles'] == [[217, 292], [234, 295]]
    assert overrides['bands'] == ['whigs-g', 'cfis_lsb-r']
    assert 'update_tiles' not in overrides  # False, so not included


def test_build_cli_overrides_table():
    """Test that table path is correctly added to overrides."""

    # ARRANGE
    args = argparse.Namespace(
        table='/path/to/catalog.csv',
        bands=['ps-i'],
        update_tiles=False,
        tiles=None,
        coordinates=None,
        all_tiles=False,
    )

    # ACT
    overrides = build_cli_overrides(args)

    # ASSERT
    assert overrides['table'] == '/path/to/catalog.csv'
    assert overrides['bands'] == ['ps-i']


def test_build_cli_overrides_odd_number_tiles_raises_error():
    """Test that odd number of tile arguments raises ValueError."""

    # ARRANGE
    args = argparse.Namespace(
        tiles=[217, 292, 234],  # Odd number!
        bands=None,
        update_tiles=False,
        coordinates=None,
        table=None,
        all_tiles=False,
    )

    # ACT & ASSERT
    from unionsdata.main import build_cli_overrides

    with pytest.raises(ValueError, match='Tiles must be provided as pairs'):
        build_cli_overrides(args)


def test_build_cli_overrides_odd_number_coordinates_raises_error():
    """Test that odd number of coordinate arguments raises ValueError."""

    # ARRANGE
    args = argparse.Namespace(
        coordinates=[227.3042, 52.5285, 231.4445],  # Odd number!
        bands=None,
        update_tiles=False,
        tiles=None,
        table=None,
        all_tiles=False,
    )

    # ACT & ASSERT
    with pytest.raises(ValueError, match='Coordinates must be provided as pairs'):
        build_cli_overrides(args)


def test_build_cli_overrides_empty_args():
    """Test that empty args returns empty overrides dict."""

    # ARRANGE
    args = argparse.Namespace(
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        bands=None,
        update_tiles=False,
    )

    # ACT
    overrides = build_cli_overrides(args)

    # ASSERT
    assert overrides == {}


def test_build_cli_overrides_no_input_source():
    """Test that only bands can be specified without input source."""

    # ARRANGE
    args = argparse.Namespace(
        bands=['whigs-g'],
        tiles=None,
        coordinates=None,
        table=None,
        all_tiles=False,
        update_tiles=False,
    )

    # ACT
    overrides = build_cli_overrides(args)

    # ASSERT
    assert overrides == {'bands': ['whigs-g']}
    assert 'tiles' not in overrides
    assert 'coordinates' not in overrides
    assert 'table' not in overrides
    assert 'all_tiles' not in overrides
    assert 'update_tiles' not in overrides
