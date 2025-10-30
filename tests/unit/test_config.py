from pathlib import Path

import pytest
from pydantic_core import ValidationError

from unionsdata.config import get_user_data_dir, is_first_run, load_settings

# A minimal valid config content
MINIMAL_CONFIG = """
machine: local
logging:
  name: test
  level: INFO
runtime:
  n_download_threads: 4
  bands: ["whigs-g", "cfis_lsb-r"]
  resume: false
tiles:
  update_tiles: false
  show_tile_statistics: true
  band_constraint: 1

inputs:
  source: "tiles"

  tiles:
    - [217, 292]
    - [234, 295]

  coordinates:
    - [227.3042, 52.5285]
    - [231.4445, 52.4447]

  dataframe:
    path: "/home/nick/astro/unionsdata/tables/M101_lsb.csv"
    columns:
      ra: "ra"
      dec: "dec"
      id: "ID"

paths_database:
  tile_info_dirname: "tile_info"
  logs_dirname:      "logs"
paths_by_machine:
  local:
    root_dir_main: "/test/main"
    root_dir_data: "/test/data"

  canfar:
    root_dir_main: "/arc/test/main"
    root_dir_data: "/arc/test/data"

  narval:
    root_dir_main: "/home/user/projects/profile/main"
    root_dir_data: "/home/user/projects/profile/data"

bands:
  cfis-u:
    name: "CFIS"
    band: "u"
    vos: "vos:cfis/tiles_DR5"
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

  wishes-z:
    name: "WISHES"
    band: "z"
    vos: "vos:cfis/wishes_1/coadd"
    suffix: ".z.fits"
    delimiter: "."
    fits_ext: 1
    zfill: 0
    zp: 27.0

  ps-z:
    name: "PSS.DR4"
    band: "ps-z"
    vos: "vos:cfis/panstarrs/DR4/resamp"
    suffix: ".z.fits"
    delimiter: "."
    fits_ext: 0
    zfill: 3
    zp: 30.0
"""


def test_load_settings_basic(tmp_path: Path, mocker):
    """Test that a valid config file is loaded correctly."""

    config_file = tmp_path / 'config.yaml'
    config_file.write_text(MINIMAL_CONFIG)

    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    settings = load_settings(config_path=config_file)

    # Test multiple sections
    assert settings.machine == 'local'
    assert settings.logging.name == 'test'
    assert settings.logging.level == 'INFO'
    assert settings.runtime.bands == ['whigs-g', 'cfis_lsb-r']
    assert settings.runtime.n_download_threads == 4
    assert settings.runtime.resume is False
    assert settings.tiles.band_constraint == 1
    assert settings.paths.root_dir_data == Path('/test/data')
    assert settings.paths.root_dir_main == Path('/test/main')
    assert len(settings.bands) == 6


def test_load_settings_cli_override(tmp_path: Path, mocker):
    """Test that CLI overrides correctly replace config file values."""

    config_file = tmp_path / 'config.yaml'
    config_file.write_text(MINIMAL_CONFIG)

    # CLI overrides
    overrides = {
        'bands': ['ps-i', 'wishes-z'],
        'input_source': 'dataframe',
        'coordinates': [(11.1111, 22.2222), (33.3333, 44.4444)],
        'update_tiles': True,
    }

    # mock is_first_run to return False
    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    settings = load_settings(config_path=config_file, cli_overrides=overrides)

    # Check that the override took effect
    assert settings.runtime.bands == ['ps-i', 'wishes-z']
    # Since we simulated passing coordinates on the CLI the input type
    # should be changed to 'coordinate' and override the previously set 'dataframe'
    assert settings.inputs.source == 'coordinates'
    assert settings.inputs.coordinates == [(11.1111, 22.2222), (33.3333, 44.4444)]
    assert settings.tiles.update_tiles is True

    # Check that other fields are NOT affected
    assert settings.runtime.n_download_threads == 4
    assert settings.machine == 'local'
    assert settings.paths.root_dir_data == Path('/test/data')
    assert settings.paths.root_dir_main == Path('/test/main')


def test_load_settings_invalid_machine(tmp_path: Path, mocker) -> None:
    """Test that invalid machine name raises appropriate error."""

    config_file = tmp_path / 'config.yaml'
    invalid_config = MINIMAL_CONFIG.replace('machine: local', 'machine: invalid_machine')
    config_file.write_text(invalid_config)

    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    available = ', '.join(['local', 'canfar', 'narval'])
    with pytest.raises(
        ValueError,
        match=f'Machine "invalid_machine" not found in paths_by_machine. Available machines: {available}',
    ):
        load_settings(config_path=config_file)


def test_load_settings_missing_required_field(tmp_path: Path, mocker) -> None:
    """Test that missing required fields raise validation errors."""

    config_file = tmp_path / 'config.yaml'
    # Remove required field
    incomplete_config = MINIMAL_CONFIG.replace('machine: local', '')
    config_file.write_text(incomplete_config)

    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    with pytest.raises(ValidationError):
        load_settings(config_path=config_file)


def test_load_settings_invalid_band(tmp_path: Path, mocker):
    """Test that invalid band name raises appropriate error."""

    config_file = tmp_path / 'config.yaml'
    config_file.write_text(MINIMAL_CONFIG)

    overrides = {'bands': ['invalid-band']}

    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    with pytest.raises(ValueError, match='Unknown band in runtime.bands: invalid-band'):
        load_settings(config_path=config_file, cli_overrides=overrides)


def test_load_settings_path_resolution(tmp_path: Path, mocker):
    """Test that paths are correctly resolved for different machines."""

    config_file = tmp_path / 'config.yaml'
    # Test with 'canfar' machine
    canfar_config = MINIMAL_CONFIG.replace('machine: local', 'machine: canfar')
    config_file.write_text(canfar_config)

    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    settings = load_settings(config_path=config_file)

    assert settings.machine == 'canfar'
    assert settings.paths.root_dir_main == Path('/arc/test/main')
    assert settings.paths.root_dir_data == Path('/arc/test/data')
    assert settings.paths.log_directory == Path('/arc/test/main/logs')
    assert settings.paths.tile_info_directory == Path('/arc/test/main/tile_info')


def test_load_settings_input_source_validation(tmp_path: Path, mocker):
    """Test that input source is validated correctly."""

    config_file = tmp_path / 'config.yaml'
    invalid_config = MINIMAL_CONFIG.replace('source: "tiles"', 'source: "invalid_source"')
    config_file.write_text(invalid_config)

    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    with pytest.raises(ValidationError):
        load_settings(config_path=config_file)


def test_load_settings_empty_bands_list(tmp_path: Path, mocker):
    """Test behavior with empty bands list."""

    config_file = tmp_path / 'config.yaml'
    config_file.write_text(MINIMAL_CONFIG)

    overrides = {'bands': []}

    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    with pytest.raises(ValidationError):
        load_settings(config_path=config_file, cli_overrides=overrides)


def test_is_first_run_missing_directory(tmp_path: Path):
    """Test first run detection when directory doesn't exist."""

    non_existent_dir = tmp_path / 'does_not_exist'
    assert is_first_run(non_existent_dir) is True


def test_is_first_run_missing_files(tmp_path: Path):
    """Test first run detection when directory exists but files are missing."""

    tile_dir = tmp_path / 'tile_info'
    tile_dir.mkdir()

    # Directory exists but no files
    assert is_first_run(tile_dir) is True


def test_is_first_run_has_files(tmp_path: Path):
    """Test first run detection when all required files exist."""

    tile_dir = tmp_path / 'tile_info'
    tile_dir.mkdir()

    # Create dummy files
    (tile_dir / 'whigs-g_tiles.txt').write_text('217_292\n')
    (tile_dir / 'kdtree_xyz.joblib').write_text('dummy')

    assert is_first_run(tile_dir) is False


def test_load_settings_xdg_data_dir_usage(tmp_path: Path, mocker):
    """Test that XDG data directory is used for pip installs."""

    config_file = tmp_path / 'config.yaml'
    # Use home directory as root_dir_main to trigger XDG path logic
    home_config = MINIMAL_CONFIG.replace(
        'root_dir_main: "/test/main"', f'root_dir_main: "{Path.home()}"'
    )
    config_file.write_text(home_config)

    mocker.patch('unionsdata.config.is_first_run', return_value=False)
    # Mock to simulate non-editable install
    mocker.patch('unionsdata.config.distribution', side_effect=Exception('Not editable'))

    settings = load_settings(config_path=config_file)

    # Should use XDG data directory, not the config-specified path

    expected_data_dir = get_user_data_dir()

    assert settings.paths.root_dir_main == expected_data_dir
    assert settings.paths.tile_info_directory == expected_data_dir / 'tile_info'
