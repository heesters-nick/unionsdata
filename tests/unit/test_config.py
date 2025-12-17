from pathlib import Path

import pytest
from pydantic_core import ValidationError

from unionsdata.config import get_user_data_dir, is_first_run, load_settings


@pytest.fixture
def mock_cert_file(tmp_path: Path) -> Path:
    """Create a mock certificate file."""
    cert_file = tmp_path / 'cadcproxy.pem'
    cert_file.write_text('mock certificate')
    return cert_file


# A minimal valid config content
@pytest.fixture
def test_config(tmp_path: Path, mock_cert_file: Path) -> Path:
    """Create a minimal test config file."""
    config_content = f"""
machine: local

logging:
  name: test
  level: INFO

runtime:
  n_download_threads: 4
  n_cutout_processes: 2
  bands: ["whigs-g", "cfis_lsb-r"]
  resume: false
  max_retries: 3

tiles:
  update_tiles: false
  show_tile_statistics: true
  band_constraint: 1
  require_all_specified_bands: false

cutouts:
    enable: true
    cutouts_only: false
    size_pix: 256
    output_subdir: "cutouts"

plotting:
  enable: true
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
  logs_dirname: "logs"

paths_by_machine:
  local:
    root_dir_main: "/test/main"
    root_dir_data: "/test/data"
    dir_tables: "/test/tables"
    dir_figures: "/test/figures"
    cert_path: "{mock_cert_file}"

  canfar:
    root_dir_main: "/arc/test/main"
    root_dir_data: "/arc/test/data"
    dir_tables: "/arc/test/tables"
    dir_figures: "/arc/test/figures"
    cert_path: "{mock_cert_file}"

  narval:
    root_dir_main: "/home/user/projects/profile/main"
    root_dir_data: "/home/user/projects/profile/data"
    dir_tables: "/home/user/projects/profile/tables"
    dir_figures: "/home/user/projects/profile/figures"
    cert_path: "{mock_cert_file}"

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
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(config_content)
    return config_file


def test_load_settings_basic(test_config: Path) -> None:
    """Test that a valid config file is loaded correctly."""

    settings = load_settings(config_path=test_config, check_first_run=False)

    # Test multiple sections
    assert settings.machine == 'local'
    assert settings.logging.name == 'test'
    assert settings.logging.level == 'INFO'
    assert settings.runtime.bands == ['whigs-g', 'cfis_lsb-r']
    assert settings.runtime.n_download_threads == 4
    assert settings.runtime.resume is False
    assert settings.tiles.band_constraint == 1
    assert settings.tiles.require_all_specified_bands is False
    assert settings.paths.root_dir_data == Path('/test/data')
    assert settings.paths.root_dir_main == Path('/test/main')
    assert len(settings.bands) == 6


def test_load_settings_cli_override(tmp_path: Path, test_config: Path) -> None:
    """Test that CLI overrides correctly replace config file values."""

    # CLI overrides
    overrides = {
        'bands': ['ps-i', 'wishes-z'],
        'input_source': 'dataframe',
        'coordinates': [(11.1111, 22.2222), (33.3333, 44.4444)],
        'update_tiles': True,
    }

    settings = load_settings(
        config_path=test_config, cli_overrides=overrides, check_first_run=False
    )

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


def test_load_settings_missing_cert_file(test_config: Path, mock_cert_file: Path) -> None:
    """Test that missing certificate file raises appropriate error."""

    invalid_config = test_config.read_text()
    invalid_config = invalid_config.replace(
        f'cert_path: "{mock_cert_file}"', 'cert_path: "/non/existent/cert.pem"'
    )
    test_config.write_text(invalid_config)

    with pytest.raises(ValueError, match='Certificate file does not exist'):
        load_settings(config_path=test_config, check_first_run=False)


def test_load_settings_invalid_machine(tmp_path: Path, mocker, test_config: Path) -> None:
    """Test that invalid machine name raises appropriate error."""

    invalid_config = test_config.read_text()
    invalid_config = invalid_config.replace('machine: local', 'machine: invalid_machine')
    test_config.write_text(invalid_config)

    available = ', '.join(['local', 'canfar', 'narval'])
    with pytest.raises(
        ValueError,
        match=f'Machine "invalid_machine" not found in paths_by_machine. Available machines: {available}',
    ):
        load_settings(config_path=test_config, check_first_run=False)


def test_load_settings_missing_required_field(test_config: Path) -> None:
    """Test that missing required fields raise validation errors."""

    # Remove required field
    incomplete_config = test_config.read_text()
    incomplete_config = incomplete_config.replace('machine: local', '')
    test_config.write_text(incomplete_config)

    with pytest.raises(ValidationError):
        load_settings(config_path=test_config, check_first_run=False)


def test_load_settings_invalid_band(test_config: Path) -> None:
    """Test that invalid band name raises appropriate error."""

    overrides = {'bands': ['invalid-band']}

    with pytest.raises(ValueError, match='Unknown band in runtime.bands: invalid-band'):
        load_settings(config_path=test_config, cli_overrides=overrides, check_first_run=False)


def test_load_settings_path_resolution(test_config: Path) -> None:
    """Test that paths are correctly resolved for different machines."""

    canfar_config = test_config.read_text()
    canfar_config = canfar_config.replace('machine: local', 'machine: canfar')
    test_config.write_text(canfar_config)

    settings = load_settings(config_path=test_config, check_first_run=False)

    assert settings.machine == 'canfar'
    assert settings.paths.root_dir_main == Path('/arc/test/main')
    assert settings.paths.root_dir_data == Path('/arc/test/data')
    assert settings.paths.log_directory == Path('/arc/test/main/logs')
    assert settings.paths.tile_info_directory == Path('/arc/test/main/tile_info')


def test_load_settings_input_source_validation(test_config: Path) -> None:
    """Test that input source is validated correctly."""

    invalid_config = test_config.read_text()
    invalid_config = invalid_config.replace('source: "tiles"', 'source: "invalid_source"')
    test_config.write_text(invalid_config)

    with pytest.raises(ValidationError):
        load_settings(config_path=test_config, check_first_run=False)


def test_load_settings_empty_bands_list(test_config: Path) -> None:
    """Test behavior with empty bands list."""

    overrides = {'bands': []}

    with pytest.raises(ValidationError):
        load_settings(config_path=test_config, cli_overrides=overrides, check_first_run=False)


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
    (tile_dir / 'kdtree_xyz.pkl').write_text('dummy')

    assert is_first_run(tile_dir) is False


def test_load_settings_xdg_data_dir_usage(mocker, test_config: Path) -> None:
    """Test that XDG data directory is used for pip installs."""

    mocker.patch('unionsdata.config.determine_install_mode', return_value=False)
    mocker.patch('unionsdata.config.get_config_path', return_value=test_config)
    mocker.patch('unionsdata.config.is_first_run', return_value=False)

    settings = load_settings(config_path=test_config, check_first_run=False)

    # Should use XDG data directory, not the config-specified path
    expected_data_dir = get_user_data_dir()

    assert settings.paths.root_dir_main == expected_data_dir
    assert settings.paths.tile_info_directory == expected_data_dir / 'tile_info'
    assert settings.paths.log_directory == expected_data_dir / 'logs'
    # root_dir_data should still be from config
    assert settings.paths.root_dir_data == Path('/test/data')
