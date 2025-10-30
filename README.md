# UNIONSdata

[![CI](https://github.com/heesters-nick/unionsdata/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/heesters-nick/unionsdata/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

A Python package for downloading multi-band imaging data from the Ultraviolet Near Infrared Optical Northern Survey ([UNIONS](https://www.skysurvey.cc/)). The package downloads the reduced images from the CANFAR VOSpace vault using the [vos tool](https://pypi.org/project/vos/).

## Features

✨ **Multi-threaded downloads** - Parallel downloading for improved performance\
🎯 **Flexible input methods** - Use coordinates, tile numbers, or CSV catalogs\
🌳 **Spatial indexing** - KD-tree for efficient tile-to-coordinate matching\
📊 **Progress tracking** - Real-time download status and completion reports\
⚙️ **Configuration validation** - Pydantic-based config with clear error messages\
🛡️ **Graceful shutdown** - Clean interrupt handling with temp file cleanup

## Quick Start

```bash
# Install
pip install unionsdata

# Setup (opens config in editor)
unionsdata init
unionsdata edit  # Set your paths

# Get credentials (valid 10 days)
cadc-get-cert -u YOUR_CANFAR_USERNAME

# Download tiles
unionsdata download --tiles 217 292 --bands whigs-g cfis_lsb-r ps-i
```

## Prerequisites

1. **CANFAR VOSpace Account**\
    Register at https://www.canfar.net/en/

2. **UNIONS survey membership**\
    Until the first public data release only collaboration members have access to the data.

3. **Valid X.509 certificate for VOSpace access**\
    See below.

4. **System dependencies**\
    Installed automatically (see pyproject.toml)

## Installation & Setup

### Option 1: Install from PyPI (Recommended)

**Step 1:** Install the package

```bash
pip install unionsdata
```

**Step 2:** Initialize the configuration file

```bash
unionsdata init
```

This creates your configuration file at:
- **Linux/Mac**: `~/.config/unionsdata/config.yaml`
- **Windows**: `%APPDATA%/unionsdata/config.yaml`

**Step 3:** Edit the configuration

```bash
unionsdata edit
```

This opens the config file in your default editor. Update the paths and other parameters:

```yaml
machine: local

paths_by_machine:
  local:
    root_dir_main: "/path/to/your/project"
    # **Important**: define location for downloaded data
    root_dir_data: "/path/to/download/data"
```

**Step 4:** Set up CANFAR credentials

> 🔑 **Important:** Credentials expire after 10 days. Re-run this command when needed.

```bash
cadc-get-cert -u YOUR_CANFAR_USERNAME
```

**Step 5:** Validate your configuration

```bash
unionsdata validate
```

### Option 2: Install from Source (For Development)

**Step 1:** Clone and install

```bash
# Clone the repository
git clone https://github.com/heesters-nick/unionsdata.git

# Change into the cloned repository
cd unionsdata

# Install in editable development mode
pip install -e ".[dev]"
```

**Step 2:** Edit the configuration file directly at `src/unionsdata/config.yaml`

Update the paths:

```yaml
machine: local

paths_by_machine:
  local:
    root_dir_main: "/path/to/your/project"
    # **Important**: define location for downloaded data
    root_dir_data: "/path/to/download/data"
```

**Step 3:** Set up CANFAR credentials

```bash
cadc-get-cert -u YOUR_CANFAR_USERNAME
```

**Step 4:** Validate your configuration

```bash
unionsdata validate
```

## Usage

### Command Line Interface

The package provides a `unionsdata` command with several subcommands:

| Command | Description |
|---------|-------------|
| `unionsdata init` | Initialize configuration file (first-time setup) |
| `unionsdata edit` | Open configuration file in default editor |
| `unionsdata validate` | Validate your configuration |
| `unionsdata download` | Start downloading data |
| `unionsdata` | Shortcut for `unionsdata download` |



#### Download Specific Tiles

Download tiles by their tile numbers (x, y pairs):

```bash
unionsdata download --tiles 217 292 234 295
```

Download specific bands only:

```bash
unionsdata download --tiles 217 292 --bands whigs-g cfis_lsb-r ps-i
```

#### Download by Coordinates

Download tiles containing specific RA/Dec coordinates (in degrees):

```bash
unionsdata download --coordinates 227.3042 52.5285 231.4445 52.4447
```

#### Download from CSV Catalog

Download tiles for objects in a CSV file:

```bash
unionsdata download --dataframe /path/to/catalog.csv
```

Your CSV should have columns for RA, Dec, and object ID. Example:

```csv
ID,ra,dec
M101,210.8022,54.3489
2,231.4445,52.4447
```

> **Note:** Column names are customizable in the configuration file.

#### Download All Available Tiles

> **⚠️ Warning:** This will download a large amount of data!

```bash
unionsdata download --all-tiles --bands whigs-g cfis_lsb-r
```

### Using Configuration File

Instead of command-line arguments, you can configure downloads in your config file.

Example configuration:

```yaml
machine: local

logging:
  name: download_test
  level: INFO

runtime:
  n_download_threads: 12
  bands: ["whigs-g", "cfis_lsb-r", "ps-i"]
  resume: false

inputs:
  source: "tiles"  # Options: tiles, coordinates, dataframe, all_available
  tiles:
    - [217, 292]
    - [234, 295]

  coordinates:
    - [227.3042, 52.5285]

  dataframe:
    path: "/path/to/catalog.csv"
    columns:
      ra: "ra"
      dec: "dec"
      id: "ID"
```

Then run:

```bash
unionsdata download
```

Or simply:

```bash
unionsdata
```

## Supported Bands

| Band | Survey | Filter |
|------|--------|--------|
| `cfis-u` | CFIS | u-band |
| `whigs-g` | WHIGS | g-band |
| `cfis_lsb-r` | CFIS | r-band (LSB optimized) |
| `ps-i` | Pan-STARRS | i-band |
| `wishes-z` | WISHES | z-band |
| `ps-z` | Pan-STARRS | z-band |

## Output Structure

Downloaded files are organized by tile and band:

```
data/
├── 217_292/
│   ├── whigs-g/
│   │   └── calexp-CFIS_217_292.fits
│   ├── cfis_lsb-r/
│   │   └── CFIS_LSB.217.292.r.fits
│   └── ps-i/
│       └── PSS.DR4.217.292.i.fits
└── 234_295/
    └── ...
```

## Configuration Reference

### Key Configuration Options

| Section | Option | Description |
|---------|--------|-------------|
| `runtime` | `n_download_threads` | Number of parallel download threads (1-32) |
| `runtime` | `bands` | List of bands to download |
| `runtime` | `resume` | Overwrite existing log file if `false` |
| `tiles` | `update_tiles` | Refresh tile lists from VOSpace |
| `tiles` | `band_constraint` | Minimum bands required per tile |
| `inputs` | `source` | Input method: `tiles`, `coordinates`, `dataframe`, or `all_available` |

### Band Configuration

Each band has a specific file structure and location. Example for the WHIGS g-band:

```yaml
bands:
  whigs-g:
    name: "calexp-CFIS"
    band: "g"
    vos: "vos:cfis/whigs/stack_images_CFIS_scheme"
    suffix: ".fits"
    delimiter: "_"
    fits_ext: 1  # data extension in fits file
    zfill: 0  # No zero padding the tile numbers in the file name
    zp: 27.0  # Zero point magnitude
```

> **Note:** Data paths or file formats may change over time. Check the [CANFAR vault](https://www.canfar.net/storage/vault/list/cfis) for current locations:

| Band | vault directory |
|------|--------|
| `cfis-u` | tiles_DR6 |
| `whigs-g` | whigs |
| `cfis_lsb-r` | tiles_LSB_DR6 |
| `ps-i` | panstarrs |
| `wishes-z` | wishes_1 |
| `ps-z` | panstarrs |

## Troubleshooting

### Certificate Expired

```bash
cadc-get-cert -u YOUR_CANFAR_USERNAME
```

### Config Issues

```bash
# Check config
unionsdata validate
# Reset (Linux/Mac)
rm ~/.config/unionsdata/config.yaml
# Create a fresh copy
unionsdata init
```

## Acknowledgments

- UNIONS collaboration
- CANFAR (Canadian Advanced Network for Astronomical Research)

## Links

- [**UNIONS Survey**](http://www.skysurvey.cc/)
- [**CANFAR**](https://www.canfar.net/)
- [**CANFAR Storage Documentation**](https://www.opencadc.org/canfar/latest/platform/storage/)
- [**CANFAR VOSpace Documentation**](https://www.opencadc.org/canfar/latest/platform/storage/vospace/)
- [**vostools**](https://github.com/opencadc/vostools)

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: nick.heesters@epfl.ch

---
