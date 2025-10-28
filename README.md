# UNIONSdata

A Python package for downloading multi-band imaging data from the Ultraviolet Near Infrared Optical Northern Survey ([UNIONS](https://www.skysurvey.cc/)). The package downloads the reduced images from the VOS vault.

## Overview

UNIONSdata provides a streamlined interface for downloading reduced images from the UNIONS survey hosted on the CANFAR VOSpace vault. The package supports multi-threaded downloads, automatic tile discovery, and flexible input methods (coordinates, tile numbers, or CSV catalogs).

### Supported bands

- **cfis-u**: CFIS u-band
- **whigs-g**: WHIGS g-band
- **cfis_lsb-r**: CFIS r-band (optimized for low-surface-brightness science)
- **ps-i**: Pan-STARRS i-band
- **wishes-z**: WISHES z-band
- **ps-z**: Pan-STARRS z-band

## Features

✨ **Multi-threaded downloads** - Parallel downloading for improved performance\
🎯 **Flexible input methods** - Use coordinates, tile numbers, or CSV catalogs\
🌳 **Spatial indexing** - KD-tree for efficient tile-to-coordinate matching\
📊 **Progress tracking** - Real-time download status and completion reports\
⚙️ **Configuration validation** - Pydantic-based config with clear error messages\
🛡️ **Graceful shutdown** - Clean interrupt handling with temp file cleanup\

## Prerequisites

1. **CANFAR VOSpace Account**\
    Register at https://www.canfar.net/en/

2. **UNIONS survey membership**\
    Until the first public data release only collaboration members have access to the data.

2. **Python 3.13.1+**

3. **System Dependencies**
   - Astropy 7.1.0+
   - Common scientific packages (see pyproject.toml)
   - Valid X.509 certificate for VOSpace access

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/heesters-nick/unionsdata.git
cd unionsdata

# Install in development mode
pip install -e ".[dev]"
```

### Using pip (coming soon)

```bash
pip install unionsdata
```

## Setup

### 1. Obtain CANFAR Credentials

Generate your X.509 certificate for VOSpace access:

```bash
cadc-get-cert -u YOUR_CANFAR_USERNAME
```
Type your CANFAR password. This creates a certificate at `~/.ssl/cadcproxy.pem` that's valid for 10 days.

### 2. Configure Download Settings

Edit the configuration file in

```bash
configs/download_config.yaml
```

Edit the paths for your machine:

```yaml
machine: local

paths_by_machine:
  local:
    root_dir_main: "/path/to/your/project"
    root_dir_data: "/path/to/download/data"
```

## Usage

### Command Line Interface

The package installs a `udownload` command for easy access.

#### Download Specific Tiles

```bash
# Download tiles by tile numbers (x, y pairs)
udownload --tiles 217 292 234 295

# Download specific bands only
udownload --tiles 217 292 --bands whigs-g cfis_lsb-r ps-i
```

#### Download by Coordinates

```bash
# Download tiles containing these RA/Dec coordinates
udownload --coordinates 227.3042 52.5285 231.4445 52.4447
```

#### Download from CSV Catalog

```bash
# Use a CSV file with RA, Dec, and ID columns
udownload --dataframe /path/to/catalog.csv
```

Your CSV should have columns for RA, Dec, and object ID (customizable in config):

```csv
ID,ra,dec
1,227.3042,52.5285
2,231.4445,52.4447
```

#### Download All Available Tiles

```bash
# ⚠️ Warning: This will download a large amount of data!
udownload --all-tiles --bands whigs-g cfis_lsb-r
```

### Configuration File Usage

Alternatively, configure inputs in `configs/download_config.yaml`:

```yaml
runtime:
  n_download_threads: 12
  bands: ["whigs-g", "cfis_lsb-r", "ps-i"]
  resume: false

inputs:
  source: "tiles"  # or "coordinates", "dataframe", "all_available"
  tiles:
    - [217, 292]
    - [234, 295]

  coordinates:
    - [227.3042, 52.5285]

  dataframe:
    path: "/path/to/catalog.csv"
    # Specify the column names for RA/Dec/ID in your table
    columns:
      ra: "ra"
      dec: "dec"
      id: "ID"
```

Then run:

```bash
udownload
```

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

## Acknowledgments

- UNIONS collaboration
- CANFAR (Canadian Advanced Network for Astronomical Research)

## Links

- [**UNIONS Survey**](http://www.skysurvey.cc/)
- [**CANFAR**](https://www.canfar.net/)
- [**CANFAR Storage Documentation**](https://www.opencadc.org/canfar/latest/platform/storage/)
- [**CANFAR VOSpace Documentation**](https://www.opencadc.org/canfar/latest/platform/storage/vospace/)

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: [nick.heesters@epfl.ch]

---
