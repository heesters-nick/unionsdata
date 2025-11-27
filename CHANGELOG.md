# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `require_all_specified_bands` flag in config.yaml to specify if all specified bands need to be available for a tile to download it.
- download verification by checking if downloaded files have the expected size
- decompress compressed g-band and z-band files to speed up subsequent file loading times
- post-download cutout creation, store in hdf5 files
- retry logic for downloads
- plotting routine for cutouts (unionsdata plot)
- integrate config validation into unionsdata edit and remove unionsdata validate

## [0.1.1] - 2025-10-31

### Fixed
- Corrected documented Python requirement to Python >= 3.11.

## [0.1.0] - 2025-10-30

### Added
- Initial public release of `unionsdata`. This release provides a CLI and Python package to download UNIONS imaging data in a reproducible way, using a YAML configuration, XDG user directories, and a concurrent download engine.
- CLI and user interface
  - Top level CLI entry point `unionsdata`.
  - Subcommands for common tasks:
    - `unionsdata init` to copy the default config into the user config directory (for example `~/.config/unionsdata/config.yaml`).
    - `unionsdata edit` to open the user config in the system editor.
    - `unionsdata validate` to check the current config against the Pydantic model.
    - `unionsdata download` to actually download tiles or cutouts based on the selected input mode.
  - Global options for verbosity, config path override, and dry runs.
  - Consistent exit codes and structured logging so this can be used in automation.
- Configuration system:
  - YAML based configuration with a default file shipped inside the package.
  - XDG aware lookup, the application prefers a user specific config in `~/.config/unionsdata/` if present.
  - Pydantic models that validate paths, URIs and survey parameters at startup.
  - Support for per environment or per cluster overrides (local, CANFAR, Narval).
  - Resolution of relative paths to absolute ones at load time, so jobs run from anywhere.
  - Clear error messages when required tools for remote storage are missing.
- Download engine
  - Concurrent download workers that pull jobs from a queue.
  - Separate handling for tiles versus cutouts.
  - Tracking of download progress per tile and per band, so partial downloads can resume.
  - Atomic or temporary file writes, to avoid corrupt products when a download is interrupted.
  - Timeouts and a shutdown flag so the process can stop cleanly on SIGINT.
  - Logging of completed and failed downloads so users can re run only the missing ones.
- Core logic and inputs
  - Multiple input modes:
    - a list of UNIONS tile numbers,
    - a list of sky coordinates (RA, Dec),
    - a dataframe or table of objects with positions,
    - an option to download all tiles for a band.
  - Mapping from band names in UNIONS to the correct remote product names.
  - Support for the main UNIONS bands (CFIS, WHIGS, WISHES, Pan STARRS i and z), including the LSB r channel.
  - Basic validation that the requested bands are actually available for the selected input.
  - Hooks to integrate with VOSpace or CANFAR tooling if present.
- Development and CI/CD
  - Project configured with `hatch` for building and publishing.
  - GitHub Actions workflow to run tests and static checks on push and pull requests.
  - GitHub Actions workflow to publish to TestPyPI on pre release tags that contain `alpha`, `beta` or `rc`.
  - GitHub Actions workflow to publish to PyPI on normal version tags (for example `v0.1.0`).
  - A separate job to create a GitHub Release only after the PyPI publication has succeeded.

### Notes
- Actual runtime requirement: Python >= 3.11.
- The README uploaded to PyPI for this version mistakenly stated Python 3.13+. This was a documentation error and is corrected in 0.1.1.
- The CLI is intended to be stable from 0.1.x onward, but flag names may still change slightly as more back ends are added.

---

[Unreleased]: https://github.com/heesters-nick/unionsdata/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/heesters-nick/unionsdata/releases/tag/v0.1.1
[0.1.0]: https://github.com/heesters-nick/unionsdata/releases/tag/v0.1.0
