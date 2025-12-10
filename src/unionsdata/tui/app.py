"""Main Textual application for the configuration editor."""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from unionsdata.tui.validators import IntegerRange, NonEmptyValidator
from unionsdata.tui.widgets import (
    BandSelector,
    BetterCheckbox,
    CoordinateList,
    InfoIcon,
    PathInput,
    RGBBandSelector,
    TileList,
)

logger = logging.getLogger(__name__)


# Type alias for input sources
InputSource = Literal['all_available', 'tiles', 'coordinates', 'dataframe']
ButtonVariant = Literal['default', 'primary', 'success', 'warning', 'error']


class ConfirmDialog(ModalScreen[bool]):
    """A confirmation dialog that returns True/False."""

    DEFAULT_CSS = """
    ConfirmDialog {
        align: center middle;
    }

    ConfirmDialog > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        height: auto;
        max-height: 20;
    }

    ConfirmDialog .dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ConfirmDialog .dialog-message {
        margin-bottom: 2;
    }

    ConfirmDialog .dialog-buttons {
        layout: horizontal;
        align: center middle;
        height: 3;
    }

    ConfirmDialog .dialog-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        title: str,
        message: str,
        yes_label: str = 'Yes',
        no_label: str = 'No',
        yes_variant: ButtonVariant = 'error',
        no_variant: ButtonVariant = 'primary',
    ) -> None:
        super().__init__()
        self._title = title
        self._message = message
        self._yes_label = yes_label
        self._no_label = no_label
        self._yes_variant: ButtonVariant = yes_variant
        self._no_variant: ButtonVariant = no_variant

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, classes='dialog-title')
            yield Static(self._message, classes='dialog-message')
            with Horizontal(classes='dialog-buttons'):
                # ID mapping: yes -> True, no -> False
                yield Button(self._yes_label, variant=self._yes_variant, id='confirm-yes')
                yield Button(self._no_label, variant=self._no_variant, id='confirm-no')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'confirm-yes':
            self.dismiss(True)
        else:
            self.dismiss(False)


class ConfigEditorApp(App[None]):
    """Textual application for editing unionsdata configuration."""

    TITLE = 'UNIONS Data Configuration'
    SUB_TITLE = 'Interactive configuration editor'

    CSS_PATH = 'styles/config_editor.tcss'

    BINDINGS = [
        Binding('ctrl+s', 'save', 'Save & Quit', show=True),
        Binding('ctrl+q', 'quit_app', 'Quit', show=True),
        Binding('?', 'help', 'Help', show=True),
    ]

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        self._config_path = config_path
        self._config_data: dict[str, Any] = {}
        self._dirty = False
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        from unionsdata.config import get_user_config_dir

        # Determine config path
        if self._config_path is None:
            self._config_path = get_user_config_dir() / 'config.yaml'

        # Try to load existing config
        if self._config_path.exists():
            try:
                with open(self._config_path, encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f) or {}
                logger.debug(f'Loaded config from {self._config_path}')
            except Exception as e:
                logger.warning(f'Failed to load config: {e}, using defaults')
                self._config_data = {}
        else:
            # Load template from package
            try:
                template_path = files('unionsdata').joinpath('config.yaml')
                template_content = template_path.read_text(encoding='utf-8')
                self._config_data = yaml.safe_load(template_content) or {}
                logger.debug('Loaded default config template')
            except Exception as e:
                logger.warning(f'Failed to load template: {e}')
                self._config_data = self._get_minimal_defaults()

    def _get_minimal_defaults(self) -> dict[str, Any]:
        """Get minimal default configuration."""
        return {
            'machine': 'local',
            'logging': {'name': 'unionsdata', 'level': 'INFO'},
            'runtime': {
                'n_download_threads': 12,
                'n_cutout_processes': 2,
                'bands': ['cfis-r'],
                'resume': False,
                'max_retries': 5,
            },
            'tiles': {
                'update_tiles': False,
                'show_tile_statistics': True,
                'band_constraint': 1,
                'require_all_specified_bands': False,
            },
            'cutouts': {
                'enable': True,
                'cutouts_only': False,
                'size_pix': 512,
                'output_subdir': 'cutouts',
            },
            'plotting': {
                'catalog_name': 'catalog',
                'bands': None,
                'size_pix': 512,
                'mode': 'grid',
                'max_cols': 5,
                'figsize': None,
                'save_plot': True,
                'show_plot': False,
                'save_format': 'pdf',
                'rgb': {
                    'scaling_type': 'asinh',
                    'stretch': 125.0,
                    'Q': 7.0,
                    'gamma': 0.25,
                    'standard_zp': 30.0,
                },
            },
            'inputs': {
                'source': 'tiles',
                'tiles': [],
                'coordinates': [],
                'dataframe': {'path': '', 'columns': {'ra': 'ra', 'dec': 'dec', 'id': 'ID'}},
            },
            'paths_database': {'tile_info_dirname': 'tile_info', 'logs_dirname': 'logs'},
            'paths_by_machine': {
                'local': {
                    'root_dir_main': '',
                    'root_dir_data': '',
                    'dir_tables': '',
                    'dir_figures': '',
                    'cert_path': '',
                }
            },
            'bands': {},
        }

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # Escape brackets to prevent Rich markup interpretation
        path_str = str(self._config_path).replace('[', r'\[').replace(']', r'\]')
        yield Static(
            f'UNIONS Data Configuration ({path_str})',
            id='app-header',
        )

        with TabbedContent():
            with TabPane('General', id='general-tab'):
                yield from self._compose_general_tab()

            with TabPane('Paths', id='paths-tab'):
                yield from self._compose_paths_tab()

            with TabPane('Inputs', id='inputs-tab'):
                yield from self._compose_inputs_tab()

            with TabPane('Runtime', id='runtime-tab'):
                yield from self._compose_runtime_tab()

            with TabPane('Bands', id='bands-tab'):
                yield from self._compose_bands_tab()

            with TabPane('Tiles', id='tiles-tab'):
                yield from self._compose_tiles_tab()

            with TabPane('Cutouts', id='cutouts-tab'):
                yield from self._compose_cutouts_tab()

            with TabPane('Plotting', id='plotting-tab'):
                yield from self._compose_plotting_tab()

        with Horizontal(id='button-bar'):
            yield Button('💾 Save & Quit', variant='primary', id='save-btn')
            yield Button('✗ Cancel', variant='error', id='cancel-btn')

        yield Footer()

    def _compose_general_tab(self) -> ComposeResult:
        """Compose the General settings tab."""
        cfg = self._config_data

        with ScrollableContainer():
            yield Static('General Settings', classes='section-title')

            # Machine selection
            machines = list(cfg.get('paths_by_machine', {'local': {}}).keys())
            current_machine = cfg.get('machine', 'local')

            with Horizontal(classes='field-row'):
                yield Label('Machine:', classes='field-label')
                yield Select(
                    [(m, m) for m in machines],
                    value=current_machine,
                    id='machine-select',
                    classes='field-input',
                )

            yield Static('Logging', classes='section-title')

            # Logging name
            log_name = cfg.get('logging', {}).get('name', 'unionsdata')
            with Horizontal(classes='field-row'):
                yield Label('Log Name:', classes='field-label')
                yield Input(
                    value=log_name,
                    id='logging-name',
                    classes='field-input',
                    validators=[NonEmptyValidator()],
                )

            # Logging level
            log_level = cfg.get('logging', {}).get('level', 'INFO')
            levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            with Horizontal(classes='field-row'):
                yield Label('Log Level:', classes='field-label')
                yield Select(
                    [(level, level) for level in levels],
                    value=log_level,
                    id='logging-level',
                    classes='field-input',
                )

    def _compose_runtime_tab(self) -> ComposeResult:
        """Compose the Runtime settings tab."""
        runtime = self._config_data.get('runtime', {})

        with ScrollableContainer():
            yield Static('Runtime Configuration', classes='section-title')

            # Download threads
            with Horizontal(classes='field-row'):
                yield Label('Download Threads:', classes='field-label')
                yield Input(
                    value=str(runtime.get('n_download_threads', 12)),
                    id='n-download-threads',
                    classes='field-input',
                    validators=[IntegerRange(1, 32)],
                )

            # Cutout processes
            with Horizontal(classes='field-row'):
                yield Label('Cutout Processes:', classes='field-label')
                yield Input(
                    value=str(runtime.get('n_cutout_processes', 2)),
                    id='n-cutout-processes',
                    classes='field-input',
                    validators=[IntegerRange(1, 32)],
                )

            # Max retries
            with Horizontal(classes='field-row'):
                yield Label('Max Retries:', classes='field-label')
                yield Input(
                    value=str(runtime.get('max_retries', 5)),
                    id='max-retries',
                    classes='field-input',
                    validators=[IntegerRange(1, 10)],
                )

            # Resume
            with Horizontal(classes='field-row'):
                yield Label('Resume:', classes='field-label')
                yield BetterCheckbox(
                    'Continue from previous run',
                    runtime.get('resume', False),
                    id='resume-checkbox',
                    classes='field-checkbox',
                )

    def _compose_bands_tab(self) -> ComposeResult:
        """Compose the Bands settings tab."""
        # Get bands from runtime config
        runtime = self._config_data.get('runtime', {})

        with ScrollableContainer():
            with Horizontal(classes='section-title'):
                yield Label('Band Selection')
                yield InfoIcon('Select bands to download (at least one required)')

            # BandSelector moved here
            yield BandSelector(
                selected=runtime.get('bands', ['cfis-r']),
                min_selected=1,
                id='band-selector',
            )

    def _compose_tiles_tab(self) -> ComposeResult:
        """Compose the Tiles settings tab."""
        tiles = self._config_data.get('tiles', {})

        with ScrollableContainer():
            yield Static('Tile Configuration', classes='section-title')

            # Update tiles
            with Horizontal(classes='field-row'):
                yield Label('Update Tiles:', classes='field-label')
                yield BetterCheckbox(
                    'Fetch latest tile list from CANFAR',
                    tiles.get('update_tiles', False),
                    id='update-tiles-checkbox',
                    classes='field-checkbox',
                )

            # Show statistics
            with Horizontal(classes='field-row'):
                yield Label('Show Statistics:', classes='field-label')
                yield BetterCheckbox(
                    'Display tile availability statistics',
                    tiles.get('show_tile_statistics', True),
                    id='show-tile-stats-checkbox',
                    classes='field-checkbox',
                )

            # Band constraint
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Band Constraint')
                    yield InfoIcon(
                        'Minimum number of bands a tile must have to qualify for download'
                    )
                    yield Label(':')
                yield Input(
                    value=str(tiles.get('band_constraint', 1)),
                    id='band-constraint',
                    classes='field-input',
                    validators=[IntegerRange(1, 7)],
                )

            # Require all bands
            with Horizontal(classes='field-row'):
                yield Label('Require All Bands:', classes='field-label')
                yield BetterCheckbox(
                    'Only download tiles available in ALL specified bands',
                    tiles.get('require_all_specified_bands', False),
                    id='require-all-bands-checkbox',
                    classes='field-checkbox',
                )

    def _compose_cutouts_tab(self) -> ComposeResult:
        """Compose the Cutouts settings tab."""
        cutouts = self._config_data.get('cutouts', {})

        with ScrollableContainer():
            yield Static('Cutout Configuration', classes='section-title')

            # Enable cutouts
            with Horizontal(classes='field-row'):
                yield Label('Enable Cutouts:', classes='field-label')
                yield BetterCheckbox(
                    'Create cutouts after tile download',
                    cutouts.get('enable', True),
                    id='cutouts-enable-checkbox',
                    classes='field-checkbox',
                )

            # Cutouts only
            with Horizontal(classes='field-row'):
                yield Label('Cutouts Only:', classes='field-label')
                yield BetterCheckbox(
                    'Skip full tile download, only create cutouts',
                    cutouts.get('cutouts_only', False),
                    id='cutouts-only-checkbox',
                    classes='field-checkbox',
                )

            # Cutout size
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Size (pixels)')
                    yield InfoIcon('Square cutout size in pixels (pixel scale: 0.1857 arcsec/pix)')
                    yield Label(':')
                yield Input(
                    value=str(cutouts.get('size_pix', 512)),
                    id='cutout-size',
                    classes='field-input',
                    validators=[IntegerRange(1, 10000)],
                )

            # Output subdirectory
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Output Subdir')
                    yield InfoIcon('Subdirectory for cutout output files')
                    yield Label(':')
                yield Input(
                    value=cutouts.get('output_subdir', 'cutouts'),
                    id='cutout-subdir',
                    classes='field-input',
                    validators=[NonEmptyValidator()],
                )

    def _compose_plotting_tab(self) -> ComposeResult:
        """Compose the Plotting settings tab."""
        plotting = self._config_data.get('plotting', {})
        rgb = plotting.get('rgb', {})

        with ScrollableContainer():
            yield Static('Plotting Configuration', classes='section-title')

            # Catalog name
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Catalog Name')
                    yield InfoIcon('Name of catalog file (without _augmented.csv suffix)')
                    yield Label(':')
                yield Input(
                    value=plotting.get('catalog_name', 'catalog'),
                    id='plot-catalog-name',
                    classes='field-input',
                    validators=[NonEmptyValidator()],
                )

            # RGB Band Selection; none = will use first three runtime bands
            plot_bands = plotting.get('bands') or []

            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('RGB Bands')
                    yield InfoIcon(
                        'Select bands for RGB channels, or leave unset to use first 3 runtime bands. Wavelength order must be followed (Blue < Green < Red). Hit the reset button to clear selections and start over.'
                    )
                    yield Label(':')

                # New Widget Integration
                yield RGBBandSelector(selected_bands=plot_bands, id='rgb-band-selector')

            # Plot size
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Size (pixels)')
                    yield InfoIcon('Square cutout size in pixels')
                    yield Label(':')
                yield Input(
                    value=str(plotting.get('size_pix', 512)),
                    id='plot-size',
                    classes='field-input',
                    validators=[IntegerRange(1, 10000)],
                )

            # Mode
            mode = plotting.get('mode', 'grid')
            modes = [('Grid', 'grid'), ('Channel', 'channel')]
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Display Mode')
                    yield InfoIcon('grid: thumbnails; channel: show R,G,B separately')
                    yield Label(':')
                yield Select(
                    modes,
                    value=mode,
                    id='plot-mode',
                    classes='field-input',
                )

            # Max columns
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Max Columns')
                    yield InfoIcon('Maximum columns in grid mode')
                    yield Label(':')
                yield Input(
                    value=str(plotting.get('max_cols', 5)),
                    id='plot-max-cols',
                    classes='field-input',
                    validators=[IntegerRange(1, 20)],
                )

            # Save plot
            with Horizontal(classes='field-row'):
                yield Label('Save Plot:', classes='field-label')
                yield BetterCheckbox(
                    'Save plot to disk',
                    plotting.get('save_plot', True),
                    id='plot-save-checkbox',
                    classes='field-checkbox',
                )

            # Show plot
            with Horizontal(classes='field-row'):
                yield Label('Show Plot:', classes='field-label')
                yield BetterCheckbox(
                    'Display plot',
                    plotting.get('show_plot', False),
                    id='plot-show-checkbox',
                    classes='field-checkbox',
                )

            # Save format
            save_format = plotting.get('save_format', 'pdf')
            formats = [('PDF', 'pdf'), ('PNG', 'png'), ('JPG', 'jpg'), ('SVG', 'svg')]
            with Horizontal(classes='field-row'):
                yield Label('Save Format:', classes='field-label')
                yield Select(
                    formats,
                    value=save_format,
                    id='plot-save-format',
                    classes='field-input',
                )

            yield Static('RGB Scaling', classes='section-title')

            # Scaling type
            scaling_type = rgb.get('scaling_type', 'asinh')
            scaling_types = [('Asinh', 'asinh'), ('Linear', 'linear')]
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Scaling Type')
                    yield InfoIcon('asinh preserves both bright and faint details')
                    yield Label(':')
                yield Select(
                    scaling_types,
                    value=scaling_type,
                    id='rgb-scaling-type',
                    classes='field-input',
                )

            # Stretch
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Stretch')
                    yield InfoIcon('Scaling factor controlling overall brightness')
                    yield Label(':')
                yield Input(
                    value=str(rgb.get('stretch', 125.0)),
                    id='rgb-stretch',
                    classes='field-input',
                )

            # Q
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Q')
                    yield InfoIcon('Softening parameter for asinh (higher = more linear)')
                    yield Label(':')
                yield Input(
                    value=str(rgb.get('Q', 7.0)),
                    id='rgb-q',
                    classes='field-input',
                )

            # Gamma
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Gamma')
                    yield InfoIcon('Gamma correction (lower = enhances faint features)')
                    yield Label(':')
                yield Input(
                    value=str(rgb.get('gamma', 0.25)),
                    id='rgb-gamma',
                    classes='field-input',
                )

            # Standard ZP
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Standard ZP')
                    yield InfoIcon('Standard zero-point for flux normalization')
                    yield Label(':')
                yield Input(
                    value=str(rgb.get('standard_zp', 30.0)),
                    id='rgb-standard-zp',
                    classes='field-input',
                )

    def _compose_inputs_tab(self) -> ComposeResult:
        """Compose the Inputs settings tab."""
        inputs = self._config_data.get('inputs', {})

        with ScrollableContainer():
            yield Static('Input Source Configuration', classes='section-title')

            # Source selection
            sources: list[tuple[str, InputSource]] = [
                ('All Available Tiles', 'all_available'),
                ('Specific Tiles', 'tiles'),
                ('Sky Coordinates', 'coordinates'),
                ('DataFrame (CSV)', 'dataframe'),
            ]
            current_source = inputs.get('source', 'tiles')

            with Horizontal(classes='field-row'):
                yield Label('Input Source:', classes='field-label')
                yield Select(
                    sources,
                    value=current_source,
                    id='input-source-select',
                    classes='field-input',
                )

            # Tiles input section
            initial_tiles = [tuple(t) for t in inputs.get('tiles', [])]
            with Vertical(
                id='tiles-input-section',
                classes='' if current_source == 'tiles' else 'hidden',
            ):
                yield Static('Tile Numbers', classes='subsection-title')
                yield TileList(
                    tiles=cast(list[tuple[int, int]], initial_tiles),
                    id='tiles-list',
                )

            # Coordinates input section
            initial_coords = [tuple(c) for c in inputs.get('coordinates', [])]
            with Vertical(
                id='coordinates-input-section',
                classes='' if current_source == 'coordinates' else 'hidden',
            ):
                yield Static(r'Sky Coordinates (RA/Dec) \[deg]', classes='subsection-title')
                yield CoordinateList(
                    coordinates=cast(list[tuple[float, float]], initial_coords),
                    label1='RA',
                    label2='Dec',
                    id='coordinates-list',
                )

            # DataFrame input section
            df_config = inputs.get('dataframe', {})
            with Vertical(
                id='dataframe-input-section',
                classes='' if current_source == 'dataframe' else 'hidden',
            ):
                yield Static('DataFrame Configuration', classes='subsection-title')

                with Horizontal(classes='field-row'):
                    yield Label('CSV Path:', classes='field-label')
                    yield PathInput(
                        value=str(df_config.get('path', '')),
                        must_exist=True,
                        must_be_file=True,
                        id='dataframe-path',
                    )

                yield Static('Column Mappings', classes='subsection-title')
                columns = df_config.get('columns', {})

                with Horizontal(classes='field-row'):
                    yield Label('RA Column:', classes='field-label')
                    yield Input(
                        value=columns.get('ra', 'ra'),
                        id='df-col-ra',
                        classes='field-input',
                        validators=[NonEmptyValidator()],
                    )

                with Horizontal(classes='field-row'):
                    yield Label('Dec Column:', classes='field-label')
                    yield Input(
                        value=columns.get('dec', 'dec'),
                        id='df-col-dec',
                        classes='field-input',
                        validators=[NonEmptyValidator()],
                    )

                with Horizontal(classes='field-row'):
                    yield Label('ID Column:', classes='field-label')
                    yield Input(
                        value=columns.get('id', 'ID'),
                        id='df-col-id',
                        classes='field-input',
                        validators=[NonEmptyValidator()],
                    )

    def _compose_paths_tab(self) -> ComposeResult:
        """Compose the Paths settings tab."""
        machine = self._config_data.get('machine', 'local')
        paths = self._config_data.get('paths_by_machine', {}).get(machine, {})

        with ScrollableContainer():
            with Horizontal(classes='section-title'):
                yield Label(f'Paths Configuration (Machine: {machine})')
                yield InfoIcon('Changing machine in General tab will reload these paths')

            # Root directory main
            with Horizontal(classes='field-row'):
                yield Label('Root Directory:', classes='field-label')
                yield PathInput(
                    value=str(paths.get('root_dir_main', '')),
                    must_exist=False,
                    id='path-root-main',
                )

            # Root directory data
            with Horizontal(classes='field-row'):
                yield Label('Data Directory:', classes='field-label')
                yield PathInput(
                    value=str(paths.get('root_dir_data', '')),
                    must_exist=False,
                    id='path-root-data',
                )

            # Tables directory
            with Horizontal(classes='field-row'):
                yield Label('Tables Directory:', classes='field-label')
                yield PathInput(
                    value=str(paths.get('dir_tables', '')),
                    must_exist=False,
                    id='path-tables',
                )

            # Figures directory
            with Horizontal(classes='field-row'):
                yield Label('Figures Directory:', classes='field-label')
                yield PathInput(
                    value=str(paths.get('dir_figures', '')),
                    must_exist=False,
                    id='path-figures',
                )

            # Certificate path
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Certificate Path')
                    yield InfoIcon(
                        'CADC proxy certificate\nGenerate with: cadc-get-cert -u USERNAME'
                    )
                    yield Label(':')
                yield PathInput(
                    value=str(paths.get('cert_path', '')),
                    must_exist=True,
                    must_be_file=True,
                    is_certificate=True,
                    id='path-cert',
                )

    # ==================== Event Handlers ====================

    def on_mount(self) -> None:
        """Initialize the app after mounting."""
        self._update_header()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Track changes to mark config as dirty."""
        self._dirty = True
        self._update_header()

    def on_better_checkbox_changed(self, event: BetterCheckbox.Changed) -> None:
        """Track checkbox changes."""
        self._dirty = True
        self._update_header()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        self._dirty = True
        self._update_header()

        # Handle input source change
        if event.select.id == 'input-source-select':
            self._update_input_source_visibility(str(event.value))

        # Handle machine change
        elif event.select.id == 'machine-select':
            self._update_paths_for_machine(str(event.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == 'save-btn':
            self.action_save()
        elif event.button.id == 'cancel-btn':
            self.action_quit_app()

    def on_band_selector_changed(self, event: BandSelector.Changed) -> None:
        """Track band selection changes."""
        self._dirty = True
        self._update_header()

    def on_coordinate_list_changed(self, event: CoordinateList.Changed) -> None:
        """Track coordinate list changes."""
        self._dirty = True
        self._update_header()

    def on_tile_list_changed(self, event: TileList.Changed) -> None:
        """Track tile list changes."""
        self._dirty = True
        self._update_header()

    def on_rgb_band_selector_changed(self, event: RGBBandSelector.Changed) -> None:
        """Track RGB band selector changes."""
        self._dirty = True
        self._update_header()

    def on_path_input_changed(self, event: PathInput.Changed) -> None:
        """Track path input changes."""
        self._dirty = True
        self._update_header()

    # ==================== Actions ====================

    def action_save(self) -> None:
        """Save configuration to file and quit if successful."""
        # Validate first
        errors = self._validate_config()
        if errors:
            self.push_screen(
                ConfirmDialog(
                    'Validation Failed',
                    'Configuration has errors:\n\n• '
                    + '\n• '.join(errors[:5])
                    + '\n\nSave aborted. Return to editing?',
                    yes_label='Discard & Quit',
                    no_label='Back to Edit',
                ),
                self._handle_save_error_result,
            )
            return

        # Collect values
        config_dict = self._collect_all_values()

        # Write to file
        try:
            # Ensure directory exists
            if self._config_path:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self._config_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(config_dict, f, sort_keys=False, default_flow_style=False)

                self.notify(
                    f'Saved to {self._config_path}', title='Success', severity='information'
                )
                self.exit()
        except Exception as e:
            self.notify(str(e), title='Save Failed', severity='error')

    def _handle_save_error_result(self, quit: bool | None) -> None:
        """Handle result of validation failure dialog.

        Args:
            quit: True if 'Discard & Quit' was clicked, False if 'Back to Edit'
        """
        if quit:
            self.exit()

    def action_quit_app(self) -> None:
        """Quit the application, prompting if there are unsaved changes."""
        if self._dirty:
            self.push_screen(
                ConfirmDialog(
                    'Unsaved Changes',
                    'You have unsaved changes. Discard and exit?',
                ),
                self._handle_quit_confirm,
            )
        else:
            self.exit()

    def _handle_quit_confirm(self, confirmed: bool | None) -> None:
        """Handle quit confirmation dialog result."""
        if confirmed:
            self.exit()

    def action_help(self) -> None:
        """Show help information."""
        help_text = """
UNIONS Data Configuration Editor

Keyboard Shortcuts:
  Ctrl+S  - Save configuration and quit
  Ctrl+Q  - Quit
  Tab     - Next field
  Shift+Tab - Previous field

Tips:
  • Green ✓ indicates valid path/value
  • Red ✗ indicates invalid path/value
  • Yellow ⚠ indicates warning (e.g., expiring certificate)
"""
        self.notify(help_text, title='Help', severity='information', timeout=10)

    # ==================== Helper Methods ====================

    def _update_header(self) -> None:
        """Update header to show unsaved indicator."""
        header = self.query_one('#app-header', Static)
        indicator = r' \[unsaved]' if self._dirty else ''
        # Escape brackets to prevent Rich markup interpretation
        path_str = str(self._config_path).replace('[', r'\[').replace(']', r'\]')
        header.update(f'UNIONS Data Configuration ({path_str}){indicator}')

    def _update_input_source_visibility(self, source: str) -> None:
        """Show/hide input source sections based on selection."""
        sections = {
            'tiles': '#tiles-input-section',
            'coordinates': '#coordinates-input-section',
            'dataframe': '#dataframe-input-section',
        }

        for src, selector in sections.items():
            try:
                section = self.query_one(selector)
                if src == source:
                    section.remove_class('hidden')
                else:
                    section.add_class('hidden')
            except Exception:
                pass

    def _update_paths_for_machine(self, machine: str) -> None:
        """Update path fields when machine selection changes."""
        paths = self._config_data.get('paths_by_machine', {}).get(machine, {})

        path_mappings = [
            ('path-root-main', 'root_dir_main'),
            ('path-root-data', 'root_dir_data'),
            ('path-tables', 'dir_tables'),
            ('path-figures', 'dir_figures'),
            ('path-cert', 'cert_path'),
        ]

        for widget_id, config_key in path_mappings:
            try:
                path_input = self.query_one(f'#{widget_id}', PathInput)
                path_input.set_value(str(paths.get(config_key, '')))
            except Exception:
                pass

        # Update section title
        try:
            title = self.query_one('#paths-tab .section-title', Static)
            title.update(f'Path Configuration (Machine: {machine})')
        except Exception:
            pass

    def _collect_all_values(self) -> dict[str, Any]:
        """Collect all widget values into a configuration dictionary."""
        config: dict[str, Any] = {}

        # Start with existing config to preserve bands dict and other sections
        config = dict(self._config_data)

        # General
        try:
            config['machine'] = str(self.query_one('#machine-select', Select).value)
        except Exception:
            config['machine'] = 'local'

        config['logging'] = {
            'name': self._get_input_value('#logging-name', 'unionsdata'),
            'level': str(self.query_one('#logging-level', Select).value),
        }

        # Runtime
        config['runtime'] = {
            'n_download_threads': int(self._get_input_value('#n-download-threads', '12')),
            'n_cutout_processes': int(self._get_input_value('#n-cutout-processes', '2')),
            'bands': self.query_one('#band-selector', BandSelector).get_selected(),
            'resume': self.query_one('#resume-checkbox', BetterCheckbox).value,
            'max_retries': int(self._get_input_value('#max-retries', '5')),
        }

        # Tiles
        config['tiles'] = {
            'update_tiles': self.query_one('#update-tiles-checkbox', BetterCheckbox).value,
            'show_tile_statistics': self.query_one(
                '#show-tile-stats-checkbox', BetterCheckbox
            ).value,
            'band_constraint': int(self._get_input_value('#band-constraint', '1')),
            'require_all_specified_bands': self.query_one(
                '#require-all-bands-checkbox', BetterCheckbox
            ).value,
        }

        # Cutouts
        config['cutouts'] = {
            'enable': self.query_one('#cutouts-enable-checkbox', BetterCheckbox).value,
            'cutouts_only': self.query_one('#cutouts-only-checkbox', BetterCheckbox).value,
            'size_pix': int(self._get_input_value('#cutout-size', '512')),
            'output_subdir': self._get_input_value('#cutout-subdir', 'cutouts'),
        }

        # Plotting
        plot_bands = self.query_one('#rgb-band-selector', RGBBandSelector).get_selected_bands()
        # Empty list means None (use runtime bands)
        plot_bands_value = plot_bands if plot_bands else None

        config['plotting'] = {
            'catalog_name': self._get_input_value('#plot-catalog-name', 'catalog'),
            'bands': plot_bands_value,
            'size_pix': int(self._get_input_value('#plot-size', '512')),
            'mode': str(self.query_one('#plot-mode', Select).value),
            'max_cols': int(self._get_input_value('#plot-max-cols', '5')),
            'figsize': None,
            'save_plot': self.query_one('#plot-save-checkbox', BetterCheckbox).value,
            'show_plot': self.query_one('#plot-show-checkbox', BetterCheckbox).value,
            'save_format': str(self.query_one('#plot-save-format', Select).value),
            'rgb': {
                'scaling_type': str(self.query_one('#rgb-scaling-type', Select).value),
                'stretch': float(self._get_input_value('#rgb-stretch', '125.0')),
                'Q': float(self._get_input_value('#rgb-q', '7.0')),
                'gamma': float(self._get_input_value('#rgb-gamma', '0.25')),
                'standard_zp': float(self._get_input_value('#rgb-standard-zp', '30.0')),
            },
        }

        # Inputs
        source = str(self.query_one('#input-source-select', Select).value)
        config['inputs'] = {
            'source': source,
            'tiles': [list(t) for t in self.query_one('#tiles-list', TileList).get_tiles()],
            'coordinates': [
                list(c)
                for c in self.query_one('#coordinates-list', CoordinateList).get_coordinates()
            ],
            'dataframe': {
                'path': self.query_one('#dataframe-path', PathInput).value,
                'columns': {
                    'ra': self._get_input_value('#df-col-ra', 'ra'),
                    'dec': self._get_input_value('#df-col-dec', 'dec'),
                    'id': self._get_input_value('#df-col-id', 'ID'),
                },
            },
        }

        # Paths - update for current machine only
        machine = config['machine']
        if 'paths_by_machine' not in config:
            config['paths_by_machine'] = {}
        if machine not in config['paths_by_machine']:
            config['paths_by_machine'][machine] = {}

        config['paths_by_machine'][machine] = {
            'root_dir_main': self.query_one('#path-root-main', PathInput).value,
            'root_dir_data': self.query_one('#path-root-data', PathInput).value,
            'dir_tables': self.query_one('#path-tables', PathInput).value,
            'dir_figures': self.query_one('#path-figures', PathInput).value,
            'cert_path': self.query_one('#path-cert', PathInput).value,
        }

        # Preserve other sections from original config
        for key in ['paths_database', 'bands', 'plotting']:
            if key in self._config_data and key not in config:
                config[key] = self._config_data[key]

        return config

    def _get_input_value(self, selector: str, default: str = '') -> str:
        """Safely get input value with default."""
        try:
            return self.query_one(selector, Input).value
        except Exception:
            return default

    def _validate_config(self) -> list[str]:
        """Validate the current configuration and return list of errors."""
        errors: list[str] = []

        # Validate bands
        band_selector = self.query_one('#band-selector', BandSelector)
        if not band_selector.is_valid():
            errors.append('At least one band must be selected')

        # Validate numeric fields
        numeric_validations = [
            ('#n-download-threads', 'Download threads', 1, 32),
            ('#n-cutout-processes', 'Cutout processes', 1, 32),
            ('#max-retries', 'Max retries', 1, 10),
            ('#band-constraint', 'Band constraint', 1, 7),
            ('#cutout-size', 'Cutout size', 1, 10000),
        ]

        for selector, name, min_val, max_val in numeric_validations:
            try:
                value = int(self._get_input_value(selector, '0'))
                if value < min_val or value > max_val:
                    errors.append(f'{name} must be between {min_val} and {max_val}')
            except ValueError:
                errors.append(f'{name} must be a number')

        # Validate plotting bands (should be exactly 3 in correct order)
        rgb_selector = self.query_one('#rgb-band-selector', RGBBandSelector)
        selected_rgb = rgb_selector.get_selected_bands()
        if selected_rgb and len(selected_rgb) != 3:
            # Partial selection is invalid - must be all 3 or none
            errors.append('RGB bands must be either all 3 selected or empty (to use runtime bands)')

        # Validate paths
        cert_path = self.query_one('#path-cert', PathInput)
        if not cert_path.is_valid():
            errors.append('Certificate path is invalid or file does not exist')

        # Validate input source specifics
        source = str(self.query_one('#input-source-select', Select).value)
        if source == 'dataframe':
            df_path = self.query_one('#dataframe-path', PathInput)
            if not df_path.value.strip():
                errors.append('DataFrame path is required when source is "dataframe"')
            elif not df_path.is_valid():
                errors.append('DataFrame file does not exist')

        # Try Pydantic validation
        try:
            from unionsdata.config import RawConfig

            config_dict = self._collect_all_values()
            RawConfig.model_validate(config_dict)
        except Exception as e:
            errors.append(f'Config validation: {e}')

        return errors


def run_config_editor(config_path: Path | None = None) -> None:
    """
    Run the configuration editor TUI.

    Args:
        config_path: Optional path to config file. If None, uses default location.
    """
    app = ConfigEditorApp(config_path=config_path)
    app.run()
