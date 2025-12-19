"""Main Textual application for the configuration editor."""

from __future__ import annotations

import logging
import re
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import pexpect
from textual import work
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
    Switch,
    TabbedContent,
    TabPane,
)

from unionsdata.config import RawConfig, get_user_config_dir
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
from unionsdata.tui.widgets.rgb_selector import BANDS
from unionsdata.yaml_utils import load_yaml, parse_yaml, save_yaml

logger = logging.getLogger(__name__)


# Type alias for input sources
InputSource = Literal['all_available', 'tiles', 'coordinates', 'dataframe']
ButtonVariant = Literal['default', 'primary', 'success', 'warning', 'error']


def get_select_value(select: Select[str], default: str = '') -> str:
    """
    Safely get string value from Select widget.

    Handles Select.BLANK and None cases consistently.

    Args:
        select: The Select widget
        default: Default value if nothing selected

    Returns:
        The selected value as string, or default if nothing selected
    """
    value = select.value
    if value is None or value == Select.BLANK:
        return default
    return str(value)


def is_select_empty(select: Select[str]) -> bool:
    """Check if a Select widget has no selection."""
    return select.value is None or select.value == Select.BLANK


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
        max-height: 80%;
    }

    ConfirmDialog .dialog-title {
        text-style: bold;
        margin-bottom: 1;
        dock: top;
    }

    ConfirmDialog .dialog-message {
        margin-bottom: 2;
        height: auto;
        max-height: 1fr;
        overflow-y: auto;
    }

    ConfirmDialog .dialog-buttons {
        layout: horizontal;
        align: center middle;
        height: 3;
        dock: bottom;
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
            yield Static(self._message, classes='dialog-message', markup=False)
            with Horizontal(classes='dialog-buttons'):
                # ID mapping: yes -> True, no -> False
                yield Button(self._yes_label, variant=self._yes_variant, id='confirm-yes')
                yield Button(self._no_label, variant=self._no_variant, id='confirm-no')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'confirm-yes':
            self.dismiss(True)
        else:
            self.dismiss(False)


class RenewCertificateDialog(ModalScreen[dict[str, Any] | None]):
    """Dialog to renew the CADC certificate."""

    DEFAULT_CSS = """
    RenewCertificateDialog {
        align: center middle;
    }

    RenewCertificateDialog > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        height: auto;
    }

    RenewCertificateDialog .title {
        text-style: bold;
        margin-bottom: 1;
        text-align: center;
    }

    /* Spacing between inputs */
    RenewCertificateDialog #renew-username {
        margin-bottom: 1;
    }

    RenewCertificateDialog .buttons {
        margin-top: 2;
        align: center middle;
        height: 3;
    }

    RenewCertificateDialog Button {
        margin: 0 1;
    }

    /* Use !important to override the theme's specificity */
    #btn-renew:disabled,
    #btn-renew:disabled:hover {
        opacity: 0.4 !important;
        tint: 0% !important;
        border: none !important; /* Removes the 3D border that shimmers on hover */
        background: $primary !important; /* Locks background color */
        height: 3 !important; /* Enforce height since border removal might shrink it */
    }

    RenewCertificateDialog #status-message {
        margin-top: 1;
        text-align: center;
        color: $text-muted;
    }

    RenewCertificateDialog #status-message.error {
        color: $error;
    }
    """

    def __init__(self, current_username: str = '') -> None:
        super().__init__()
        self._initial_username = current_username

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static('Create/Renew CADC Certificate', classes='title')

            # validators=[NonEmptyValidator()] ensures the box turns Green (valid)
            # when text is present, and Red (invalid) when empty.
            yield Input(
                value=self._initial_username,
                placeholder='CADC Username',
                id='renew-username',
                validators=[NonEmptyValidator()],
            )

            yield Input(
                placeholder='Password',
                password=True,
                id='renew-password',
                validators=[NonEmptyValidator()],
            )

            yield Static('', id='status-message')

            with Horizontal(classes='buttons'):
                yield Button('🌱 Create/Renew', variant='primary', id='btn-renew', disabled=True)
                yield Button('✗ Cancel', variant='error', id='btn-cancel')

    def on_input_changed(self, event: Input.Changed) -> None:
        """Enable the renew button only when both fields are filled."""
        try:
            username = self.query_one('#renew-username', Input).value.strip()
            password = self.query_one('#renew-password', Input).value.strip()

            # Enable button only if both fields have text
            self.query_one('#btn-renew', Button).disabled = not (username and password)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'btn-cancel':
            self.dismiss(None)
        elif event.button.id == 'btn-renew':
            self._start_renewal()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields."""
        if event.input.id == 'renew-username':
            # Move focus to password field
            self.query_one('#renew-password', Input).focus()
        elif event.input.id == 'renew-password':
            # Submit if ready
            btn = self.query_one('#btn-renew', Button)
            if not btn.disabled:
                self._start_renewal()

    def _start_renewal(self) -> None:
        username = self.query_one('#renew-username', Input).value.strip()
        password = self.query_one('#renew-password', Input).value

        if not username or not password:
            self._update_status('Username and password required', error=True)
            return

        if not shutil.which('cadc-get-cert'):
            self._update_status("Error: 'cadc-get-cert' command not found", error=True)
            return

        self._update_status('Requesting certificate...', error=False)
        self.query_one('#btn-renew', Button).disabled = True

        self._run_cert_command(username, password)

    @work(thread=True)
    def _run_cert_command(self, username: str, password: str) -> None:
        """Run the command using pexpect to handle the interactive password prompt."""

        try:
            cmd = 'cadc-get-cert'
            args = ['-u', username]

            child = pexpect.spawn(cmd, args, encoding='utf-8', timeout=15)

            index = child.expect(['[Pp]assword:', pexpect.EOF, pexpect.TIMEOUT])

            if index == 0:
                child.sendline(password)
                child.expect(pexpect.EOF)
                child.close()

                raw_output = child.before or ''
                output = raw_output.strip()

                # Check for explicit failure messages first
                if 'invalid username/password combination' in output or 'FAILED' in output:
                    self.app.call_from_thread(self._handle_error, 'Invalid username or password')
                    return

                if child.exitstatus == 0:
                    # Strict success check: We MUST find the "saved in" message.
                    match = re.search(r'saved in\s+(.+)$', output, re.MULTILINE)

                    if match:
                        path = match.group(1).strip()
                        if path.endswith('.'):
                            path = path[:-1]
                        self.app.call_from_thread(self._handle_success, path)
                    else:
                        # No "saved in" message = FAILURE.
                        self.app.call_from_thread(
                            self._handle_error,
                            f'Certificate renewal failed. Unexpected output: {output}',
                        )
                else:
                    self.app.call_from_thread(self._handle_error, output)

            elif index == 1:
                child.close()
                err_output = (child.before or '').strip() or 'Process exited unexpectedly'
                self.app.call_from_thread(self._handle_error, err_output)

            elif index == 2:
                child.close()
                self.app.call_from_thread(self._handle_error, 'Connection timed out')

        except Exception as e:
            self.app.call_from_thread(self._handle_error, str(e))

    def _handle_success(self, path: str) -> None:
        """Return success data to the main app."""
        self.dismiss({'success': True, 'path': path})

    def _handle_error(self, message: str) -> None:
        if 'Permission denied' in message or 'Login failed' in message:
            ui_msg = 'Invalid username or password'
        else:
            ui_msg = f'Error: {message}'
        self._update_status(ui_msg, error=True)
        self.query_one('#btn-renew', Button).disabled = False

    def _update_status(self, message: str, error: bool = False) -> None:
        status = self.query_one('#status-message', Static)
        status.update(message)
        if error:
            status.add_class('error')
        else:
            status.remove_class('error')


class NewProfileDialog(ModalScreen[str | None]):
    """Dialog to create a new profile."""

    DEFAULT_CSS = """
    NewProfileDialog {
        align: center middle;
    }

    NewProfileDialog > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        height: auto;
    }

    NewProfileDialog .title {
        text-style: bold;
        margin-bottom: 1;
        text-align: center;
    }

    NewProfileDialog .buttons {
        margin-top: 2;
        align: center middle;
        height: 3;
    }

    NewProfileDialog Button {
        margin: 0 1;
    }

    #btn-create:disabled,
    #btn-create:disabled:hover {
        opacity: 0.4 !important;
        tint: 0% !important;
        border: none !important; /* Removes the 3D border that shimmers on hover */
        background: $primary !important; /* Locks background color */
        height: 3 !important; /* Enforce height since border removal might shrink it */
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static('Create New Profile', classes='title')
            yield Input(
                placeholder='Profile Name (e.g., cluster_x)',
                id='profile-name',
                validators=[NonEmptyValidator()],
            )
            with Horizontal(classes='buttons'):
                yield Button('🌱 Create', variant='primary', id='btn-create', disabled=True)
                yield Button('✗ Cancel', variant='error', id='btn-cancel')

    def on_input_changed(self, event: Input.Changed) -> None:
        """Enable the create button only when the field is filled."""
        try:
            name = self.query_one('#profile-name', Input).value.strip()

            # Enable button only if field has text
            self.query_one('#btn-create', Button).disabled = not bool(name)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'btn-cancel':
            self.dismiss(None)
        elif event.button.id == 'btn-create':
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        name = self.query_one('#profile-name', Input).value.strip()
        if name:
            self.dismiss(name)
        else:
            self.notify('Profile name cannot be empty', severity='error')


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
        self._previous_machine = self._config_data.get('machine', 'local')

    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        # Determine config path
        if self._config_path is None:
            self._config_path = get_user_config_dir() / 'config.yaml'

        # Try to load existing config
        if self._config_path.exists():
            try:
                self._config_data = load_yaml(self._config_path)
                logger.debug(f'Loaded config from {self._config_path}')
            except Exception as e:
                logger.warning(f'Failed to load config: {e}, using defaults')
                self._config_data = {}
        else:
            # Load template from package
            try:
                template_path = files('unionsdata').joinpath('config.yaml')
                template_content = template_path.read_text(encoding='utf-8')
                self._config_data = parse_yaml(template_content)
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
                'bands': None,
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
                'mode': 'after_download',
                'size_pix': 512,
                'output_subdir': 'cutouts',
            },
            'plotting': {
                'enable': False,
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
                'mono': {
                    'scaling_type': 'asinh',
                    'stretch': 125.0,
                    'Q': 7.0,
                    'gamma': 0.25,
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

            options = [(m, m) for m in machines]
            options.append(('+ create new...', 'create_new'))

            with Horizontal(classes='field-row'):
                yield Label('Machine:', classes='field-label')
                yield Select(
                    options,
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
        current_bands = runtime.get('bands') or []

        with ScrollableContainer():
            with Horizontal(classes='section-title'):
                yield Label('Band Selection')
                yield InfoIcon('Select bands to download (at least one required)')

            # BandSelector moved here
            yield BandSelector(
                selected=current_bands,
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
                yield Select(
                    [(str(i), str(i)) for i in range(1, 8)],
                    value=str(tiles.get('band_constraint', 1)),
                    id='band-constraint',
                    classes='field-input',
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

            # Cutout mode
            cutout_modes: list[tuple[str, str]] = [
                ('Disabled', 'disabled'),
                ('After Download', 'after_download'),
                ('Direct Only', 'direct_only'),
            ]
            current_mode = cutouts.get('mode', 'after_download')

            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Cutout Mode')
                    yield InfoIcon(
                        'Disabled: No cutouts created. '
                        'After Download: Download full tiles, then extract cutouts. '
                        'Direct Only: Fetch cutouts directly without storing full tiles.'
                    )
                    yield Label(':')
                yield Select(
                    cutout_modes,
                    value=current_mode,
                    id='cutout-mode-select',
                    classes='field-input',
                )

            with Vertical(
                id='cutout-options-container',
                classes='' if current_mode != 'disabled' else 'hidden',
            ):
                # Cutout size
                with Horizontal(classes='field-row'):
                    with Horizontal(classes='field-label'):
                        yield Label('Size (pixels)')
                        yield InfoIcon(
                            'Square cutout size in pixels (pixel scale: 0.1857 arcsec/pix)'
                        )
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
        # Get the enable state
        is_enabled = plotting.get('enable', False)

        rgb = plotting.get('rgb', {})
        mono = plotting.get('mono', {})

        # get initial plot bands from config
        plot_bands = plotting.get('bands') or []
        # if 3 or more bands are set, default to RGB mode
        is_rgb = len(plot_bands) >= 3

        with ScrollableContainer():
            # enable plotting switch
            yield Static('Plotting Module', classes='section-title')
            with Horizontal(classes='field-row'):
                with Horizontal(classes='field-label'):
                    yield Label('Enable Plotting')
                    yield InfoIcon(
                        'Enable or disable cutout plotting after creation. See Cutouts section.'
                    )
                    yield Label(':')
                yield Switch(value=is_enabled, id='plot-enable-switch')
            with Vertical(id='plotting-content', classes='' if is_enabled else 'hidden'):
                with ScrollableContainer():
                    # general plotting config
                    yield Static('General Plotting Configuration', classes='section-title')

                    with Horizontal(classes='field-row'):
                        with Horizontal(classes='field-label'):
                            yield Label('Catalog Name')
                            yield InfoIcon(
                                'Name of catalog file (without _augmented.csv suffix). Use auto to use the most recent input catalog.'
                            )
                            yield Label(':')

                        current_catalog = plotting.get('catalog_name', 'auto')
                        options = self._get_catalog_options()
                        # Ensure current value is in options
                        existing_values = {opt[1] for opt in options}
                        if current_catalog and current_catalog not in existing_values:
                            options.append((current_catalog, current_catalog))

                        yield Select(
                            options,
                            value=current_catalog,
                            id='plot-catalog-name',
                            classes='field-input',
                        )

                    # cutout size
                    with Horizontal(classes='field-row'):
                        with Horizontal(classes='field-label'):
                            yield Label('Size (pixels)')
                            yield InfoIcon(
                                'Square cutout size in pixels (pixel scale: 0.1857 arcsec/pix)'
                            )
                            yield Label(':')
                        yield Input(
                            value=str(plotting.get('size_pix', 512)),
                            id='plot-size',
                            classes='field-input',
                            validators=[IntegerRange(1, 10000)],
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

                    with Horizontal(classes='field-row'):
                        yield Label('Save Plot:', classes='field-label')
                        yield BetterCheckbox(
                            'Save to disk',
                            plotting.get('save_plot', True),
                            id='plot-save-checkbox',
                            classes='field-checkbox',
                        )

                    with Horizontal(classes='field-row'):
                        yield Label('Show Plot:', classes='field-label')
                        yield BetterCheckbox(
                            'Display plot',
                            plotting.get('show_plot', False),
                            id='plot-show-checkbox',
                            classes='field-checkbox',
                        )

                    save_format = plotting.get('save_format', 'pdf')
                    formats = [('PDF', 'pdf'), ('PNG', 'png'), ('JPG', 'jpg'), ('SVG', 'svg')]
                    with Horizontal(classes='field-row'):
                        yield Label('Save Format:', classes='field-label')
                        yield Select(
                            formats, value=save_format, id='plot-save-format', classes='field-input'
                        )

                    # switch between RGB and mono mode
                    yield Static('Plot Mode', classes='section-title')
                    with Horizontal(classes='field-row'):
                        yield Label('Monochromatic', classes='field-label')
                        # The Switch that controls visibility
                        yield Switch(value=is_rgb, id='plot-mode-switch')
                        yield Label('RGB', classes='field-label')

                    # mono mode (hidden if RGB)
                    with Vertical(id='plot-mono-container', classes='hidden' if is_rgb else ''):
                        yield Static('Monochromatic Plotting Options', classes='section-title')
                        # Single Band Selector (All known bands)
                        all_band_opts = [(b.display, b.name) for b in BANDS]
                        with Horizontal(classes='field-row'):
                            yield Label('Band:', classes='field-label')
                            yield Select(
                                all_band_opts,
                                value=plot_bands[0] if plot_bands else Select.BLANK,
                                id='plot-mono-band',
                                classes='field-input',
                            )

                        # Scaling
                        scaling_type = mono.get('scaling_type', 'asinh')
                        scaling_types = [('Asinh', 'asinh'), ('Linear', 'linear')]
                        with Horizontal(classes='field-row'):
                            with Horizontal(classes='field-label'):
                                yield Label('Scaling Type')
                                yield InfoIcon('asinh preserves both bright and faint details')
                                yield Label(':')
                            yield Select(
                                scaling_types,
                                value=scaling_type,
                                id='mono-scaling-type',
                                classes='field-input',
                            )

                        # Stretch
                        with Horizontal(classes='field-row'):
                            with Horizontal(classes='field-label'):
                                yield Label('Stretch')
                                yield InfoIcon('Scaling factor controlling overall brightness')
                                yield Label(':')
                            yield Input(
                                value=str(mono.get('stretch', 125.0)),
                                id='mono-stretch',
                                classes='field-input',
                            )

                        # Q
                        with Horizontal(classes='field-row'):
                            with Horizontal(classes='field-label'):
                                yield Label('Q')
                                yield InfoIcon(
                                    'Softening parameter for asinh (higher = more linear)'
                                )
                                yield Label(':')
                            yield Input(
                                value=str(mono.get('Q', 7.0)),
                                id='mono-q',
                                classes='field-input',
                            )

                        # Gamma
                        with Horizontal(classes='field-row'):
                            with Horizontal(classes='field-label'):
                                yield Label('Gamma')
                                yield InfoIcon('Gamma correction (lower = enhances faint features)')
                                yield Label(':')
                            yield Input(
                                value=str(mono.get('gamma', 0.25)),
                                id='mono-gamma',
                                classes='field-input',
                            )

                    # RGB mode (hidden if mono)
                    with Vertical(id='plot-rgb-container', classes='' if is_rgb else 'hidden'):
                        yield Static('RGB Plotting Options', classes='section-title')
                        with Horizontal(classes='field-row'):
                            with Horizontal(classes='field-label'):
                                yield Label('RGB Bands')
                                yield InfoIcon(
                                    'Select bands for RGB channels, or leave unset to use first 3 runtime bands. Wavelength order must be followed (Blue < Green < Red). Hit the reset button to clear selections and start over.'
                                )
                                yield Label(':')

                            # New Widget Integration
                            yield RGBBandSelector(
                                selected_bands=plot_bands,
                                id='rgb-band-selector',
                            )

                        # Display Mode is only relevant for RGB
                        mode = plotting.get('mode', 'grid')
                        modes = [('Grid', 'grid'), ('Channel', 'channel')]
                        with Horizontal(classes='field-row'):
                            with Horizontal(classes='field-label'):
                                yield Label('Display Mode')
                                yield InfoIcon('Grid: thumbnails; Channel: show R,G,B separately')
                                yield Label(':')
                            yield Select(
                                modes,
                                value=mode,
                                id='plot-mode',
                                classes='field-input',
                            )

                        # RGB Specific Scaling
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
                                yield InfoIcon(
                                    'Softening parameter for asinh (higher = more linear)'
                                )
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
                        'Path to CADC certificate. Generate a certificate by pressing the Create/Renew button to the right. The certificate path will be filled out automatically after creation.'
                    )
                    yield Label(':')

                # Pass the button directly to PathInput
                yield PathInput(
                    value=str(paths.get('cert_path', '')),
                    must_exist=True,
                    must_be_file=True,
                    is_certificate=True,
                    id='path-cert',
                    # Create button here and pass it in
                    action_button=Button('🌱 Create/Renew', id='btn-open-renew', variant='default'),
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
        # Handle cutout mode change
        elif event.select.id == 'cutout-mode-select':
            try:
                container = self.query_one('#cutout-options-container')
                if event.value == 'disabled':
                    container.add_class('hidden')
                else:
                    container.remove_class('hidden')
            except Exception:
                pass
        # Handle machine change
        elif event.select.id == 'machine-select':
            if event.value == 'create_new':
                # Open the dialog
                self.push_screen(NewProfileDialog(), self._handle_new_profile)
            else:
                self._update_paths_for_machine(str(event.value))
                self._previous_machine = str(event.value)
                self._refresh_catalog_options()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == 'save-btn':
            self.action_save()
        elif event.button.id == 'cancel-btn':
            self.action_quit_app()
        elif event.button.id == 'btn-open-renew':
            # Open dialog without arguments (we don't pass current path anymore)
            self.push_screen(RenewCertificateDialog(), self._handle_renew_result)

    def on_band_selector_changed(self, event: BandSelector.Changed) -> None:
        """Track band selection changes and auto-toggle plotting mode."""
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

        # If the tables directory changes, refresh the catalog dropdown
        if event.control.id == 'path-tables':
            self._refresh_catalog_options()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes (Plot Mode)."""
        # master enable switch for plotting
        if event.switch.id == 'plot-enable-switch':
            is_enabled = event.value
            content = self.query_one('#plotting-content')

            if is_enabled:
                content.remove_class('hidden')
            else:
                content.add_class('hidden')

            self._dirty = True
            self._update_header()

        # switch between RGB and Mono plotting modes
        if event.switch.id == 'plot-mode-switch':
            is_rgb = event.value

            mono = self.query_one('#plot-mono-container')
            rgb = self.query_one('#plot-rgb-container')

            if is_rgb:
                # RGB Mode: Hide Mono, Show RGB
                mono.add_class('hidden')
                rgb.remove_class('hidden')
            else:
                # Mono Mode: Hide RGB, Show Mono
                rgb.add_class('hidden')
                mono.remove_class('hidden')

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
                save_yaml(self._config_path, config_dict)

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

    def _handle_new_profile(self, name: str | None) -> None:
        """Handle the creation of a new machine profile."""
        select = self.query_one('#machine-select', Select)

        # handle cancellation
        if not name:
            select.value = self._previous_machine
            return

        # handle duplicate name
        if name in self._config_data.get('paths_by_machine', {}):
            self.notify(f"Profile '{name}' already exists", severity='error')
            select.value = self._previous_machine
            return

        # create new entry in config
        if 'paths_by_machine' not in self._config_data:
            self._config_data['paths_by_machine'] = {}

        self._config_data['paths_by_machine'][name] = {
            'root_dir_main': '',
            'root_dir_data': '',
            'dir_tables': '',
            'dir_figures': '',
            'cert_path': '',
        }

        # update select options
        # we must regenerate the options list to include the new key
        machines = list(self._config_data['paths_by_machine'].keys())
        options = [(m, m) for m in machines]
        options.append(('+ Create New Profile...', 'create_new'))
        select.set_options(options)

        # select the new profile and switch tabs
        # this will trigger on_select_changed again, which will call _update_paths_for_machine
        select.value = name
        self._previous_machine = name

        # Switch to the Paths tab so user can start editing
        self.query_one(TabbedContent).active = 'paths-tab'

        self.notify(f"Created profile '{name}'. Please configure paths.")

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

    def _handle_renew_result(self, result: dict[str, Any] | None) -> None:
        """Called when the renewal dialog closes."""
        if result and result.get('success'):
            path = result['path']

            # update the UI with the auto-detected path
            try:
                cert_input = self.query_one('#path-cert', PathInput)
                cert_input.set_value(path)
                cert_input.force_validate()
            except Exception:
                pass

            # show notification
            msg = f'Certificate saved successfully.\nLocation: {path}'
            self.notify(msg, title='Success', severity='information', timeout=5.0)

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

    def _get_catalog_options(self) -> list[tuple[str, str]]:
        """Get catalog options from the tables directory."""
        # Always start with Auto
        options = [('Auto', 'auto')]

        try:
            # Determine the table directory to search
            try:
                # Try getting path from the "Paths" tab input if it's mounted
                dir_str = self.query_one('#path-tables', PathInput).value
            except Exception:
                # Fallback to initial config
                machine = self._config_data.get('machine', 'local')
                paths = self._config_data.get('paths_by_machine', {}).get(machine, {})
                dir_str = str(paths.get('dir_tables', ''))

            # Scan directory
            dir_path = Path(dir_str).expanduser()
            if dir_path.exists() and dir_path.is_dir():
                found_catalogs = []
                for file_path in dir_path.glob('*_augmented.csv'):
                    # Remove the suffix to get the catalog name
                    # e.g., "my_data_augmented.csv" -> "my_data"
                    name = file_path.name.replace('_augmented.csv', '')
                    found_catalogs.append((name, name))

                # Sort alphabetically and add to options
                found_catalogs.sort()
                options.extend(found_catalogs)

        except Exception:
            # Fail silently and just show "Auto" if paths are invalid
            pass

        return options

    def _refresh_catalog_options(self) -> None:
        """Refresh the catalog options based on the current tables directory."""
        try:
            # get the widget
            try:
                catalog_select = self.query_one('#plot-catalog-name', Select)
            except Exception:
                return

            # get the new options based on the current path input
            options = self._get_catalog_options()

            # update options
            catalog_select.set_options(options)

            # if the selection became empty (BLANK), force it to 'auto'.
            if catalog_select.value == Select.BLANK:
                catalog_select.value = 'auto'

        except Exception:
            pass

    def _collect_all_values(self) -> dict[str, Any]:
        """Collect all widget values into a configuration dictionary."""
        config: dict[str, Any] = {}

        # start with existing config to preserve bands dict and other sections
        config = dict(self._config_data)

        # general
        config['machine'] = get_select_value(
            self.query_one('#machine-select', Select), default='local'
        )

        config['logging'] = {
            'name': self._get_input_value('#logging-name', 'unionsdata'),
            'level': str(self.query_one('#logging-level', Select).value),
        }

        # runtime
        config['runtime'] = {
            'n_download_threads': int(self._get_input_value('#n-download-threads', '12')),
            'n_cutout_processes': int(self._get_input_value('#n-cutout-processes', '2')),
            'bands': self.query_one('#band-selector', BandSelector).get_selected(),
            'resume': self.query_one('#resume-checkbox', BetterCheckbox).value,
            'max_retries': int(self._get_input_value('#max-retries', '5')),
        }

        # tiles
        config['tiles'] = {
            'update_tiles': self.query_one('#update-tiles-checkbox', BetterCheckbox).value,
            'show_tile_statistics': self.query_one(
                '#show-tile-stats-checkbox', BetterCheckbox
            ).value,
            'band_constraint': int(str(self.query_one('#band-constraint', Select).value)),
            'require_all_specified_bands': self.query_one(
                '#require-all-bands-checkbox', BetterCheckbox
            ).value,
        }

        # cutouts
        config['cutouts'] = {
            'mode': str(self.query_one('#cutout-mode-select', Select).value),
            'size_pix': int(self._get_input_value('#cutout-size', '512')),
            'output_subdir': self._get_input_value('#cutout-subdir', 'cutouts'),
        }

        # plotting
        # determine if we are in RGB or Mono mode
        is_rgb = self.query_one('#plot-mode-switch', Switch).value

        if is_rgb:
            # rgb mode: get 3 bands
            rgb_bands = self.query_one('#rgb-band-selector', RGBBandSelector).get_selected_bands()
            plot_bands_value = rgb_bands if rgb_bands else None
        else:
            # mono mode: get 1 band
            mono_band = str(self.query_one('#plot-mono-band', Select).value)
            # save as a list of 1 string for config consistency
            plot_bands_value = [mono_band] if mono_band != Select.BLANK else None

        config['plotting'] = {
            'enable': self.query_one('#plot-enable-switch', Switch).value,
            'catalog_name': str(self.query_one('#plot-catalog-name', Select).value),
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
            'mono': {
                'scaling_type': str(self.query_one('#mono-scaling-type', Select).value),
                'stretch': float(self._get_input_value('#mono-stretch', '125.0')),
                'Q': float(self._get_input_value('#mono-q', '7.0')),
                'gamma': float(self._get_input_value('#mono-gamma', '0.25')),
            },
        }

        # inputs
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

        # paths - update for current machine only
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

        # preserve other sections from original config
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

        # check all Select widgets for missing selections
        selects = {
            '#machine-select': 'Machine',
            '#logging-level': 'Log Level',
            '#plot-save-format': 'Save Format',
            '#input-source-select': 'Input Source',
            '#band-constraint': 'Band Constraint',
            '#cutout-mode-select': 'Cutout Mode',
        }

        for selector, name in selects.items():
            try:
                select = self.query_one(selector, Select)
                if is_select_empty(select):
                    errors.append(f'{name} must be selected')
            except Exception:
                pass

        # validate bands
        band_selector = self.query_one('#band-selector', BandSelector)
        if not band_selector.is_valid():
            errors.append('At least one band must be selected')

        # validate numeric fields
        numeric_validations = [
            ('#n-download-threads', 'Download threads', 1, 32),
            ('#n-cutout-processes', 'Cutout processes', 1, 32),
            ('#max-retries', 'Max retries', 1, 10),
            ('#cutout-size', 'Cutout size', 1, 10000),
        ]

        for selector, name, min_val, max_val in numeric_validations:
            try:
                value = int(self._get_input_value(selector, '0'))
                if value < min_val or value > max_val:
                    errors.append(f'{name} must be between {min_val} and {max_val}')
            except ValueError:
                errors.append(f'{name} must be a number')

        # plotting validation
        # only validate if plotting is enabled
        plotting_enabled = self.query_one('#plot-enable-switch', Switch).value
        if plotting_enabled:
            plot_required_selects = {
                '#plot-catalog-name': 'Catalog Name',
                '#plot-save-format': 'Save Format',
            }

            for selector, name in plot_required_selects.items():
                try:
                    select = self.query_one(selector, Select)
                    if is_select_empty(select):
                        errors.append(f'{name} must be selected')
                except Exception:
                    pass
            # check if we are in RGB or Mono mode
            is_rgb = self.query_one('#plot-mode-switch', Switch).value

            if is_rgb:
                # Validate RGB-specific fields
                rgb_selects = {
                    '#plot-mode': 'Display Mode',
                    '#rgb-scaling-type': 'Scaling Type',
                }
                for selector, name in rgb_selects.items():
                    try:
                        select = self.query_one(selector, Select)
                        if is_select_empty(select):
                            errors.append(f'{name} must be selected')
                    except Exception:
                        pass

                # Validate RGB bands
                rgb_selector = self.query_one('#rgb-band-selector', RGBBandSelector)
                selected_rgb = rgb_selector.get_selected_bands()
                if not selected_rgb or len(selected_rgb) != 3:
                    errors.append('All three RGB bands must be selected')
            else:
                # validate mono-specific fields
                mono_selects = {'#mono-scaling-type': 'Scaling Type', '#plot-mono-band': 'Band'}
                # Validate Mono band
                for selector, name in mono_selects.items():
                    try:
                        select = self.query_one(selector, Select)
                        if is_select_empty(select):
                            errors.append(f'{name} must be selected')
                    except Exception:
                        pass

        # validate paths
        cert_path = self.query_one('#path-cert', PathInput)
        if not cert_path.is_valid():
            errors.append('Certificate path is invalid or file does not exist')

        # validate input source specifics
        source = str(self.query_one('#input-source-select', Select).value)
        if source == 'dataframe':
            df_path = self.query_one('#dataframe-path', PathInput)
            if not df_path.value.strip():
                errors.append('DataFrame path is required when source is "dataframe"')
            elif not df_path.is_valid():
                errors.append('DataFrame file does not exist')

        if errors:
            return errors

        # try pydantic validation
        try:
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
