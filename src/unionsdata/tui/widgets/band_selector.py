"""Band selector widget with checkboxes for multi-band selection."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from unionsdata.tui.widgets.better_checkbox import BetterCheckbox


# Band definitions with display names
AVAILABLE_BANDS: list[tuple[str, str, str]] = [
    ('cfis-u', 'u', 'CFIS u-band'),
    ('whigs-g', 'g', 'WHIGS g-band'),
    ('cfis-r', 'r', 'CFIS r-band'),
    ('cfis_lsb-r', 'r (LSB)', 'CFIS LSB r-band'),
    ('ps-i', 'i', 'Pan-STARRS i-band'),
    ('wishes-z', 'z', 'WISHES z-band'),
    ('ps-z', 'z (PS)', 'Pan-STARRS z-band'),
]


class BandSelector(Static):
    """
    Multi-select widget for choosing bands.

    Displays checkboxes for all available bands and tracks selection state.
    """

    DEFAULT_CSS = """
    BandSelector {
        height: auto;
        padding: 0 1;
    }

    BandSelector .band-grid {
        height: auto;
        layout: grid;
        grid-size: 2;
        grid-gutter: 0 2;
        padding: 0;
    }

    BandSelector .band-checkbox {
        height: 1;
        padding: 0;
        margin: 0;
    }

    BandSelector .validation-error {
        color: $error;
        height: 1;
        margin-top: 1;
    }
    """

    selected_bands: reactive[set[str]] = reactive(set, layout=True)

    class Changed(Message):
        """Message sent when band selection changes."""

        def __init__(self, band_selector: BandSelector, selected: set[str]) -> None:
            self.band_selector = band_selector
            self.selected = selected
            super().__init__()

        @property
        def control(self) -> BandSelector:
            return self.band_selector

    def __init__(
        self,
        selected: list[str] | set[str] | None = None,
        *,
        min_selected: int = 1,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """
        Initialize BandSelector.

        Args:
            selected: Initially selected band IDs
            min_selected: Minimum number of bands that must be selected
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self._initial_selected = set(selected) if selected else set()
        self._min_selected = min_selected

    def compose(self) -> ComposeResult:
        with Vertical(classes='band-grid'):
            for band_id, short_name, description in AVAILABLE_BANDS:
                is_checked = band_id in self._initial_selected
                # Create a safe ID by replacing hyphens and underscores
                safe_id = band_id.replace('-', '_').replace('.', '_')
                yield BetterCheckbox(
                    f'{short_name}: {description}',
                    is_checked,
                    id=f'band_{safe_id}',
                    classes='band-checkbox',
                )
        yield Static('', classes='validation-error', id='band-validation-error')

    def on_mount(self) -> None:
        """Initialize selected bands on mount."""
        self.selected_bands = self._initial_selected.copy()
        self._update_validation()

    def on_better_checkbox_changed(self, event: BetterCheckbox.Changed) -> None:
        """Handle checkbox state changes."""
        if not event.checkbox.id or not event.checkbox.id.startswith('band_'):
            return

        # Extract band ID from checkbox ID
        safe_id = event.checkbox.id[5:]  # Remove 'band_' prefix

        # Convert safe_id back to band_id
        band_id = None
        for bid, _, _ in AVAILABLE_BANDS:
            if bid.replace('-', '_').replace('.', '_') == safe_id:
                band_id = bid
                break

        if band_id is None:
            return

        # Update selection
        new_selection = self.selected_bands.copy()
        if event.value:
            new_selection.add(band_id)
        else:
            new_selection.discard(band_id)

        self.selected_bands = new_selection
        self._update_validation()
        self.post_message(self.Changed(self, self.selected_bands))

    def _update_validation(self) -> None:
        """Update validation error message."""
        error_widget = self.query_one('#band-validation-error', Static)

        if len(self.selected_bands) < self._min_selected:
            error_widget.update(
                f'⚠ At least {self._min_selected} band(s) must be selected'
            )
        else:
            error_widget.update('')

    def get_selected(self) -> list[str]:
        """
        Get list of selected band IDs in wavelength order.

        Returns:
            List of selected band IDs sorted by wavelength (u → g → r → i → z)
        """
        # Return in the order defined in AVAILABLE_BANDS (wavelength order)
        return [bid for bid, _, _ in AVAILABLE_BANDS if bid in self.selected_bands]

    def set_selected(self, bands: list[str] | set[str]) -> None:
        """
        Programmatically set selected bands.

        Args:
            bands: Band IDs to select
        """
        self.selected_bands = set(bands)

        # Update checkbox states
        for band_id, _, _ in AVAILABLE_BANDS:
            safe_id = band_id.replace('-', '_').replace('.', '_')
            try:
                checkbox = self.query_one(f'#band_{safe_id}', BetterCheckbox)
                checkbox.value = band_id in self.selected_bands
            except Exception:
                pass

        self._update_validation()

    def is_valid(self) -> bool:
        """Check if selection meets minimum requirement."""
        return len(self.selected_bands) >= self._min_selected
