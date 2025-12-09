"""RGB band selector widget with wavelength-aware constraints."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, Select, Static


@dataclass(frozen=True)
class BandInfo:
    """Band information with wavelength rank for ordering."""

    name: str  # The actual band name used in config (e.g., 'cfis-r')
    display: str  # Human-readable display name
    rank: int  # Wavelength rank (1=shortest, 5=longest)


# Band definitions ordered by wavelength
# Bands with the same rank are alternatives (e.g., two r-band options)
BANDS = [
    BandInfo('cfis-u', 'CFIS u', 1),
    BandInfo('whigs-g', 'WHIGS g', 2),
    BandInfo('cfis-r', 'CFIS r', 3),
    BandInfo('cfis_lsb-r', 'CFIS LSB r', 3),  # Same rank as cfis-r
    BandInfo('ps-i', 'PS i', 4),
    BandInfo('wishes-z', 'WISHES z', 5),
    BandInfo('ps-z', 'PS z', 5),  # Same rank as wishes-z
]

# Lookup by band name for quick access
BAND_BY_NAME: dict[str, BandInfo] = {b.name: b for b in BANDS}

# Precompute all valid (blue, green, red) combinations
# Blue must have lower rank than green, green lower than red
VALID_COMBINATIONS = [
    (b1, b2, b3) for b1 in BANDS for b2 in BANDS for b3 in BANDS if b1.rank < b2.rank < b3.rank
]


class RGBBandSelector(Static):
    """
    Widget for selecting 3 bands (Blue, Green, Red).
    Selections are replaced by static labels ("locked in") immediately.
    """

    DEFAULT_CSS = """
    RGBBandSelector {
        height: auto;
        margin-bottom: 1;
    }

    RGBBandSelector Horizontal {
        height: auto;
        align: left top;
    }

    /* * Wrappers hold the colored top-border and the content (Select or Label).
     * Fixed width ensures alignment consistency.
     */
    .band-wrapper {
        width: 30;
        height: auto;
        margin-right: 1;
        border-top: heavy transparent;
    }

    /* Persistent colored top borders */
    .blue-band { border-top: heavy #3b82f6; }
    .green-band { border-top: heavy #22c55e; }
    .red-band { border-top: heavy #ef4444; }

    /* Interactive Select Widget */
    .band-wrapper Select {
        width: 100%;
        margin: 0;
        height: 3; /* Enforce standard height */
    }

    /* Locked Label State - Styles match Select to prevent layout jump */
    .locked-label {
        width: 100%;
        height: 3;
        padding: 0 2;
        content-align: left middle;
        background: $surface;
        color: $text;
        border: tall $primary-background; /* Matches Select default border */
        text-style: bold;
    }

    #reset-btn {
        min-width: 10;
        margin-left: 2;
        /* Push down 1 line to align with text, ignoring the heavy top border */
        margin-top: 1;
        height: 3;
    }
    """

    class Changed(Message):
        """Posted when any band selection changes."""

        def __init__(self, selector: RGBBandSelector) -> None:
            self.selector = selector
            super().__init__()

    def __init__(self, selected_bands: list[str] | None = None, id: str | None = None) -> None:
        super().__init__(id=id)
        self.initial_bands = selected_bands or []
        # Store actual band names (e.g., 'cfis-r'), not internal IDs
        self.selections: dict[str, str | None] = {'blue': None, 'green': None, 'red': None}

    def compose(self) -> ComposeResult:
        with Horizontal():
            # Blue Slot
            with Vertical(id='wrap_blue', classes='band-wrapper blue-band'):
                yield Select[str]([], prompt='Blue (Short)', id='sel_blue')

            # Green Slot
            with Vertical(id='wrap_green', classes='band-wrapper green-band'):
                yield Select[str]([], prompt='Green (Mid)', id='sel_green')

            # Red Slot
            with Vertical(id='wrap_red', classes='band-wrapper red-band'):
                yield Select[str]([], prompt='Red (Long)', id='sel_red')

            yield Button('Reset', variant='default', id='reset-btn')

    def on_mount(self) -> None:
        """Initialize."""
        # 1. Populate all options using band names as values
        all_opts: list[tuple[str, str]] = [(b.display, b.name) for b in BANDS]
        for slot in ['blue', 'green', 'red']:
            self.query_one(f'#sel_{slot}', Select).set_options(all_opts)

        # 2. Apply initial config if valid
        sorted_bands = self._sort_bands_by_rank(self.initial_bands)
        if len(sorted_bands) == 3:
            # Lock all slots immediately without animation/focus
            self._lock_slot('blue', sorted_bands[0], focus_next=False)
            self._lock_slot('green', sorted_bands[1], focus_next=False)
            self._lock_slot('red', sorted_bands[2], focus_next=False)

        self._update_options()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle selection: Lock the slot and move focus."""
        if event.value == Select.BLANK:
            return

        slot_map = {'sel_blue': 'blue', 'sel_green': 'green', 'sel_red': 'red'}

        if event.control.id in slot_map:
            slot_name = slot_map[event.control.id]
            self._lock_slot(slot_name, str(event.value), focus_next=True)
            self.post_message(self.Changed(self))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'reset-btn':
            self._reset_all()

    def _lock_slot(self, slot: str, band_name: str, focus_next: bool = True) -> None:
        """Replace the Select widget with a Label.

        Args:
            slot: Slot name ('blue', 'green', or 'red')
            band_name: The band name (e.g., 'cfis-r')
            focus_next: Whether to advance focus to the next slot
        """
        self.selections[slot] = band_name

        wrapper = self.query_one(f'#wrap_{slot}', Vertical)

        # 1. Find focus target BEFORE removing the current widget
        if focus_next:
            self._advance_focus(skip_widget=wrapper.query_one(Select))

        # 2. Swap UI - get display name from band info
        band_info = BAND_BY_NAME.get(band_name)
        display_name = band_info.display if band_info else band_name
        lbl = Label(f'✓ {display_name}', classes='locked-label')

        wrapper.remove_children()
        wrapper.mount(lbl)

        # 3. Update logic
        self._update_options()

    def _advance_focus(self, skip_widget: Widget | None = None) -> None:
        """Focus the next available Select widget or the Reset button."""
        # Find all Select widgets
        selects = list(self.query(Select))

        # Focus the first one that isn't the one we are currently removing
        for sel in selects:
            if sel is not skip_widget:
                sel.focus()
                return

        # If no selects left, focus Reset
        self.query_one('#reset-btn').focus()

    def _reset_all(self) -> None:
        """Revert all slots to Select widgets."""
        self.selections = {'blue': None, 'green': None, 'red': None}
        all_opts: list[tuple[str, str]] = [(b.display, b.name) for b in BANDS]

        slot_defs = [('blue', 'Blue (Short)'), ('green', 'Green (Mid)'), ('red', 'Red (Long)')]

        # Rebuild UI
        for slot, prompt in slot_defs:
            wrapper = self.query_one(f'#wrap_{slot}', Vertical)
            wrapper.remove_children()

            # Mount fresh Select
            new_sel: Select[str] = Select(all_opts, prompt=prompt, id=f'sel_{slot}')
            wrapper.mount(new_sel)

        # Reset Logic
        self._update_options()

        # Focus first slot
        try:
            self.query_one('#sel_blue').focus()
        except Exception:
            pass

        self.post_message(self.Changed(self))

    def _update_options(self) -> None:
        """
        Filter options for any remaining unlocked Select widgets.
        Iterates through slots and calculates valid options based on OTHER slots.
        """
        current_vals = [self.selections['blue'], self.selections['green'], self.selections['red']]
        slot_names = ['blue', 'green', 'red']

        def to_opts(valid_names: set[str]) -> list[tuple[str, str]]:
            """Convert set of band names to sorted Select options."""
            valid_bands = [b for b in BANDS if b.name in valid_names]
            valid_bands.sort(key=lambda x: (x.rank, x.name))
            return [(b.display, b.name) for b in valid_bands]

        with self.prevent(Select.Changed):
            for i, slot in enumerate(slot_names):
                # If slot is locked (value is set), skip it
                if current_vals[i] is not None:
                    continue

                # Find valid band names for this slot (i) based on others
                valid_names: set[str] = set()
                for combo in VALID_COMBINATIONS:
                    match = True
                    for other_i, other_val in enumerate(current_vals):
                        if i == other_i:
                            continue  # Skip self
                        # If other slot is set, combo must match it
                        if other_val is not None and combo[other_i].name != other_val:
                            match = False
                            break
                    if match:
                        valid_names.add(combo[i].name)

                # Update the Select widget
                try:
                    self.query_one(f'#sel_{slot}', Select).set_options(to_opts(valid_names))
                except Exception:
                    pass

    def _sort_bands_by_rank(self, band_names: list[str]) -> list[str]:
        """Sort band names by wavelength rank.

        Args:
            band_names: List of band names from config (e.g., ['whigs-g', 'cfis-r', 'ps-i'])

        Returns:
            List of band names sorted by wavelength (shortest to longest)
        """
        # Filter to only known bands and sort by rank
        valid_bands = [b for b in band_names if b in BAND_BY_NAME]
        valid_bands.sort(key=lambda name: BAND_BY_NAME[name].rank)
        return valid_bands

    def get_selected_bands(self) -> list[str]:
        """Get the selected bands as a list of band names.

        Returns:
            List of 3 band names in wavelength order [blue, green, red],
            or empty list if selection is incomplete.
        """
        blue = self.selections['blue']
        green = self.selections['green']
        red = self.selections['red']

        if blue and green and red:
            # Return in wavelength order (blue=short, green=mid, red=long)
            return [blue, green, red]
        return []

    def is_complete(self) -> bool:
        """Check if all three bands have been selected."""
        return len(self.get_selected_bands()) == 3
