"""RGB band selector widget with wavelength-aware constraints."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, Select, Static


@dataclass(frozen=True)
class BandFilter:
    id: str
    name: str
    rank: int


FILTERS = [
    BandFilter('u', 'cfis-u', 1),
    BandFilter('g', 'whigs-g', 2),
    BandFilter('r1', 'cfis-r', 3),  # Variant 1
    BandFilter('r2', 'cfis_lsb-r', 3),  # Variant 2 (Same rank!)
    BandFilter('i', 'ps-i', 4),
    BandFilter('z1', 'wishes-z', 5),  # Variant 1
    BandFilter('z2', 'ps-z', 5),  # Variant 2 (Same rank!)
]

# Precompute all valid (red, green, blue) combinations
VALID_WORLDS = [
    (f1, f2, f3)
    for f1 in FILTERS
    for f2 in FILTERS
    for f3 in FILTERS
    if f1.rank < f2.rank < f3.rank
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
        self.selections: dict[str, str | None] = {'blue': None, 'green': None, 'red': None}

    def compose(self) -> ComposeResult:
        with Horizontal():
            # Blue Slot
            with Vertical(id='wrap_blue', classes='band-wrapper blue-band'):
                yield Select([], prompt='Blue (Short)', id='sel_blue')

            # Green Slot
            with Vertical(id='wrap_green', classes='band-wrapper green-band'):
                yield Select([], prompt='Green (Mid)', id='sel_green')

            # Red Slot
            with Vertical(id='wrap_red', classes='band-wrapper red-band'):
                yield Select([], prompt='Red (Long)', id='sel_red')

            yield Button('Reset', variant='default', id='reset-btn')

    def on_mount(self) -> None:
        """Initialize."""
        # 1. Populate all options
        all_opts = [(f.name, f.id) for f in FILTERS]
        for slot in ['blue', 'green', 'red']:
            self.query_one(f'#sel_{slot}', Select).set_options(all_opts)

        # 2. Apply initial config if valid
        sorted_bands = self._sort_bands_by_rank(self.initial_bands)
        if len(sorted_bands) == 3:
            # Lock all slots immediately without animation/focus
            self._lock_slot('blue', sorted_bands[0], focus_next=False)
            self._lock_slot('green', sorted_bands[1], focus_next=False)
            self._lock_slot('red', sorted_bands[2], focus_next=False)

        self.update_options()

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

    def _lock_slot(self, slot: str, value: str, focus_next: bool = True) -> None:
        """Replace the Select widget with a Label."""
        self.selections[slot] = value

        wrapper = self.query_one(f'#wrap_{slot}', Vertical)

        # 1. Find focus target BEFORE removing the current widget
        if focus_next:
            self._advance_focus(skip_widget=wrapper.query_one(Select))

        # 2. Swap UI
        display_name = next((f.name for f in FILTERS if f.id == value), value)
        lbl = Label(f'✓ {display_name}', classes='locked-label')

        wrapper.remove_children()
        wrapper.mount(lbl)

        # 3. Update logic
        self.update_options()

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
        all_opts = [(f.name, f.id) for f in FILTERS]

        slot_defs = [('blue', 'Blue (Short)'), ('green', 'Green (Mid)'), ('red', 'Red (Long)')]

        # Rebuild UI
        for slot, prompt in slot_defs:
            wrapper = self.query_one(f'#wrap_{slot}', Vertical)
            wrapper.remove_children()

            # Mount fresh Select
            new_sel = Select(all_opts, prompt=prompt, id=f'sel_{slot}')
            wrapper.mount(new_sel)

        # Reset Logic
        self.update_options()

        # Focus first slot
        try:
            self.query_one('#sel_blue').focus()
        except Exception:
            pass

        self.post_message(self.Changed(self))

    def update_options(self) -> None:
        """
        Filter options for any remaining unlocked Select widgets.
        Iterates through slots and calculates valid options based on OTHER slots.
        """
        current_vals = [self.selections['blue'], self.selections['green'], self.selections['red']]
        slot_names = ['blue', 'green', 'red']

        def to_opts(id_set: set[str]) -> list[tuple[str, str]]:
            objs = [f for f in FILTERS if f.id in id_set]
            objs.sort(key=lambda x: (x.rank, x.name))
            return [(x.name, x.id) for x in objs]

        with self.prevent(Select.Changed):
            for i, slot in enumerate(slot_names):
                # If slot is locked (value is set), skip it
                if current_vals[i] is not None:
                    continue

                # Find valid IDs for this slot (i) based on others
                valid_ids = set()
                for w in VALID_WORLDS:
                    match = True
                    for other_i, other_val in enumerate(current_vals):
                        if i == other_i:
                            continue  # Skip self
                        # If other slot is set, world must match it
                        if other_val is not None and w[other_i].id != other_val:
                            match = False
                            break
                    if match:
                        valid_ids.add(w[i].id)

                # Update the Select widget
                try:
                    self.query_one(f'#sel_{slot}', Select).set_options(to_opts(valid_ids))
                except Exception:
                    pass

    def _sort_bands_by_rank(self, band_ids: list[str]) -> list[str]:
        rank_map = {f.id: f.rank for f in FILTERS}
        valid_bands = [b for b in band_ids if b in rank_map]
        valid_bands.sort(key=lambda b: rank_map[b])
        return valid_bands

    def get_selected_bands(self) -> list[str]:
        b = self.selections['blue']
        g = self.selections['green']
        r = self.selections['red']
        if b and g and r:
            return [str(b), str(g), str(r)]
        return []

    def is_complete(self) -> bool:
        return len(self.get_selected_bands()) == 3
