"""Coordinate list widgets for editing tile and sky coordinate pairs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.validation import Validator
from textual.widgets import Button, Input, Label, Static

from unionsdata.tui.validators import FloatValidator, IntegerRange


class CoordinateRow(Static):
    """A single row with two coordinate inputs and a remove button."""

    DEFAULT_CSS = """
    CoordinateRow {
        height: 3;
        layout: horizontal;
    }

    CoordinateRow > Horizontal {
        width: 100%;
        height: 3;
    }

    CoordinateRow .coord-input {
    width: 18;
    min-width: 18;
    max-width: 18;
    margin-right: 1;
    }

    CoordinateRow .remove-btn {
        width: 3;
        min-width: 3;
    }

    CoordinateRow .row-index {
        height: 3;
        width: 4;
        min-width: 4;
        max-width: 4;
        text-align: right;
        content-align: right middle;
        padding-right: 1;
        color: $text-muted;
    }
    """

    class Removed(Message):
        """Message sent when row is removed."""

        def __init__(self, row: CoordinateRow) -> None:
            self.row = row
            super().__init__()

    class ValueChanged(Message):
        """Message sent when coordinate values change."""

        def __init__(self, row: CoordinateRow) -> None:
            self.row = row
            super().__init__()

    def __init__(
        self,
        index: int,
        value1: str = '',
        value2: str = '',
        *,
        label1: str = 'X',
        label2: str = 'Y',
        coord_type: str = 'int',
        id: str | None = None,
    ) -> None:
        """
        Initialize CoordinateRow.

        Args:
            index: Row index (for display)
            value1: First coordinate value
            value2: Second coordinate value
            label1: Label for first coordinate
            label2: Label for second coordinate
            coord_type: 'int' for tiles, 'float' for RA/Dec
            id: Widget ID
        """
        super().__init__(id=id)
        self._index = index
        self._value1 = value1
        self._value2 = value2
        self._label1 = label1
        self._label2 = label2
        self._coord_type = coord_type

    def compose(self) -> ComposeResult:
        # Define validators
        val1: Validator
        val2: Validator
        if self._coord_type == 'int':
            # Tile coordinates (0-999)
            val1 = val2 = IntegerRange(minimum=0, maximum=999)
        else:
            # Sky coordinates: RA (0-360) and Dec (-90 to +90)
            val1 = FloatValidator(minimum=0.0, maximum=360.0)
            val2 = FloatValidator(minimum=-90.0, maximum=90.0)

        with Horizontal():
            yield Static(f'{self._index + 1}.', classes='row-index')
            yield Input(
                value=self._value1,
                placeholder=self._label1,
                validators=[val1],
                classes='coord-input',
                id=f'{self.id}-input1' if self.id else None,
            )
            yield Input(
                value=self._value2,
                placeholder=self._label2,
                validators=[val2],
                classes='coord-input',
                id=f'{self.id}-input2' if self.id else None,
            )
            yield Button('✗', variant='error', classes='remove-btn')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle remove button press."""
        self.post_message(self.Removed(self))

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        self.post_message(self.ValueChanged(self))

    def get_values(self) -> tuple[str, str]:
        """Get the current coordinate values."""
        inputs = list(self.query('.coord-input').results(Input))
        if len(inputs) >= 2:
            return inputs[0].value, inputs[1].value
        return '', ''

    def is_valid(self) -> bool:
        """Check if both inputs are valid."""
        inputs = list(self.query('.coord-input').results(Input))
        if len(inputs) < 2:
            return False

        for inp in inputs:
            if not inp.value.strip():
                return False
            if inp.validators:
                for validator in inp.validators:
                    if not validator.validate(inp.value).is_valid:
                        return False
        return True


class CoordinateList(Static):
    """
    Editable list of coordinate pairs.

    Used for RA/Dec coordinate input.
    """

    DEFAULT_CSS = """
    CoordinateList {
        height: auto;
        min-height: 5;
        border: solid $primary;
        padding: 1;
    }

    CoordinateList .coord-header {
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }

     CoordinateList .header-spacer {
        width: 4;
        min-width: 4;
        max-width: 4;
    }

    CoordinateList .header-label {
        width: 18;
        min-width: 18;
        max-width: 18;
        text-align: center;
        color: $text-muted;
        margin-right: 1;
    }

    CoordinateList .coord-rows {
        height: auto;
    }

    CoordinateList .button-row {
        width: 100%;
        height: auto;
        margin-top: 1;
    }

    CoordinateList .button-spacer {
        width: 4;
        min-width: 4;
        max-width: 4;
    }

    CoordinateList .button-container {
        width: 38;
        min-width: 38;
        max-width: 38;
        height: 3;
        align: center middle;
    }

    CoordinateList .add-row-btn {
        width: auto;
    }

    CoordinateList .validation-feedback {
        width: 1fr;
        color: $error;
        height: 3;
        content-align: left middle;
        margin-left: 1;
    }

    CoordinateList .empty-message {
        color: $text-muted;
        text-style: italic;
        padding: 1;
    }
    """

    coordinates: reactive[list[tuple[float, float]]] = reactive(list, layout=True)

    class Changed(Message):
        """Message sent when coordinates change."""

        def __init__(
            self, coord_list: CoordinateList, coordinates: list[tuple[float, float]]
        ) -> None:
            self.coord_list = coord_list
            self.coordinates = coordinates
            super().__init__()

    def __init__(
        self,
        coordinates: list[tuple[float, float]] | None = None,
        *,
        label1: str = 'RA',
        label2: str = 'Dec',
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """
        Initialize CoordinateList.

        Args:
            coordinates: Initial list of (ra, dec) tuples
            label1: Label for first coordinate column
            label2: Label for second coordinate column
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self._initial_coords = coordinates or []
        self._label1 = label1
        self._label2 = label2
        self._row_counter = 0

    def compose(self) -> ComposeResult:
        with Horizontal(classes='coord-header'):
            yield Label('', classes='header-spacer')
            yield Label(self._label1, classes='header-label')
            yield Label(self._label2, classes='header-label')

        with Vertical(classes='coord-rows', id='coord-rows-container'):
            if self._initial_coords:
                for i, (v1, v2) in enumerate(self._initial_coords):
                    yield CoordinateRow(
                        index=i,
                        value1=str(v1),
                        value2=str(v2),
                        label1=self._label1,
                        label2=self._label2,
                        coord_type='float',
                        id=f'coord-row-{self._row_counter}',
                    )
                    self._row_counter += 1

        with Horizontal(classes='button-row'):
            yield Label('', classes='button-spacer')
            with Horizontal(classes='button-container'):
                yield Button('+ Add Coordinate', variant='primary', classes='add-row-btn')
            yield Label('', classes='validation-feedback')

    def on_mount(self) -> None:
        """Initialize coordinates on mount."""
        self.coordinates = list(self._initial_coords)
        self._update_validation_ui()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle add button press."""
        if 'add-row-btn' in event.button.classes:
            self._add_row()

    def on_coordinate_row_removed(self, event: CoordinateRow.Removed) -> None:
        """Handle row removal."""
        event.row.remove()
        self._update_indices(exclude=event.row)
        self._collect_values(exclude=event.row)

    def on_coordinate_row_value_changed(self, event: CoordinateRow.ValueChanged) -> None:
        """Handle value changes in rows."""
        self._collect_values()

    def _add_row(self) -> None:
        """Add a new coordinate row."""
        container = self.query_one('#coord-rows-container', Vertical)

        # Remove empty message if present
        for msg in container.query('.empty-message'):
            msg.remove()

        current_rows = list(container.query(CoordinateRow))
        new_index = len(current_rows)

        new_row = CoordinateRow(
            index=new_index,
            label1=self._label1,
            label2=self._label2,
            coord_type='float',
            id=f'coord-row-{self._row_counter}',
        )
        self._row_counter += 1
        container.mount(new_row)

    def _update_indices(self, exclude: CoordinateRow | None = None) -> None:
        """Update row indices, ignoring excluded row."""
        container = self.query_one('#coord-rows-container', Vertical)
        current_idx = 0
        for row in container.query(CoordinateRow):
            if row is exclude:
                continue
            index_label = row.query_one('.row-index', Static)
            index_label.update(f'{current_idx + 1}.')
            row._index = current_idx
            current_idx += 1

    def _update_validation_ui(self) -> None:
        """Update validation feedback visibility."""
        try:
            label = self.query_one('.validation-feedback', Label)
            if not self.coordinates:
                label.update('✗ At least one coordinate pair required')
            else:
                label.update('')
        except Exception:
            pass

    def _collect_values(self, exclude: CoordinateRow | None = None) -> None:
        """Collect all coordinate values, ignoring excluded row."""
        container = self.query_one('#coord-rows-container', Vertical)
        new_coords: list[tuple[float, float]] = []

        for row in container.query(CoordinateRow):
            if row is exclude:
                continue

            v1, v2 = row.get_values()
            if v1.strip() and v2.strip():
                try:
                    new_coords.append((float(v1), float(v2)))
                except ValueError:
                    pass

        self.coordinates = new_coords
        self.post_message(self.Changed(self, self.coordinates))
        self._update_validation_ui()

    def get_coordinates(self) -> list[tuple[float, float]]:
        """Get the current list of coordinates."""
        return list(self.coordinates)

    def set_coordinates(self, coordinates: list[tuple[float, float]]) -> None:
        """Set coordinates programmatically."""
        container = self.query_one('#coord-rows-container', Vertical)

        # Remove existing rows
        for row in list(container.query(CoordinateRow)):
            row.remove()
        for msg in list(container.query('.empty-message')):
            msg.remove()

        # Add new rows
        if coordinates:
            for i, (v1, v2) in enumerate(coordinates):
                new_row = CoordinateRow(
                    index=i,
                    value1=str(v1),
                    value2=str(v2),
                    label1=self._label1,
                    label2=self._label2,
                    coord_type='float',
                    id=f'coord-row-{self._row_counter}',
                )
                self._row_counter += 1
                container.mount(new_row)

        self.coordinates = list(coordinates)
        self._update_validation_ui()

    def is_valid(self) -> bool:
        """Check if valid (not empty and all rows valid)."""
        # Fail if empty
        if not self.coordinates:
            return False

        # Fail if any individual row is invalid
        container = self.query_one('#coord-rows-container', Vertical)
        for row in container.query(CoordinateRow):
            if not row.is_valid():
                return False
        return True


class TileList(Static):
    """
    Editable list of tile coordinate pairs.

    Used for tile X/Y coordinate input (integers).
    """

    DEFAULT_CSS = """
    TileList {
        height: auto;
        min-height: 5;
        border: solid $primary;
        padding: 1;
    }

    TileList .coord-header {
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    TileList .header-spacer {
        width: 4;
        min-width: 4;
        max-width: 4;
    }

    TileList .header-label {
        width: 18;
        min-width: 18;
        max-width: 18;
        text-align: center;
        color: $text-muted;
        margin-right: 1;
    }

    TileList .coord-rows {
        height: auto;
    }

    TileList .button-row {
        width: 100%;
        height: auto;
        margin-top: 1;
    }

    TileList .button-spacer {
        width: 4;
        min-width: 4;
        max-width: 4;
    }

    TileList .button-container {
        width: 38;
        min-width: 38;
        max-width: 38;
        height: 3;
        align: center middle;
    }

    TileList .add-row-btn {
        width: auto;
    }

    TileList .validation-feedback {
        width: 1fr;
        color: $error;
        height: 3;
        content-align: left middle;
        margin-left: 1;
    }

    TileList .empty-message {
        color: $text-muted;
        text-style: italic;
        padding: 1;
    }
    """

    tiles: reactive[list[tuple[int, int]]] = reactive(list, layout=True)

    class Changed(Message):
        """Message sent when tiles change."""

        def __init__(self, tile_list: TileList, tiles: list[tuple[int, int]]) -> None:
            self.tile_list = tile_list
            self.tiles = tiles
            super().__init__()

    def __init__(
        self,
        tiles: list[tuple[int, int]] | None = None,
        *,
        label1: str = 'XXX',
        label2: str = 'YYY',
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """
        Initialize TileList.

        Args:
            tiles: Initial list of (x, y) tile tuples
            label1: Label for first coordinate column (default 'XXX')
            label2: Label for second coordinate column (default 'YYY')
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self._initial_tiles = tiles or []
        self._label1 = label1
        self._label2 = label2
        self._row_counter = 0

    def compose(self) -> ComposeResult:
        with Horizontal(classes='coord-header'):
            yield Label('', classes='header-spacer')
            yield Label(self._label1, classes='header-label')
            yield Label(self._label2, classes='header-label')

        with Vertical(classes='coord-rows', id='tile-rows-container'):
            if self._initial_tiles:
                for i, (x, y) in enumerate(self._initial_tiles):
                    yield CoordinateRow(
                        index=i,
                        value1=str(x),
                        value2=str(y),
                        label1=self._label1,
                        label2=self._label2,
                        coord_type='int',
                        id=f'tile-row-{self._row_counter}',
                    )
                    self._row_counter += 1

        with Horizontal(classes='button-row'):
            yield Label('', classes='button-spacer')
            with Horizontal(classes='button-container'):
                yield Button('+ Add Tile', variant='primary', classes='add-row-btn')
            yield Label('', classes='validation-feedback')

    def on_mount(self) -> None:
        """Initialize tiles on mount."""
        self.tiles = list(self._initial_tiles)
        self._update_validation_ui()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle add button press."""
        if 'add-row-btn' in event.button.classes:
            self._add_row()

    def on_coordinate_row_removed(self, event: CoordinateRow.Removed) -> None:
        """Handle row removal."""
        event.row.remove()
        self._update_indices(exclude=event.row)
        self._collect_values(exclude=event.row)

    def on_coordinate_row_value_changed(self, event: CoordinateRow.ValueChanged) -> None:
        """Handle value changes in rows."""
        self._collect_values()

    def _add_row(self) -> None:
        """Add a new tile row."""
        container = self.query_one('#tile-rows-container', Vertical)

        # Remove empty message if present
        for msg in container.query('.empty-message'):
            msg.remove()

        current_rows = list(container.query(CoordinateRow))
        new_index = len(current_rows)

        new_row = CoordinateRow(
            index=new_index,
            label1=self._label1,
            label2=self._label2,
            coord_type='int',
            id=f'tile-row-{self._row_counter}',
        )
        self._row_counter += 1
        container.mount(new_row)

    def _update_indices(self, exclude: CoordinateRow | None = None) -> None:
        """Update row indices, ignoring excluded row."""
        container = self.query_one('#tile-rows-container', Vertical)
        current_idx = 0
        for row in container.query(CoordinateRow):
            if row is exclude:
                continue
            index_label = row.query_one('.row-index', Static)
            index_label.update(f'{current_idx + 1}.')
            row._index = current_idx
            current_idx += 1

    def _update_validation_ui(self) -> None:
        """Update validation feedback visibility."""
        try:
            label = self.query_one('.validation-feedback', Label)
            if not self.tiles:
                label.update('✗ At least one tile required')
            else:
                label.update('')
        except Exception:
            pass

    def _collect_values(self, exclude: CoordinateRow | None = None) -> None:
        """Collect values, ignoring excluded row."""
        container = self.query_one('#tile-rows-container', Vertical)
        new_tiles: list[tuple[int, int]] = []

        for row in container.query(CoordinateRow):
            if row is exclude:
                continue

            v1, v2 = row.get_values()
            if v1.strip() and v2.strip():
                try:
                    new_tiles.append((int(v1), int(v2)))
                except ValueError:
                    pass

        self.tiles = new_tiles
        self.post_message(self.Changed(self, self.tiles))
        self._update_validation_ui()

    def get_tiles(self) -> list[tuple[int, int]]:
        """Get the current list of tiles."""
        return list(self.tiles)

    def set_tiles(self, tiles: list[tuple[int, int]]) -> None:
        """Set tiles programmatically."""
        container = self.query_one('#tile-rows-container', Vertical)

        # Remove existing rows
        for row in list(container.query(CoordinateRow)):
            row.remove()
        for msg in list(container.query('.empty-message')):
            msg.remove()

        # Add new rows
        if tiles:
            for i, (x, y) in enumerate(tiles):
                new_row = CoordinateRow(
                    index=i,
                    value1=str(x),
                    value2=str(y),
                    label1=self._label1,
                    label2=self._label2,
                    coord_type='int',
                    id=f'tile-row-{self._row_counter}',
                )
                self._row_counter += 1
                container.mount(new_row)

        self.tiles = list(tiles)
        self._update_validation_ui()

    def is_valid(self) -> bool:
        """Check if valid (not empty and all rows valid)."""
        # Fail if empty
        if not self.tiles:
            return False

        # Fail if any individual row is invalid
        container = self.query_one('#tile-rows-container', Vertical)
        for row in container.query(CoordinateRow):
            if not row.is_valid():
                return False
        return True
