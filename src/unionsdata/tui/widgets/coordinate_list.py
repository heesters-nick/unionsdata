"""Coordinate list widgets for editing tile and sky coordinate pairs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Static

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
        width: 15;
        margin-right: 1;
    }

    CoordinateRow .remove-btn {
        width: 3;
        min-width: 3;
    }

    CoordinateRow .row-index {
        width: 4;
        text-align: right;
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
        validator = (
            IntegerRange(minimum=0, maximum=999)
            if self._coord_type == 'int'
            else FloatValidator()
        )

        with Horizontal():
            yield Static(f'{self._index + 1}.', classes='row-index')
            yield Input(
                value=self._value1,
                placeholder=self._label1,
                validators=[validator],
                classes='coord-input',
                id=f'{self.id}-input1' if self.id else None,
            )
            yield Input(
                value=self._value2,
                placeholder=self._label2,
                validators=[validator],
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
        max-height: 20;
        border: solid $primary;
        padding: 1;
    }

    CoordinateList .coord-header {
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    CoordinateList .coord-rows {
        height: auto;
    }

    CoordinateList .add-row-btn {
        margin-top: 1;
        width: auto;
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
        yield Static(f'     {self._label1:^15} {self._label2:^15}', classes='coord-header')
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
            else:
                yield Static('No coordinates added', classes='empty-message')
        yield Button('+ Add Coordinate', variant='primary', classes='add-row-btn')

    def on_mount(self) -> None:
        """Initialize coordinates on mount."""
        self.coordinates = list(self._initial_coords)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle add button press."""
        if 'add-row-btn' in event.button.classes:
            self._add_row()

    def on_coordinate_row_removed(self, event: CoordinateRow.Removed) -> None:
        """Handle row removal."""
        event.row.remove()
        self._update_indices()
        self._collect_values()

        # Show empty message if no rows left
        container = self.query_one('#coord-rows-container', Vertical)
        if not list(container.query(CoordinateRow)):
            container.mount(Static('No coordinates added', classes='empty-message'))

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

    def _update_indices(self) -> None:
        """Update row indices after removal."""
        container = self.query_one('#coord-rows-container', Vertical)
        for i, row in enumerate(container.query(CoordinateRow)):
            index_label = row.query_one('.row-index', Static)
            index_label.update(f'{i + 1}.')
            row._index = i

    def _collect_values(self) -> None:
        """Collect all coordinate values from rows."""
        container = self.query_one('#coord-rows-container', Vertical)
        new_coords: list[tuple[float, float]] = []

        for row in container.query(CoordinateRow):
            v1, v2 = row.get_values()
            if v1.strip() and v2.strip():
                try:
                    new_coords.append((float(v1), float(v2)))
                except ValueError:
                    pass

        self.coordinates = new_coords
        self.post_message(self.Changed(self, self.coordinates))

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
        else:
            container.mount(Static('No coordinates added', classes='empty-message'))

        self.coordinates = list(coordinates)

    def is_valid(self) -> bool:
        """Check if all rows have valid values."""
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
        max-height: 20;
        border: solid $primary;
        padding: 1;
    }

    TileList .coord-header {
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    TileList .coord-rows {
        height: auto;
    }

    TileList .add-row-btn {
        margin-top: 1;
        width: auto;
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
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """
        Initialize TileList.

        Args:
            tiles: Initial list of (x, y) tile tuples
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self._initial_tiles = tiles or []
        self._row_counter = 0

    def compose(self) -> ComposeResult:
        yield Static('     X              Y', classes='coord-header')
        with Vertical(classes='coord-rows', id='tile-rows-container'):
            if self._initial_tiles:
                for i, (x, y) in enumerate(self._initial_tiles):
                    yield CoordinateRow(
                        index=i,
                        value1=str(x),
                        value2=str(y),
                        label1='X',
                        label2='Y',
                        coord_type='int',
                        id=f'tile-row-{self._row_counter}',
                    )
                    self._row_counter += 1
            else:
                yield Static('No tiles added', classes='empty-message')
        yield Button('+ Add Tile', variant='primary', classes='add-row-btn')

    def on_mount(self) -> None:
        """Initialize tiles on mount."""
        self.tiles = list(self._initial_tiles)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle add button press."""
        if 'add-row-btn' in event.button.classes:
            self._add_row()

    def on_coordinate_row_removed(self, event: CoordinateRow.Removed) -> None:
        """Handle row removal."""
        event.row.remove()
        self._update_indices()
        self._collect_values()

        # Show empty message if no rows left
        container = self.query_one('#tile-rows-container', Vertical)
        if not list(container.query(CoordinateRow)):
            container.mount(Static('No tiles added', classes='empty-message'))

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
            label1='X',
            label2='Y',
            coord_type='int',
            id=f'tile-row-{self._row_counter}',
        )
        self._row_counter += 1
        container.mount(new_row)

    def _update_indices(self) -> None:
        """Update row indices after removal."""
        container = self.query_one('#tile-rows-container', Vertical)
        for i, row in enumerate(container.query(CoordinateRow)):
            index_label = row.query_one('.row-index', Static)
            index_label.update(f'{i + 1}.')
            row._index = i

    def _collect_values(self) -> None:
        """Collect all tile values from rows."""
        container = self.query_one('#tile-rows-container', Vertical)
        new_tiles: list[tuple[int, int]] = []

        for row in container.query(CoordinateRow):
            v1, v2 = row.get_values()
            if v1.strip() and v2.strip():
                try:
                    new_tiles.append((int(v1), int(v2)))
                except ValueError:
                    pass

        self.tiles = new_tiles
        self.post_message(self.Changed(self, self.tiles))

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
                    label1='X',
                    label2='Y',
                    coord_type='int',
                    id=f'tile-row-{self._row_counter}',
                )
                self._row_counter += 1
                container.mount(new_row)
        else:
            container.mount(Static('No tiles added', classes='empty-message'))

        self.tiles = list(tiles)

    def is_valid(self) -> bool:
        """Check if all rows have valid values."""
        container = self.query_one('#tile-rows-container', Vertical)
        for row in container.query(CoordinateRow):
            if not row.is_valid():
                return False
        return True
