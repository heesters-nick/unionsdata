"""Custom checkbox widget with improved visual feedback."""

from __future__ import annotations

from rich.text import Text
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static


class BetterCheckbox(Static, can_focus=True):
    """
    A checkbox with clear visual states.

    Unchecked: [ ] (empty box, dimmed)
    Checked: [✓] (green checkmark)

    Supports mouse clicks and keyboard (Enter/Space to toggle).
    """

    DEFAULT_CSS = """
    BetterCheckbox {
        width: auto;
        height: 1;
        padding: 0 1;
        background: transparent;
    }

    BetterCheckbox:hover {
        background: $surface-lighten-1;
    }

    BetterCheckbox:focus {
        background: $primary-darken-2;
        text-style: bold;
    }
    """

    BINDINGS = [
        ('enter', 'toggle_value', 'Toggle'),
        ('space', 'toggle_value', 'Toggle'),
    ]

    value: reactive[bool] = reactive(False, init=False)

    class Changed(Message):
        """Posted when the checkbox value changes."""

        def __init__(self, checkbox: BetterCheckbox, value: bool) -> None:
            self.checkbox = checkbox
            self.value = value
            super().__init__()

        @property
        def control(self) -> BetterCheckbox:
            """The checkbox that sent the message."""
            return self.checkbox

    def __init__(
        self,
        label: str = '',
        value: bool = False,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """
        Initialize the checkbox.

        Args:
            label: Text label displayed next to the checkbox
            value: Initial checked state
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self._label = label
        self._initial_value = value

    def on_mount(self) -> None:
        """Set initial value after mounting."""
        self.value = self._initial_value

    def render(self) -> Text:
        """Render the checkbox with visual state."""
        text = Text()

        if self.value:
            # Checked: green checkmark
            text.append('[', style='dim')
            text.append('✓', style='bold green')
            text.append(']', style='dim')
        else:
            # Unchecked: empty box
            text.append('[ ]', style='dim')

        if self._label:
            text.append(' ')
            text.append(self._label)

        return text

    def on_click(self) -> None:
        """Toggle on mouse click."""
        self.action_toggle_value()

    def action_toggle_value(self) -> None:
        """Toggle the checkbox value."""
        self.value = not self.value

    def watch_value(self, value: bool) -> None:
        """React to value changes."""
        self.refresh()
        self.post_message(self.Changed(self, value))
