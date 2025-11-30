"""Info icon widget with tooltip."""

from __future__ import annotations

from typing import Any

from textual.widgets import Label


class InfoIcon(Label):
    """
    A small icon that displays a tooltip when hovered.
    Used for showing help text without cluttering the UI.
    """

    DEFAULT_CSS = """
    InfoIcon {
        width: 3;
        content-align: center middle;
        color: $text-muted;
        margin-left: 1;
        text-style: bold;
    }

    InfoIcon:hover {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self, tooltip: str, **kwargs: Any) -> None:
        """
        Initialize the InfoIcon.

        Args:
            tooltip: The text to display when hovering over the icon.
            **kwargs: Additional arguments passed to Label.
        """
        super().__init__('([italic]i[/italic])', **kwargs)
        self.tooltip = tooltip
