"""Path input widget with validation indicator."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static

from unionsdata.tui.validators import CertificateValidator, PathExistsValidator


class PathInput(Static):
    """
    Path input field with real-time validation indicator.

    Shows a status indicator next to the input:
    - ✓ (green): Path exists and is valid
    - ✗ (red): Path does not exist or is invalid
    - ⚠ (yellow): Warning (e.g., certificate expiring soon)
    """

    DEFAULT_CSS = """
    PathInput {
        height: 3;
        layout: horizontal;
    }

    PathInput > Horizontal {
        width: 100%;
        height: 3;
    }

    PathInput .path-input {
        width: 1fr;
    }

    PathInput .path-status {
        width: 3;
        text-align: center;
        padding: 0 1;
    }

    PathInput .path-status.valid {
        color: $success;
    }

    PathInput .path-status.invalid {
        color: $error;
    }

    PathInput .path-status.warning {
        color: $warning;
    }

    PathInput .path-tooltip {
        width: auto;
        max-width: 40;
        color: $text-muted;
        padding-left: 1;
    }
    """

    value: reactive[str] = reactive('', layout=True)

    class Changed(Message):
        """Message sent when the path value changes."""

        def __init__(self, path_input: PathInput, value: str) -> None:
            self.path_input = path_input
            self.value = value
            super().__init__()

        @property
        def control(self) -> PathInput:
            return self.path_input

    def __init__(
        self,
        value: str = '',
        *,
        placeholder: str = 'Enter path...',
        must_exist: bool = True,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        is_certificate: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """
        Initialize PathInput.

        Args:
            value: Initial path value
            placeholder: Placeholder text when empty
            must_exist: Whether the path must exist for validation
            must_be_file: Whether the path must be a file
            must_be_dir: Whether the path must be a directory
            is_certificate: Whether to validate as a certificate file
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self._initial_value = value
        self._placeholder = placeholder
        self._must_exist = must_exist
        self._must_be_file = must_be_file
        self._must_be_dir = must_be_dir
        self._is_certificate = is_certificate

        # Set up validators
        self._validator: CertificateValidator | PathExistsValidator | None
        if is_certificate:
            self._validator = CertificateValidator()
        elif must_exist:
            self._validator = PathExistsValidator(
                must_be_file=must_be_file, must_be_dir=must_be_dir
            )
        else:
            self._validator = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Input(
                value=self._initial_value,
                placeholder=self._placeholder,
                classes='path-input',
                id=f'{self.id}-input' if self.id else None,
            )
            yield Static('', classes='path-status')
            yield Static('', classes='path-tooltip')

    def on_mount(self) -> None:
        """Initialize value and validation on mount."""
        self.value = self._initial_value
        self._update_status()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if 'path-input' in event.input.classes:
            self.value = event.value
            self._update_status()
            self.post_message(self.Changed(self, event.value))

    def _update_status(self) -> None:
        """Update the status indicator based on validation."""
        status_widget = self.query_one('.path-status', Static)
        tooltip_widget = self.query_one('.path-tooltip', Static)

        if not self.value.strip():
            status_widget.update('')
            status_widget.remove_class('valid', 'invalid', 'warning')
            tooltip_widget.update('')
            return

        if self._validator is None:
            status_widget.update('✓')
            status_widget.remove_class('invalid', 'warning')
            status_widget.add_class('valid')
            tooltip_widget.update('')
            return

        result = self._validator.validate(self.value)

        if result.is_valid:
            # Check for certificate warnings
            if self._is_certificate:
                path = Path(self.value).expanduser()
                if path.exists():
                    is_warning, msg = self._validator.get_expiry_info(path)  # type: ignore
                    if is_warning:
                        status_widget.update('⚠')
                        status_widget.remove_class('valid', 'invalid')
                        status_widget.add_class('warning')
                        tooltip_widget.update(msg)
                        return

            status_widget.update('✓')
            status_widget.remove_class('invalid', 'warning')
            status_widget.add_class('valid')
            tooltip_widget.update('')
        else:
            status_widget.update('✗')
            status_widget.remove_class('valid', 'warning')
            status_widget.add_class('invalid')
            failure_msg = result.failure_descriptions[0] if result.failure_descriptions else ''
            tooltip_widget.update(failure_msg)

    def set_value(self, value: str) -> None:
        """Programmatically set the path value."""
        self.value = value
        input_widget = self.query_one('.path-input', Input)
        input_widget.value = value
        self._update_status()

    def is_valid(self) -> bool:
        """Check if the current value is valid."""
        if self._validator is None:
            return True
        return self._validator.validate(self.value).is_valid
