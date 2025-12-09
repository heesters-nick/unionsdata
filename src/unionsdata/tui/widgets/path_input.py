"""Path input widget with validation indicator."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static

from unionsdata.tui.validators import (
    CertificateValidator,
    NonEmptyValidator,
    PathExistsValidator,
    ValidationResult,
    Validator,
)


class PathInput(Static):
    """Path input field using native Input validation."""

    DEFAULT_CSS = """
    PathInput {
        height: 3;
        layout: horizontal;
        width: 100%;
    }

    PathInput .path-input {
        width: 60;
        min-width: 60;
    }

    PathInput .path-status {
        width: 3;
        min-width: 3;
        height: 100%;
        text-align: center;
        content-align: center middle;
    }

    PathInput .path-status.valid { color: $success; }
    PathInput .path-status.invalid { color: $error; }
    PathInput .path-status.warning { color: $warning; }

    PathInput .path-tooltip {
        width: 1fr;
        height: 100%;
        content-align: left middle;
        color: $text-muted;
        padding-left: 1;
    }
    """

    value: reactive[str] = reactive('', layout=True)

    class Changed(Message):
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
        super().__init__(id=id, classes=classes)
        self._initial_value = value
        self._placeholder = placeholder
        self._is_certificate = is_certificate

        # 1. Determine the correct validator
        self._validator: Validator | None = None

        if is_certificate:
            self._validator = CertificateValidator(warning_days=2)
        elif must_exist:
            self._validator = PathExistsValidator(
                must_be_file=must_be_file, must_be_dir=must_be_dir
            )
        else:
            # If path doesn't need to exist, we only check if it is empty
            self._validator = NonEmptyValidator()

    def compose(self) -> ComposeResult:
        yield Input(
            value=self._initial_value,
            placeholder=self._placeholder,
            classes='path-input',
            id=f'{self.id}-input' if self.id else None,
            validators=[self._validator] if self._validator else [],
        )
        yield Static('', classes='path-status')
        yield Static('', classes='path-tooltip')

    def on_mount(self) -> None:
        self.value = self._initial_value
        self.query_one(Input).validate(self.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        if 'path-input' in event.input.classes:
            self.value = event.value
            self._update_ui_state(event.validation_result)
            self.post_message(self.Changed(self, event.value))

    def _update_ui_state(self, result: ValidationResult | None) -> None:
        status_widget = self.query_one('.path-status', Static)
        tooltip_widget = self.query_one('.path-tooltip', Static)
        input_widget = self.query_one(Input)

        # Clear explicit warning class first
        input_widget.remove_class('warning')
        status_widget.remove_class('valid', 'invalid', 'warning')

        # 1. EMPTY CASE (Handled natively by NonEmptyValidator -> Invalid)
        if not result or result.is_valid:
            # --- VALID CASE ---

            # Special Check: Certificate Warnings (Orange)
            warning_msg = None
            if self._is_certificate and isinstance(self._validator, CertificateValidator):
                warning_msg = self._validator.get_expiry_warning(self.value)

            if warning_msg:
                # Warning State: Orange Icon + Message
                status_widget.update('⚠')
                status_widget.add_class('warning')
                input_widget.add_class('warning')
                tooltip_widget.update(warning_msg)
            else:
                # Standard State: Green Border (Automatic) + NO Icon
                status_widget.update('')
                tooltip_widget.update('')

        else:
            # --- INVALID CASE (Red) ---
            # Input is already Red because of native validation
            status_widget.update('✗')
            status_widget.add_class('invalid')

            msg = result.failure_descriptions[0] if result.failure_descriptions else ''
            tooltip_widget.update(msg)

    def set_value(self, value: str) -> None:
        self.value = value
        self.query_one(Input).value = value

    def is_valid(self) -> bool:
        return self.query_one(Input).is_valid
