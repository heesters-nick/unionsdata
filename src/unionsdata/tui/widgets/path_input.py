"""Path input widget with validation indicator."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.color import Color
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Static

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

    PathInput Button {
        height: 100%;
        min-width: 7;
        margin-left: 1;
        margin-right: 1;
        border: none;
    }

    PathInput .path-tooltip {
        width: auto;
        max-width: 1fr;
        height: 100%;
        content-align: left middle;
        color: $text-muted;
        padding-left: 1;
        padding-right: 1;
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
        action_button: Button | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._initial_value = value
        self._placeholder = placeholder
        self._is_certificate = is_certificate
        self._action_button = action_button
        self._pulsing = False
        self._mounted = False
        self._validator: Validator | None = None

        if is_certificate:
            self._validator = CertificateValidator(warning_days=1)
        elif must_exist:
            self._validator = PathExistsValidator(
                must_be_file=must_be_file, must_be_dir=must_be_dir
            )
        else:
            # if path doesn't need to exist, we only check if it is empty
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

        if self._action_button:
            yield self._action_button

    def on_mount(self) -> None:
        self._mounted = True
        self.value = self._initial_value
        self.query_one(Input).validate(self.value)

    def on_unmount(self) -> None:
        """Clean up animations when widget is removed."""
        self._mounted = False
        self._stop_pulse()

    def on_input_changed(self, event: Input.Changed) -> None:
        if 'path-input' in event.input.classes:
            self.value = event.value
            self._update_ui_state(event.validation_result)
            self.post_message(self.Changed(self, event.value))

    def _start_pulse(self, color_name: str) -> None:
        """Start a pulsing tint animation on the action button."""
        btn = self._action_button
        if not btn or not self._mounted:
            return

        self._pulsing = True

        # fetch CSS variables to get exact theme colors (e.g. $error, $warning)
        css_vars = self.app.get_css_variables()
        raw_target = css_vars.get(color_name, color_name)

        try:
            base_color = Color.parse(raw_target)
        except Exception:
            # Fallback if parsing fails
            base_color = Color.parse('#FF0000')

        # Pulse between 0% tint (original look) and 80% tint (alert overlay)
        tint_low = base_color.with_alpha(0.0)
        tint_high = base_color.with_alpha(0.5)

        def animate_up() -> None:
            if not self._pulsing or not self._mounted:
                if btn:
                    btn.styles.tint = None
                return
            try:
                btn.styles.animate(
                    'tint',
                    value=tint_high,
                    duration=1.0,
                    easing='in_out_cubic',
                    on_complete=animate_down,
                )
            except Exception:
                # Widget may have been removed
                self._pulsing = False

        def animate_down() -> None:
            if not self._pulsing or not self._mounted:
                if btn:
                    btn.styles.tint = None
                return
            try:
                btn.styles.animate(
                    'tint',
                    value=tint_low,
                    duration=1.0,
                    easing='in_out_cubic',
                    on_complete=animate_up,
                )
            except Exception:
                # Widget may have been removed
                self._pulsing = False

        # Start animation sequence
        btn.styles.tint = tint_low
        animate_up()

    def _stop_pulse(self) -> None:
        """Stop any active animation and reset styles."""
        self._pulsing = False  # stop the animation loop

        if not self._action_button:
            return

        btn = self._action_button

        # stop pulsing
        try:
            btn.styles.tint = None
        except Exception:
            pass  # Widget may already be gone

        def ensure_reset() -> None:
            if not self._pulsing:
                btn.styles.tint = None

        # reset the button to neutral state
        self.set_timer(0.1, ensure_reset)

    def _update_ui_state(self, result: ValidationResult | None) -> None:
        status_widget = self.query_one('.path-status', Static)
        tooltip_widget = self.query_one('.path-tooltip', Static)
        input_widget = self.query_one(Input)

        # Clear explicit warning class
        input_widget.remove_class('warning')
        status_widget.remove_class('valid', 'invalid', 'warning')
        self._stop_pulse()

        # certificate is valid
        if not result or result.is_valid:
            warning_msg = None
            if self._is_certificate and isinstance(self._validator, CertificateValidator):
                # check if cert is expiring soon
                warning_msg = self._validator.get_expiry_warning(self.value)

            if warning_msg:
                # warning icon + message
                status_widget.update('⚠️')
                status_widget.add_class('warning')
                input_widget.add_class('warning')
                tooltip_widget.update(warning_msg)

                # Pulse the button yellow to indicate the certificate is expiring soon and should be renewed soon
                if self._action_button:
                    self._start_pulse('warning')
            else:
                # Standard State: Green Border (Automatic) + NO Icon
                status_widget.update('')
                tooltip_widget.update('')
        # certificate is invalid
        else:
            status_widget.update('✗')
            status_widget.add_class('invalid')

            # get failure message
            msg = result.failure_descriptions[0] if result.failure_descriptions else ''
            tooltip_widget.update(msg)

            # pulse the button red to indicate error (missing/expired certificate). renewal needed.
            if self._action_button and self._is_certificate:
                self._start_pulse('error')

    def set_value(self, value: str) -> None:
        self.value = value
        self.query_one(Input).value = value

    def force_validate(self) -> None:
        """Manually trigger validation to update UI. We need this after renewing or creating a certificate."""
        # get the inner input widget
        input_widget = self.query_one(Input)

        # this checks the file on disk *now*
        result = input_widget.validate(self.value)

        # remove any error/warning states if fixed
        self._update_ui_state(result)

    def is_valid(self) -> bool:
        """Check validity dynamically."""
        # Run the validator fresh.
        result = self.query_one(Input).validate(self.value)

        # If result is None, no validation was performed -> consider it valid
        if result is None:
            return True

        return result.is_valid
