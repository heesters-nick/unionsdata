"""Custom validators for the TUI configuration editor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from textual.validation import ValidationResult, Validator

if TYPE_CHECKING:
    from cryptography.x509 import Certificate


class IntegerRange(Validator):
    """Validate that input is an integer within a specified range."""

    def __init__(self, minimum: int = 0, maximum: int | None = None) -> None:
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.failure('Value is required')

        try:
            int_value = int(value)
        except ValueError:
            return self.failure('Must be a whole number')

        if int_value < self.minimum:
            return self.failure(f'Must be at least {self.minimum}')

        if self.maximum is not None and int_value > self.maximum:
            return self.failure(f'Must be at most {self.maximum}')

        return self.success()


class FloatValidator(Validator):
    """Validate that input is a valid float."""

    def __init__(self, allow_negative: bool = True) -> None:
        super().__init__()
        self.allow_negative = allow_negative

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.failure('Value is required')

        try:
            float_value = float(value)
        except ValueError:
            return self.failure('Must be a number')

        if not self.allow_negative and float_value < 0:
            return self.failure('Must be non-negative')

        return self.success()


class PathExistsValidator(Validator):
    """Validate that a path exists on the filesystem."""

    def __init__(self, must_be_file: bool = False, must_be_dir: bool = False) -> None:
        super().__init__()
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.failure('Path is required')

        path = Path(value).expanduser()

        if not path.exists():
            return self.failure('Path does not exist')

        if self.must_be_file and not path.is_file():
            return self.failure('Must be a file')

        if self.must_be_dir and not path.is_dir():
            return self.failure('Must be a directory')

        return self.success()


class CertificateValidator(Validator):
    """Validate that a certificate file exists and is not expired."""

    def __init__(self, warning_days: int = 5) -> None:
        super().__init__()
        self.warning_days = warning_days

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.failure('Certificate path is required')

        path = Path(value).expanduser()

        if not path.exists():
            return self.failure('Certificate file does not exist')

        if not path.is_file():
            return self.failure('Must be a file')

        # Try to validate the certificate
        try:
            cert = self._load_certificate(path)
            expiry = cert.not_valid_after_utc
            time_left = expiry - datetime.now(UTC)

            if time_left.total_seconds() < 0:
                return self.failure(f'Certificate EXPIRED on {expiry.date()}')

            if time_left < timedelta(days=self.warning_days):
                # This is a warning, not a failure - we return success but the
                # PathInput widget will show a warning indicator
                return self.success()

        except Exception as e:
            return self.failure(f'Cannot read certificate: {e}')

        return self.success()

    def get_expiry_info(self, path: Path) -> tuple[bool, str]:
        """
        Get certificate expiry information.

        Returns:
            Tuple of (is_warning, message)
        """
        try:
            cert = self._load_certificate(path)
            expiry = cert.not_valid_after_utc
            time_left = expiry - datetime.now(UTC)

            if time_left.total_seconds() < 0:
                return True, f'EXPIRED on {expiry.date()}'

            if time_left < timedelta(days=self.warning_days):
                return True, f'Expires in {time_left.days} days'

            return False, f'Valid until {expiry.date()}'

        except Exception as e:
            return True, f'Cannot read: {e}'

    def _load_certificate(self, path: Path) -> Certificate:
        """Load a PEM certificate from file."""
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend

        with open(path, 'rb') as f:
            return x509.load_pem_x509_certificate(f.read(), default_backend())


class NonEmptyValidator(Validator):
    """Validate that input is not empty."""

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.failure('Value is required')
        return self.success()


class CoordinatePairValidator(Validator):
    """Validate coordinate pairs (RA/Dec or tile X/Y)."""

    def __init__(self, coord_type: str = 'float') -> None:
        """
        Initialize validator.

        Args:
            coord_type: 'float' for RA/Dec, 'int' for tile coordinates
        """
        super().__init__()
        self.coord_type = coord_type

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.failure('Coordinates are required')

        parts = value.replace(',', ' ').split()

        if len(parts) != 2:
            return self.failure('Must be two values (e.g., "123.45 67.89")')

        try:
            if self.coord_type == 'int':
                int(parts[0])
                int(parts[1])
            else:
                float(parts[0])
                float(parts[1])
        except ValueError:
            type_name = 'integers' if self.coord_type == 'int' else 'numbers'
            return self.failure(f'Both values must be {type_name}')

        return self.success()
