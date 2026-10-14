"""Provider-independent errors and server retry hints."""

import math
import re
import time
from email.utils import parsedate_to_datetime


class RetryableError(Exception):
    """A transient failure; retrying and waiting belong to the caller."""

    def __init__(
        self,
        message: str,
        *,
        rate_limited: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.rate_limited = rate_limited
        self.retry_after = retry_after


class ProviderResponseError(ValueError):
    """An unusable provider response with JSON-serializable failure diagnostics."""

    def __init__(self, message: str, *, details: dict):
        super().__init__(message)
        self.details = details


class ContextLimitError(Exception):
    """The passage must be shortened before retrying."""


def retry_hint(headers, message: str) -> float | None:
    hints = []
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw) * scale
        except (TypeError, ValueError):
            if name != "retry-after":
                continue
            try:
                value = parsedate_to_datetime(raw).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(value) and value >= 0:
            hints.append(value)
    match = re.search(r"try again in ([\d.]+)(ms|s)", message, re.IGNORECASE)
    if match:
        try:
            value = float(match[1]) * (0.001 if match[2].lower() == "ms" else 1)
            if math.isfinite(value):
                hints.append(value)
        except ValueError:
            pass
    return max(hints) if hints else None
