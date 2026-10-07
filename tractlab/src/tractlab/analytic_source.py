"""Explicit analytic-source selection for margin and export follow-ups.

The registry retains only the latest live/filter/recovery result. A client must
echo its opaque ``resultId`` to use that result; a later commit makes an older
identifier unavailable rather than redirecting it to newer streamlines.
"""
from __future__ import annotations

import secrets
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


RESULT_KINDS = frozenset(("live", "filter", "recovery"))


class AnalyticSourceError(ValueError):
    """Base class for explicit source-selection failures."""


class MissingAnalyticSource(AnalyticSourceError):
    """Neither a resultId nor a bankId was supplied."""


class AmbiguousAnalyticSource(AnalyticSourceError):
    """Both a resultId and a bankId were supplied."""


class UnknownResultId(AnalyticSourceError):
    """The volatile result was never committed or has been superseded."""


class UnknownBankId(AnalyticSourceError):
    """The named bank does not resolve to its original analytic population."""


@dataclass(frozen=True)
class AnalyticSource:
    """Immutable selection passed to a margin or export operation."""

    lines: tuple[object, ...]
    kind: str
    result_id: str | None = None
    bank_id: str | None = None

    def response_headers(self) -> dict[str, str]:
        """Headers to echo for a newly committed volatile result."""
        if self.result_id is None:
            return {}
        return {
            "resultId": self.result_id,
            "analyticSourceKind": self.kind,
        }


class AnalyticSourceRegistry:
    """Keep one volatile analytic result and resolve explicit selections only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: AnalyticSource | None = None
        self._sequence = 0

    def commit_result(self, lines: Iterable[object], *, kind: str) -> AnalyticSource:
        """Commit live/filter/recovery lines and issue an opaque resultId."""
        if kind not in RESULT_KINDS:
            raise ValueError(f"result kind must be one of {sorted(RESULT_KINDS)}")
        snapshot = tuple(lines)
        with self._lock:
            self._sequence += 1
            result_id = f"result-{self._sequence}-{secrets.token_urlsafe(18)}"
            source = AnalyticSource(
                lines=snapshot,
                kind=kind,
                result_id=result_id,
            )
            self._latest = source
        return source

    def resolve(
        self,
        request: Mapping[str, object],
        *,
        load_bank: Callable[[str], Iterable[object] | None] | None = None,
    ) -> AnalyticSource:
        """Resolve exactly one explicit resultId or bankId.

        ``load_bank`` must return the requested bank's original analytic lines,
        or ``None`` when its id is unknown. It is never replaced by a volatile
        result.
        """
        result_id = _request_id(request, "resultId")
        bank_id = _request_id(request, "bankId")
        if result_id is None and bank_id is None:
            raise MissingAnalyticSource("request requires resultId or bankId")
        if result_id is not None and bank_id is not None:
            raise AmbiguousAnalyticSource("request must select exactly one source")

        if bank_id is not None:
            if load_bank is None:
                raise ValueError("named bank selection requires load_bank")
            lines = load_bank(bank_id)
            if lines is None:
                raise UnknownBankId(f"unknown bankId: {bank_id!r}")
            return AnalyticSource(
                lines=tuple(lines),
                kind="bank",
                bank_id=bank_id,
            )

        with self._lock:
            source = self._latest
        if source is None or source.result_id != result_id:
            raise UnknownResultId("unknown resultId")
        return source


def _request_id(request: Mapping[str, object], key: str) -> str | None:
    raw = request.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise AnalyticSourceError(f"{key} must be a non-empty string")
    return raw
