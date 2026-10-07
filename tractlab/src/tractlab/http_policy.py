"""Loopback / browser-origin HTTP request policy (SEC-01/02).

These checks defend the *browser-origin* boundary: a malicious web page
tricking a victim's browser into issuing a same-machine ``fetch()`` against
this loopback server. They cannot and do not defend against a local process
that already has direct socket access to 127.0.0.1 — that access is outside
this boundary by design (RESULT-API-CONTRACT.md), and non-browser callers
(curl, scripts, ``mrview`` companions) legitimately send no ``Origin`` /
``Sec-Fetch-*`` headers at all.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


class PolicyRejected(ValueError):
    """A request violates the loopback/same-origin/body policy.

    ``code`` names the typed refusal so the HTTP layer can pick a stable
    status/error code without parsing the message text.
    """

    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.code = code


def _split_host_port(value: str) -> tuple[str, str | None] | None:
    value = value.strip()
    if not value or "://" in value:
        return None
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return None
        host = value[: end + 1]
        rest = value[end + 1:]
        if not rest:
            return host, None
        if not rest.startswith(":"):
            return None
        return host, rest[1:]
    if value.count(":") > 1:
        return None
    if ":" in value:
        host, port = value.split(":", 1)
        return host, port
    return value, None


def check_host(host_header: str | None, *, expected_port: int) -> None:
    """The Host header must literally name the loopback listener we bound."""
    if not host_header:
        raise PolicyRejected("missing Host header", code="bad_host")
    parsed = _split_host_port(host_header)
    if parsed is None:
        raise PolicyRejected(f"malformed Host header {host_header!r}", code="bad_host")
    host, port = parsed
    if host.lower() not in _LOOPBACK_HOSTS:
        raise PolicyRejected(f"Host {host!r} is not the loopback listener", code="bad_host")
    if (port if port is not None else "80") != str(expected_port):
        raise PolicyRejected(
            f"Host port {port!r} does not match the bound listener", code="bad_host",
        )


def check_origin(origin_header: str | None, *, expected_port: int, expected_host: str | None = None) -> None:
    """A *present* Origin must be this same loopback origin.

    Non-browser callers send no Origin at all — that is allowed; only a
    present-but-wrong Origin is a same-origin violation.
    """
    if not origin_header:
        return
    if origin_header.strip().lower() == "null":
        raise PolicyRejected("opaque Origin (null) is not same-origin", code="bad_origin")
    try:
        parts = urlsplit(origin_header)
        port = parts.port if parts.port is not None else 80
    except ValueError as exc:
        raise PolicyRejected("malformed Origin", code="bad_origin") from exc
    if parts.scheme != "http" or (parts.hostname or "").lower() not in (
        "127.0.0.1", "localhost", "::1",
    ):
        raise PolicyRejected(f"Origin {origin_header!r} is not the loopback origin", code="bad_origin")
    if parts.username or parts.password or parts.path or parts.query or parts.fragment:
        raise PolicyRejected("Origin must contain only scheme and authority", code="bad_origin")
    if expected_host is not None and parts.hostname.lower() != expected_host.strip("[]").lower():
        raise PolicyRejected("Origin host differs from request Host", code="bad_origin")
    if port != expected_port:
        raise PolicyRejected(
            f"Origin port {port} does not match the bound listener", code="bad_origin",
        )


def check_fetch_metadata(
    sec_fetch_site: str | None,
    *,
    sec_fetch_mode: str | None = None,
    sec_fetch_dest: str | None = None,
    method: str = "GET",
) -> None:
    """Reject a browser-declared cross-site fetch, per Fetch Metadata resource
    isolation — with one carve-out: a top-level GET/HEAD *navigation* (Chrome's
    ``Sec-Fetch-Mode: navigate``) is allowed through even when cross-site,
    because that is exactly what happens when a browser-automation tool or an
    extension opens this viewer's page URL directly. ``Sec-Fetch-Dest: object``
    or ``embed`` are excluded from the carve-out (a page embedding this origin
    in a frame/plugin is not a top-level navigation a user asked for). Anything
    else cross-site — API calls, POSTs, no-cors/cors fetches, or a navigation
    missing Sec-Fetch-Mode/-Dest — is rejected exactly as before.
    """
    if not sec_fetch_site:
        return
    if sec_fetch_site.strip().lower() != "cross-site":
        return
    mode = (sec_fetch_mode or "").strip().lower()
    dest = (sec_fetch_dest or "").strip().lower()
    meth = (method or "").strip().upper()
    if meth in ("GET", "HEAD") and mode == "navigate" and dest not in ("object", "embed"):
        return
    raise PolicyRejected("Sec-Fetch-Site: cross-site rejected", code="cross_site")


def check_content_length(raw: str | None, *, max_body: int, allow_zero: bool) -> int:
    """Parse Content-Length safely — a bad header must never crash the handler."""
    if raw is None:
        if allow_zero:
            return 0
        raise PolicyRejected("missing Content-Length", code="bad_content_length")
    if not isinstance(raw, str) or not re.fullmatch(r"[0-9]+", raw):
        raise PolicyRejected("Content-Length must be an integer", code="bad_content_length")
    try:
        n = int(raw)
    except ValueError as exc:
        raise PolicyRejected("invalid Content-Length", code="bad_content_length") from exc
    if n < 0 or n > max_body or (n == 0 and not allow_zero):
        raise PolicyRejected("bad or oversized body", code="bad_content_length")
    return n


def check_json_content_type(content_type: str | None) -> None:
    """A POST body present on the wire must be declared JSON."""
    if not content_type:
        raise PolicyRejected(
            "POST with a body requires Content-Type: application/json",
            code="bad_content_type",
        )
    media = content_type.split(";", 1)[0].strip().lower()
    if media != "application/json":
        raise PolicyRejected(
            f"unsupported Content-Type {content_type!r}", code="bad_content_type",
        )


def enforce_browser_origin_policy(
    *, host_header: str | None, origin_header: str | None,
    sec_fetch_site: str | None, expected_port: int,
    sec_fetch_mode: str | None = None, sec_fetch_dest: str | None = None,
    method: str = "GET",
) -> None:
    """The three same-origin checks every request (GET and POST) must pass."""
    check_host(host_header, expected_port=expected_port)
    host, _ = _split_host_port(host_header)
    check_origin(origin_header, expected_port=expected_port, expected_host=host)
    check_fetch_metadata(
        sec_fetch_site,
        sec_fetch_mode=sec_fetch_mode,
        sec_fetch_dest=sec_fetch_dest,
        method=method,
    )
