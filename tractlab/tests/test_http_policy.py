"""SEC-01/02 — loopback/same-origin HTTP request policy (tractlab.http_policy).

Synthetic unit tests against the policy functions directly; no server socket
needed here (see test_audit_server_contract.py for the wired-up HTTP path).
"""
from __future__ import annotations

import pytest

from tractlab.http_policy import (
    PolicyRejected,
    check_content_length,
    check_fetch_metadata,
    check_host,
    check_json_content_type,
    check_origin,
    enforce_browser_origin_policy,
)

PORT = 8770


# ── Host ──────────────────────────────────────────────────────────────────

def test_host_loopback_variants_pass():
    check_host("127.0.0.1", expected_port=80)
    check_host("127.0.0.1:8770", expected_port=PORT)
    check_host("localhost:8770", expected_port=PORT)
    check_host("[::1]:8770", expected_port=PORT)


def test_host_missing_or_empty_rejects():
    with pytest.raises(PolicyRejected) as e:
        check_host(None, expected_port=PORT)
    assert e.value.code == "bad_host"
    with pytest.raises(PolicyRejected):
        check_host("", expected_port=PORT)


def test_host_wrong_hostname_rejects():
    with pytest.raises(PolicyRejected) as e:
        check_host("evil.example.com", expected_port=PORT)
    assert e.value.code == "bad_host"


def test_host_wrong_port_rejects():
    with pytest.raises(PolicyRejected):
        check_host("127.0.0.1", expected_port=PORT)
    with pytest.raises(PolicyRejected) as e:
        check_host("127.0.0.1:9999", expected_port=PORT)
    assert e.value.code == "bad_host"


def test_host_malformed_value_rejects():
    with pytest.raises(PolicyRejected):
        check_host("http://127.0.0.1:8770", expected_port=PORT)
    with pytest.raises(PolicyRejected):
        check_host("127.0.0.1:8770:extra", expected_port=PORT)


# ── Origin ────────────────────────────────────────────────────────────────

def test_origin_absent_is_allowed_non_browser_caller():
    check_origin(None, expected_port=PORT)
    check_origin("", expected_port=PORT)


def test_origin_same_loopback_passes():
    check_origin("http://127.0.0.1:8770", expected_port=PORT)
    check_origin("http://localhost:8770", expected_port=PORT)


def test_origin_cross_site_rejects():
    with pytest.raises(PolicyRejected) as e:
        check_origin("http://evil.example.com", expected_port=PORT)
    assert e.value.code == "bad_origin"


def test_origin_wrong_port_rejects():
    with pytest.raises(PolicyRejected):
        check_origin("http://127.0.0.1:9999", expected_port=PORT)


def test_origin_https_scheme_rejects():
    with pytest.raises(PolicyRejected):
        check_origin("https://127.0.0.1:8770", expected_port=PORT)


def test_origin_opaque_null_rejects():
    with pytest.raises(PolicyRejected) as e:
        check_origin("null", expected_port=PORT)
    assert e.value.code == "bad_origin"


# ── Fetch Metadata ───────────────────────────────────────────────────────

def test_fetch_metadata_cross_site_rejects():
    with pytest.raises(PolicyRejected) as e:
        check_fetch_metadata("cross-site")
    assert e.value.code == "cross_site"


@pytest.mark.parametrize("value", [None, "", "same-origin", "same-site", "none"])
def test_fetch_metadata_non_cross_site_allowed(value):
    check_fetch_metadata(value)  # must not raise


def test_fetch_metadata_cross_site_get_navigate_document_allowed():
    """Chrome sends exactly these three headers for a top-level navigation
    triggered by an extension or another site (e.g. browser-automation tools
    opening the viewer URL). Fetch Metadata resource isolation allows this."""
    check_fetch_metadata(
        "cross-site", sec_fetch_mode="navigate", sec_fetch_dest="document", method="GET",
    )
    check_fetch_metadata(
        "cross-site", sec_fetch_mode="navigate", sec_fetch_dest="document", method="HEAD",
    )


def test_fetch_metadata_cross_site_navigate_object_or_embed_rejects():
    with pytest.raises(PolicyRejected) as e:
        check_fetch_metadata(
            "cross-site", sec_fetch_mode="navigate", sec_fetch_dest="object", method="GET",
        )
    assert e.value.code == "cross_site"
    with pytest.raises(PolicyRejected) as e:
        check_fetch_metadata(
            "cross-site", sec_fetch_mode="navigate", sec_fetch_dest="embed", method="GET",
        )
    assert e.value.code == "cross_site"


def test_fetch_metadata_cross_site_post_navigate_rejects():
    """A form-POST cross-site navigation must still be rejected (CSRF)."""
    with pytest.raises(PolicyRejected) as e:
        check_fetch_metadata(
            "cross-site", sec_fetch_mode="navigate", sec_fetch_dest="document", method="POST",
        )
    assert e.value.code == "cross_site"


@pytest.mark.parametrize("mode", ["no-cors", "cors", "same-origin", ""])
def test_fetch_metadata_cross_site_non_navigate_mode_rejects(mode):
    with pytest.raises(PolicyRejected) as e:
        check_fetch_metadata(
            "cross-site", sec_fetch_mode=mode, sec_fetch_dest="document", method="GET",
        )
    assert e.value.code == "cross_site"


def test_fetch_metadata_cross_site_missing_mode_or_dest_rejects():
    """Unchanged behaviour: a cross-site request with no Sec-Fetch-Mode/-Dest
    at all (older browsers, or non-navigation cross-site fetches) is rejected."""
    with pytest.raises(PolicyRejected) as e:
        check_fetch_metadata("cross-site")
    assert e.value.code == "cross_site"


# ── Content-Length ───────────────────────────────────────────────────────

def test_content_length_missing_without_allow_zero_rejects():
    with pytest.raises(PolicyRejected) as e:
        check_content_length(None, max_body=1000, allow_zero=False)
    assert e.value.code == "bad_content_length"


def test_content_length_missing_with_allow_zero_is_zero():
    assert check_content_length(None, max_body=1000, allow_zero=True) == 0


def test_content_length_non_integer_rejects_instead_of_crashing():
    """The historical bug: int(header) with a non-numeric header raised an
    uncaught ValueError from inside the handler (a closed socket, not a
    typed response). This must be a typed PolicyRejected instead."""
    with pytest.raises(PolicyRejected) as e:
        check_content_length("not-a-number", max_body=1000, allow_zero=False)
    assert e.value.code == "bad_content_length"


def test_content_length_negative_rejects():
    with pytest.raises(PolicyRejected):
        check_content_length("-5", max_body=1000, allow_zero=False)


def test_content_length_over_max_rejects():
    with pytest.raises(PolicyRejected):
        check_content_length("2000", max_body=1000, allow_zero=False)


def test_content_length_zero_without_allow_zero_rejects():
    with pytest.raises(PolicyRejected):
        check_content_length("0", max_body=1000, allow_zero=False)


def test_content_length_valid_passes():
    assert check_content_length("42", max_body=1000, allow_zero=False) == 42


# ── JSON content-type ────────────────────────────────────────────────────

def test_json_content_type_accepts_with_charset_suffix():
    check_json_content_type("application/json; charset=utf-8")
    check_json_content_type("application/json")


def test_json_content_type_rejects_missing_or_wrong():
    with pytest.raises(PolicyRejected) as e:
        check_json_content_type(None)
    assert e.value.code == "bad_content_type"
    with pytest.raises(PolicyRejected):
        check_json_content_type("text/plain")
    with pytest.raises(PolicyRejected):
        check_json_content_type("multipart/form-data; boundary=x")


# ── Combined enforcement ─────────────────────────────────────────────────

def test_enforce_browser_origin_policy_all_clean_passes():
    enforce_browser_origin_policy(
        host_header="127.0.0.1:8770",
        origin_header="http://127.0.0.1:8770",
        sec_fetch_site="same-origin",
        expected_port=8770,
    )


def test_enforce_browser_origin_policy_curl_like_caller_passes():
    """A non-browser caller sends Host but no Origin/Sec-Fetch-Site at all."""
    enforce_browser_origin_policy(
        host_header="127.0.0.1:8770",
        origin_header=None,
        sec_fetch_site=None,
        expected_port=8770,
    )


def test_enforce_browser_origin_policy_cross_site_fetch_rejects():
    with pytest.raises(PolicyRejected) as e:
        enforce_browser_origin_policy(
            host_header="127.0.0.1:8770",
            origin_header="http://evil.example.com",
            sec_fetch_site="cross-site",
            expected_port=8770,
        )
    assert e.value.code in ("bad_origin", "cross_site")


def test_enforce_browser_origin_policy_cross_site_get_navigate_passes():
    """Chrome's real header set for a top-level navigation opened by an
    extension/other site: no Origin header, cross-site, navigate/document."""
    enforce_browser_origin_policy(
        host_header="127.0.0.1:8770",
        origin_header=None,
        sec_fetch_site="cross-site",
        sec_fetch_mode="navigate",
        sec_fetch_dest="document",
        method="GET",
        expected_port=8770,
    )


def test_enforce_browser_origin_policy_cross_site_post_navigate_rejects():
    """POST form-navigation cross-site must still be rejected (CSRF)."""
    with pytest.raises(PolicyRejected) as e:
        enforce_browser_origin_policy(
            host_header="127.0.0.1:8770",
            origin_header=None,
            sec_fetch_site="cross-site",
            sec_fetch_mode="navigate",
            sec_fetch_dest="document",
            method="POST",
            expected_port=8770,
        )
    assert e.value.code == "cross_site"
