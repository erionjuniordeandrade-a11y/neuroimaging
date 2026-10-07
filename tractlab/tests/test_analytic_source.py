"""Explicit analytic-source selection for margin and export."""
from __future__ import annotations

import pytest

from tractlab.analytic_source import (
    AmbiguousAnalyticSource,
    AnalyticSourceRegistry,
    MissingAnalyticSource,
    UnknownBankId,
    UnknownResultId,
)


def test_named_bank_always_loads_its_original_analytic_lines():
    registry = AnalyticSourceRegistry()
    registry.commit_result(["unrelated-live"], kind="live")
    original_bank_lines = ["bank-a-1", "bank-a-2"]
    requested = []

    source = registry.resolve(
        {"bankId": "bank-a"},
        load_bank=lambda bank_id: requested.append(bank_id) or original_bank_lines,
    )

    assert requested == ["bank-a"]
    assert source.kind == "bank"
    assert source.bank_id == "bank-a"
    assert source.result_id is None
    assert source.lines == tuple(original_bank_lines)


def test_unknown_bank_never_falls_back_to_a_committed_result():
    registry = AnalyticSourceRegistry()
    current = registry.commit_result(["latest-live"], kind="live")

    with pytest.raises(UnknownBankId, match="unknown bankId"):
        registry.resolve({"bankId": "does-not-exist"}, load_bank=lambda _bid: None)

    assert registry.resolve({"resultId": current.result_id}).lines == ("latest-live",)


def test_each_committed_result_requires_its_own_opaque_identity():
    registry = AnalyticSourceRegistry()
    first = registry.commit_result(["first"], kind="live")

    assert first.result_id
    assert first.response_headers() == {
        "resultId": first.result_id,
        "analyticSourceKind": "live",
    }
    assert registry.resolve({"resultId": first.result_id}).lines == ("first",)

    second = registry.commit_result(["second"], kind="filter")

    assert second.result_id and first.result_id != second.result_id
    with pytest.raises(UnknownResultId, match="unknown resultId"):
        registry.resolve({"resultId": first.result_id})
    assert registry.resolve({"resultId": second.result_id}).lines == ("second",)
    assert registry.resolve({"resultId": second.result_id}).kind == "filter"


def test_missing_ambiguous_and_unknown_result_ids_fail_loud():
    registry = AnalyticSourceRegistry()
    result = registry.commit_result(["live"], kind="recovery")

    with pytest.raises(MissingAnalyticSource, match="resultId or bankId"):
        registry.resolve({})
    with pytest.raises(AmbiguousAnalyticSource, match="exactly one"):
        registry.resolve({"resultId": result.result_id, "bankId": "bank-a"})
    with pytest.raises(UnknownResultId, match="unknown resultId"):
        registry.resolve({"resultId": "result-not-issued"})
