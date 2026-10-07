"""Offline argument-handling tests for scripts/build_connectome.py --case-root.

No MRtrix / real case needed: cx.build is monkeypatched so these exercise
only the script's own argument parsing, manifest loading, and case_id/
case_root plumbing (per CLAUDE.md: never run the build scripts on a real
case as part of this task; dry-run/argument parsing + synthetic tests only).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_connectome as bc  # noqa: E402
from tractlab import connectome as cx  # noqa: E402


def _make_case(tmp_path: Path, case_id: str = "synthetic-case", name: str = "case_dir"):
    case_dir = tmp_path / name
    case_dir.mkdir()
    # Nested inside case_dir (not case_dir itself) so this still exercises
    # "manifest.case_root differs from case_dir" while satisfying the
    # equal-or-inside containment rule in _load_case.
    canonical = case_dir / "canonical"
    canonical.mkdir()
    manifest = {"case_id": case_id, "case_root": str(canonical)}
    (case_dir / "manifest.json").write_text(json.dumps(manifest))
    return case_dir, canonical


def test_case_root_flag_selects_manifest_and_is_passed_to_build(tmp_path, monkeypatch, capsys):
    case_dir, canonical = _make_case(tmp_path)
    seen = {}

    def fake_build(case_root, corpus_key, parc_key, out_dir, radius_mm=cx.DEFAULT_RADIUS_MM, manifest_path=None):
        seen["case_root"] = case_root
        seen["out_dir"] = out_dir
        seen["manifest_path"] = manifest_path
        return cx.BuildResult(
            outcome=cx.Outcome.INVALID,
            out_dir=out_dir,
            matrix_path=None,
            assignments_path=None,
            provenance_path=None,
            provenance=None,
            wall_s=0.0,
            error="synthetic refusal — offline test",
        )

    monkeypatch.setattr(cx, "build", fake_build)
    rc = bc.main(["--case-root", str(case_dir)])
    out = capsys.readouterr().out

    assert rc == 1
    assert seen["case_root"] == str(canonical)
    assert seen["out_dir"] == cx.connectome_out_dir(str(canonical))
    # The manifest this script loaded (under case_dir, NOT canonical) is
    # passed explicitly to cx.build — it must never re-derive/guess a path.
    assert seen["manifest_path"] == str(case_dir / "manifest.json")
    assert "case_id:    synthetic-case" in out
    assert f"case_root:  {canonical}" in out
    assert f"manifest:   {case_dir / 'manifest.json'}" in out


def test_default_case_root_is_the_demo_case_dir():
    assert bc.DEFAULT_CASE_DIR == bc.REPO_ROOT / "cases" / "demo-leipzig-sub-010005"


def test_missing_case_root_arg_refuses_missing_manifest(tmp_path, capsys):
    empty_dir = tmp_path / "no-manifest"
    empty_dir.mkdir()
    rc = bc.main(["--case-root", str(empty_dir)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "REFUSE" in captured.err
    assert "no manifest.json" in captured.err


def test_manifest_without_case_root_field_refuses(tmp_path, capsys):
    case_dir = tmp_path / "case_dir"
    case_dir.mkdir()
    (case_dir / "manifest.json").write_text(json.dumps({"case_id": "x"}))
    rc = bc.main(["--case-root", str(case_dir)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "REFUSE" in captured.err
    assert "case_root" in captured.err


def test_full_without_yes_is_still_refused_with_case_root_set(tmp_path, capsys, monkeypatch):
    case_dir, _canonical = _make_case(tmp_path)

    def fail_build(*args, **kwargs):
        raise AssertionError("cx.build must not run without --yes on --full")

    monkeypatch.setattr(cx, "build", fail_build)
    rc = bc.main(["--case-root", str(case_dir), "--full"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Refusing to run the FULL 10M corpus" in captured.out


def test_case_root_manifest_indirection_is_honoured_and_printed(tmp_path, capsys, monkeypatch):
    """House convention: --case-root selects the manifest; the manifest's own
    case_root (which may live elsewhere, e.g. ~/tractlab-data) is where cx.build
    runs, and both locations are printed. A recorded case_root that does not
    exist is refused before cx.build runs."""
    case_dir = tmp_path / "case_dir"
    case_dir.mkdir()
    elsewhere = tmp_path / "data-lives-here"
    elsewhere.mkdir()
    manifest = {"case_id": "indirect", "case_root": str(elsewhere)}
    (case_dir / "manifest.json").write_text(json.dumps(manifest))
    seen = {}

    def fake_build(case_root, corpus_key, parc_key, out_dir, radius_mm=cx.DEFAULT_RADIUS_MM, manifest_path=None):
        seen["case_root"] = case_root
        seen["manifest_path"] = manifest_path
        return cx.BuildResult(
            outcome=cx.Outcome.INVALID, out_dir=out_dir, matrix_path=None,
            assignments_path=None, provenance_path=None, provenance=None,
            wall_s=0.0, error="synthetic refusal — offline test",
        )

    monkeypatch.setattr(cx, "build", fake_build)
    rc = bc.main(["--case-root", str(case_dir)])
    captured = capsys.readouterr()
    assert rc == 1
    assert seen["case_root"] == str(elsewhere)
    # The manifest lives under case_dir, NOT under elsewhere (case_root) —
    # this is the exact indirection the manifest_path plumbing exists for.
    assert seen["manifest_path"] == str(case_dir / "manifest.json")
    assert str(elsewhere) in captured.out or str(elsewhere) in captured.err
    assert "indirection honoured" in captured.err

    missing = {"case_id": "indirect", "case_root": str(tmp_path / "nope")}
    (case_dir / "manifest.json").write_text(json.dumps(missing))

    def fail_build(*args, **kwargs):
        raise AssertionError("cx.build must never run against a missing case_root")

    monkeypatch.setattr(cx, "build", fail_build)
    rc = bc.main(["--case-root", str(case_dir)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "REFUSE" in captured.err and "does not exist" in captured.err


def test_selected_and_execution_manifest_identity_match(tmp_path, monkeypatch):
    """With two distinct, self-consistent case directories, selecting one
    via --case-root must make cx.build receive THAT one's own case_root —
    never the other case's, even though both are loaded through the same
    code path in the same test process."""
    case_a, canonical_a = _make_case(tmp_path, case_id="case-a", name="case_a")
    case_b, canonical_b = _make_case(tmp_path, case_id="case-b", name="case_b")
    assert canonical_a != canonical_b

    seen = {}

    def fake_build(case_root, corpus_key, parc_key, out_dir, radius_mm=cx.DEFAULT_RADIUS_MM, manifest_path=None):
        seen["case_root"] = case_root
        seen["manifest_path"] = manifest_path
        return cx.BuildResult(
            outcome=cx.Outcome.INVALID, out_dir=out_dir, matrix_path=None,
            assignments_path=None, provenance_path=None, provenance=None,
            wall_s=0.0, error="synthetic refusal — offline test",
        )

    monkeypatch.setattr(cx, "build", fake_build)

    rc = bc.main(["--case-root", str(case_b)])
    assert rc == 1
    assert seen["case_root"] == str(canonical_b)
    assert seen["case_root"] != str(canonical_a)
    assert seen["manifest_path"] == str(case_b / "manifest.json")

    seen.clear()
    rc = bc.main(["--case-root", str(case_a)])
    assert rc == 1
    assert seen["case_root"] == str(canonical_a)
    assert seen["case_root"] != str(canonical_b)
    assert seen["manifest_path"] == str(case_a / "manifest.json")
    assert seen["manifest_path"] != str(case_b / "manifest.json")
