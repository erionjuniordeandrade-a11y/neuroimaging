from __future__ import annotations

import json
import os

from capsule import dwi


def _stage(tmp_path, model):
    work = tmp_path / "work"
    directory = work / "nifti"
    output = directory / "5tt.mif"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"5tt")
    params = {"synthstrip": {"model": dwi._file_signature(model)}}
    return work, dwi.StageSpec("act", directory, params, (output,), (output,))


def _model(tmp_path, content=b"weights-v1"):
    model = tmp_path / "synthstrip.1.pt"
    model.write_bytes(content)
    return model


def _rewrite_later(path, content):
    before = path.stat().st_mtime_ns
    path.write_bytes(content)
    os.utime(path, ns=(before + 10**9, before + 10**9))


def test_rewritten_identical_file_keeps_the_stage_current(tmp_path):
    model = _model(tmp_path)
    work, spec = _stage(tmp_path, model)
    dwi._write_stage_marker(spec, profile="fast", work_dir=work, tool_versions={}, started_utc="t")

    _rewrite_later(model, b"weights-v1")
    _, again = _stage(tmp_path, model)

    assert dwi._stage_is_current(again, "fast", work)


def test_changed_content_with_the_same_size_reruns_the_stage(tmp_path):
    model = _model(tmp_path)
    work, spec = _stage(tmp_path, model)
    dwi._write_stage_marker(spec, profile="fast", work_dir=work, tool_versions={}, started_utc="t")

    _rewrite_later(model, b"weights-v2")
    _, again = _stage(tmp_path, model)

    assert not dwi._stage_is_current(again, "fast", work)


def _legacy_marker(tmp_path):
    model = _model(tmp_path)
    work, spec = _stage(tmp_path, model)
    dwi._write_stage_marker(spec, profile="fast", work_dir=work, tool_versions={}, started_utc="t")
    marker = spec.directory / ".stage-act.json"
    record = json.loads(marker.read_text())
    stat = model.stat()
    record["params"]["synthstrip"]["model"] = {
        "path": str(model.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    marker.write_text(json.dumps(record))
    return model, work


def test_legacy_mtime_marker_still_resumes_when_the_file_is_untouched(tmp_path):
    model, work = _legacy_marker(tmp_path)
    _, spec = _stage(tmp_path, model)

    assert dwi._stage_is_current(spec, "fast", work)


def test_legacy_mtime_marker_reruns_when_the_file_time_changed(tmp_path):
    model, work = _legacy_marker(tmp_path)
    _rewrite_later(model, b"weights-v1")
    _, spec = _stage(tmp_path, model)

    assert not dwi._stage_is_current(spec, "fast", work)
