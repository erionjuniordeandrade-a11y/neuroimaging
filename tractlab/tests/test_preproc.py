from pathlib import Path
import json
from tractlab import preproc


def _case(tmp_path, with_fmaps=True):
    root = tmp_path / "case"; (root).mkdir()
    raw = {"dwi": {"path": "raw/dwi.nii.gz"}}
    for p in ["raw/dwi.nii.gz", "raw/dwi.bvec", "raw/dwi.bval", "raw/dwi.json"]:
        f = root / p; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b"x")
    (root / "raw/dwi.json").write_text(json.dumps(
        {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.04914}))
    if with_fmaps:
        for k, p in [("fmap_dwi_ap", "raw/ap.nii.gz"), ("fmap_dwi_pa", "raw/pa.nii.gz")]:
            (root / p).write_bytes(b"x"); raw[k] = {"path": p}
    (root / "manifest.json").write_text(json.dumps({"case_id": "t", "raw": raw}))
    return root


def test_plan_builds_rpe_pair_argv_with_json_readout(tmp_path):
    plan = preproc.plan_preproc(_case(tmp_path))
    flat = sum(plan.commands, [])
    assert "dwifslpreproc" in flat and "-rpe_pair" in flat
    assert "-readout_time" in flat and "0.04914" in flat
    assert "-pe_dir" in flat and "j-" in flat
    # argv purity: no token may contain a space, EXCEPT the single token
    # immediately following "-eddy_options" — MRtrix requires a literal
    # leading space inside that one argv value.
    for cmd in plan.commands:
        for i, tok in enumerate(cmd):
            if i > 0 and cmd[i - 1] == "-eddy_options":
                assert tok == " --slm=linear --repol --data_is_shelled"
                continue
            assert " " not in tok


def test_plan_wires_se_pair_via_mrcat_and_se_epi(tmp_path):
    plan = preproc.plan_preproc(_case(tmp_path))
    mrcat_cmds = [cmd for cmd in plan.commands if cmd[0] == "mrcat"]
    assert len(mrcat_cmds) == 1
    mrcat_cmd = mrcat_cmds[0]
    assert "-axis" in mrcat_cmd and "3" in mrcat_cmd

    dwifslpreproc_cmd = next(cmd for cmd in plan.commands if cmd[0] == "dwifslpreproc")
    assert "-se_epi" in dwifslpreproc_cmd
    se_epi_path = dwifslpreproc_cmd[dwifslpreproc_cmd.index("-se_epi") + 1]
    assert se_epi_path == mrcat_cmd[3]  # mrcat's own output path (before -axis)
    assert "-align_seepi" in dwifslpreproc_cmd
    assert "-eddyqc_all" in dwifslpreproc_cmd


def test_plan_without_fieldmaps_is_typed_honest_null(tmp_path):
    out = preproc.plan_preproc(_case(tmp_path, with_fmaps=False))
    assert isinstance(out, preproc.NoReversePE)
    assert "fmap" in out.reason


def test_run_stops_on_first_failure_and_names_the_command(tmp_path):
    plan = preproc.plan_preproc(_case(tmp_path))
    calls = []
    def fake_run(argv, **kw):
        calls.append(argv)
        class R: returncode = 1 if len(calls) == 2 else 0
        return R()
    res = preproc.run_preproc(plan, runner=fake_run)
    assert res.ok is False and res.failed_cmd[0] == plan.commands[1][0]
    assert len(calls) == 2  # nothing ran after the failure
