"""PHI containment — the hook must refuse patient-derived formats, and must NOT
refuse legitimate source files. Both directions are asserted: a test that only
checks refusal passes even if the hook refuses everything.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "hooks" / "pre-commit"
GITIGNORE = REPO / ".gitignore"


def _throwaway_repo(tmp_path: Path) -> Path:
    """A real git repo carrying THIS repo's hook and ignore rules."""
    r = tmp_path / "throwaway"
    (r / "hooks").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=r, check=True)
    shutil.copy(GITIGNORE, r / ".gitignore")
    dst = r / ".git" / "hooks" / "pre-commit"
    shutil.copy(HOOK, dst)
    dst.chmod(0o755)
    return r


def _try_commit(repo: Path, rel: str, content: bytes) -> bool:
    """Stage one file and attempt a commit. True if the commit SUCCEEDED.

    ⛔ `git add` failing is NOT the same as the hook refusing. Returning False on
    add failure would make every refusal test pass for the wrong reason — a
    typo'd path would read as "the gate works". That is the same error class as
    asserting at a clamped coordinate: the measurement would describe the
    harness, not the hook. So add failure raises instead.
    """
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    add = subprocess.run(["git", "add", "-f", "--", rel], cwd=repo,
                         capture_output=True, text=True)
    if add.returncode != 0:
        raise AssertionError(
            f"git add failed for {rel}, so this test cannot measure the hook: "
            f"{add.stderr.strip()}"
        )
    done = subprocess.run(["git", "commit", "-q", "-m", f"try {rel}"], cwd=repo,
                          capture_output=True, text=True)
    subprocess.run(["git", "reset", "-q"], cwd=repo, capture_output=True)
    return done.returncode == 0


@pytest.mark.parametrize("rel", [
    "cases/x/parcels.mgz",
    "cases/x/surf.annot",
    "cases/x/conn_84x84.csv",
    "cases/x/nodes.tsv",
    "cases/x/xfm.mat",
    "cases/x/coords.json",
    "cases/x/aparc.nii.gz",
    # REGRESSION: an accented filename. `git diff --cached --name-only`
    # C-quotes it, which made every $-anchored rule miss and the hook fail
    # open. pt-BR filenames make this the common case, not an edge case.
    "cases/x/crânio.mgz",
    # PIXEL_RE had no test at all; a typo in it left all tests green.
    "cases/x/render.png",
])
def test_patient_derived_formats_are_refused(tmp_path, rel):
    repo = _throwaway_repo(tmp_path)
    assert not _try_commit(repo, rel, b"x" * 2048), (
        f"{rel} committed cleanly — the containment gate has a hole"
    )


@pytest.mark.parametrize("rel", [
    "src/tractlab/connectome.py",
    "docs/note.md",
    "cases/local-case/manifest.json",
])
def test_legitimate_source_still_commits(tmp_path, rel):
    """NON-VACUITY: if this fails, the gate refuses everything and proves nothing."""
    repo = _throwaway_repo(tmp_path)
    assert _try_commit(repo, rel, b"# ok\n"), (
        f"{rel} was refused — the gate is over-broad and would block real work"
    )


@pytest.mark.parametrize("rel", [
    "cases/x/parcels.mgz",
    "cases/x/conn_84x84.csv",
    "cases/x/coords.json",
    "cases/x/render.png",
])
def test_gitignore_layer_also_denies(tmp_path, rel):
    """Defense-in-depth: .gitignore must deny these too.

    The hook tests above use `git add -f`, which DELIBERATELY bypasses
    .gitignore in order to test the hook itself. Without this test the ignore
    layer would be entirely unverified while the plan claims two layers.
    """
    repo = _throwaway_repo(tmp_path)
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    r = subprocess.run(["git", "check-ignore", "-q", "--", rel], cwd=repo)
    assert r.returncode == 0, f"{rel} is not matched by .gitignore"


def test_quoted_path_cannot_bypass_the_size_backstop(tmp_path):
    """REGRESSION for the other half of the quoted-path bug.

    C-quoting also broke `git rev-parse ":$f"`, so `|| echo 0` made the size
    check read 0 bytes and pass. A 2 MiB accented file whose extension is on
    no list must still be refused by the size backstop alone.
    """
    repo = _throwaway_repo(tmp_path)
    assert not _try_commit(repo, "cases/x/crânio_grande.txt", b"x" * (2 * 1024 * 1024)), (
        "a 2 MiB accented file committed — the size backstop is reading a "
        "C-quoted path and failing open"
    )


def test_hook_refuses_when_it_cannot_list_staged_files(tmp_path):
    """FAIL-CLOSED: if `git diff --cached` errors, the hook must refuse.

    Regression for a fix that piped git diff through `< <(process
    substitution)`. Bash's errexit does not check the exit status of a command
    inside `<(...)`, so a failing git diff produced an empty stream, every check
    saw nothing, and the hook exited 0 — fail-open on the exact axis it exists
    to close. Measured: a corrupt `.git/index` makes `git diff --cached` exit
    128. The hook is invoked directly here because `git commit` would itself
    fail first on a corrupt index, which would prove nothing about the hook.
    """
    repo = _throwaway_repo(tmp_path)
    (repo / "a.txt").write_bytes(b"x\n")
    subprocess.run(["git", "add", "-f", "--", "a.txt"], cwd=repo, check=True)
    (repo / ".git" / "index").write_bytes(b"GARBAGE-NOT-AN-INDEX")

    r = subprocess.run(["bash", str(repo / ".git" / "hooks" / "pre-commit")],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode != 0, (
        "hook exited 0 while unable to list staged files — it fails OPEN"
    )
    assert "cannot list staged files" in r.stderr


def test_gitignore_does_not_deny_the_manifest(tmp_path):
    """The negation rule must survive: manifest.json stays trackable."""
    repo = _throwaway_repo(tmp_path)
    rel = "cases/local-case/manifest.json"
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"{}")
    r = subprocess.run(["git", "check-ignore", "-q", "--", rel], cwd=repo)
    assert r.returncode != 0, "manifest.json is ignored — the negation rule is broken"
