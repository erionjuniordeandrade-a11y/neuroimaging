"""Content identity for one local TractLab server boot."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import derivation


VIEWER_PROTOCOL = 1
_CLIENT_SUFFIXES = frozenset((".css", ".html", ".js", ".json", ".svg"))
_BOOT_META = re.compile(
    br'<meta\b[^>]*\bname\s*=\s*["\']tractlab-(?:build|protocol)["\'][^>]*>',
    re.IGNORECASE,
)
_HEAD_CLOSE = re.compile(br"</head\s*>", re.IGNORECASE)


class FileHashCache:
    """Content hashes keyed by a file's stable local change stamp."""

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[tuple[int, int, int, int], str]] = {}

    def hash(self, path: Path) -> str:
        before = path.stat()
        stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        after_stamp = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if after_stamp != stamp:
            raise OSError("source changed while its identity was being read")
        value = digest.hexdigest()
        self._cache[path] = (stamp, value)
        return value


class CaseSourceChanged(RuntimeError):
    """A served case source no longer matches the content captured at boot."""


class CaseSourceRegistry:
    """Bind non-bank case inputs and evidence to one service boot."""

    def __init__(self, entries: list[tuple[str, Path]], file_hashes: FileHashCache):
        self._entries = tuple(sorted(entries, key=lambda entry: entry[0]))
        self._file_hashes = file_hashes
        self.hash = self._hash_entries()

    def validate(self) -> str:
        current = self.current_hash()
        if current != self.hash:
            raise CaseSourceChanged("case source changed after server boot")
        return self.hash

    def current_hash(self) -> str:
        return self._hash_entries()

    def _hash_entries(self) -> str:
        digest = hashlib.sha256()
        for label, path in self._entries:
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
            try:
                value = self._file_hashes.hash(path)
            except OSError:
                value = "absent"
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


def _case_source_entries(manifest_path: str | Path, manifest: object) -> list[tuple[str, Path]]:
    """Collect safe labels for files that affect served case anatomy/evidence."""
    manifest_file = Path(manifest_path).resolve()
    entries = [("manifest", manifest_file)]
    if not isinstance(manifest, dict):
        return entries
    case_root = manifest.get("case_root")
    if not isinstance(case_root, str):
        return entries
    try:
        inputs = derivation.active_inputs(manifest)
    except ValueError:
        return entries
    root = Path(case_root).resolve()

    def add(label: str, value: object) -> None:
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            return
        candidate = (root / value).resolve()
        if candidate != root and str(candidate).startswith(str(root) + "/"):
            entries.append((label, candidate))

    for key, meta in sorted(inputs.items()):
        if not isinstance(key, str) or not isinstance(meta, dict):
            continue
        if key.startswith("bank_"):
            # Named tract sources retain their own X-bankSourceHash contract.
            entries.append((
                f"fidelity/{key}",
                derivation.artifact_path(manifest, "fidelity", f"{key}.fidelity.npz"),
            ))
            continue
        add(f"input/{key}", meta.get("path"))
        add(f"input/{key}/lut", meta.get("lut_path"))
        if key == "parc_schaefer200_yeo7" and "lut_path" not in meta:
            add(
                f"input/{key}/lut",
                "normative/Schaefer2018_200Parcels_7Networks_order.txt",
            )
        for field in ("and", "not"):
            values = meta.get(field)
            if isinstance(values, (list, tuple)):
                for index, value in enumerate(values):
                    add(f"input/{key}/{field}/{index}", value)
    evidence = manifest.get("acquisition_evidence")
    if isinstance(evidence, dict):
        for criterion, meta in sorted(evidence.items()):
            if not isinstance(criterion, str) or not isinstance(meta, dict):
                continue
            for field in ("path", "b0", "mask"):
                add(f"preflight/{criterion}/{field}", meta.get(field))
            declared = meta.get("path")
            if isinstance(declared, str) and Path(declared).suffix == ".bvec":
                add(
                    f"preflight/{criterion}/bval-sibling",
                    str(Path(declared).with_suffix(".bval")),
                )
    entries.append(("preflight/record", root / "preflight.json"))
    entries.append(("fidelity/operating-point", derivation.operating_point_path(manifest)))
    entries.append(("fidelity/sweep", derivation.fidelity_sweep_path(manifest)))
    return entries


class RuntimeIdentity:
    """Fingerprint served client and Python source at boot, without Git state."""

    def __init__(
        self,
        viewer_dir: str | Path,
        source_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        file_hashes: FileHashCache | None = None,
    ):
        self._viewer_dir = Path(viewer_dir).resolve()
        self._source_dir = Path(source_dir or Path(__file__).parent).resolve()
        self._file_hashes = file_hashes or FileHashCache()
        self.case_id, self.manifest_id, manifest = self._manifest_identity(manifest_path)
        entries = [] if manifest_path is None else _case_source_entries(manifest_path, manifest)
        self._case_sources = CaseSourceRegistry(entries, self._file_hashes)
        self.case_source_hash = self._case_sources.hash
        self.build_id = self._source_hash()
        self.boot_id = uuid.uuid4().hex
        self.booted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    def public_metadata(self) -> dict[str, object]:
        return {
            "viewerProtocol": VIEWER_PROTOCOL,
            "expectedViewerProtocol": VIEWER_PROTOCOL,
            "buildId": self.build_id,
            "expectedClientBuildId": self.build_id,
            "manifestId": self.manifest_id,
            "caseSourceHash": self.case_source_hash,
            "bootId": self.boot_id,
            "bootedAt": self.booted_at,
            "runtimeStatus": "ok",
        }

    @staticmethod
    def _manifest_identity(
        manifest_path: str | Path | None,
    ) -> tuple[str | None, str | None, object | None]:
        if manifest_path is None:
            return None, None, None
        raw = Path(manifest_path).read_bytes()
        parsed = json.loads(raw)
        case_id = parsed.get("case_id") if isinstance(parsed, dict) else None
        return (
            case_id if isinstance(case_id, str) else None,
            hashlib.sha256(raw).hexdigest(),
            parsed,
        )

    def restart_payload(self, port: int) -> dict[str, object] | None:
        """Return a safe restart error if any served source changed after boot."""
        try:
            current_build = self._source_hash()
        except OSError:
            current_build = None
        try:
            current_case = self._case_sources.validate()
        except CaseSourceChanged:
            current_case = self._case_sources.current_hash()
        if current_build == self.build_id and current_case == self.case_source_hash:
            return None
        return {
            "error": (
                "TractLab source changed while this server was running. "
                f"Restart with ./serve.sh {port}."
            ),
            "code": "restart_required",
            "restartRequired": True,
            "viewerProtocol": VIEWER_PROTOCOL,
            "bootBuildId": self.build_id,
            "currentBuildId": current_build,
            "bootCaseSourceHash": self.case_source_hash,
            "currentCaseSourceHash": current_case,
            "bootId": self.boot_id,
            "caseId": self.case_id,
            "manifestId": self.manifest_id,
            "runtimeStatus": "restart-required",
        }

    def verify_case_sources(self) -> None:
        """Refuse a boot if case files changed while startup was reading them."""
        self._case_sources.validate()

    def inject_document_meta(self, document: bytes) -> bytes:
        """Serve the boot fingerprint from memory, never from mutable HTML."""
        tags = (
            f'<meta name="tractlab-build" content="{self.build_id}">'
            f'<meta name="tractlab-protocol" content="{VIEWER_PROTOCOL}">'
        ).encode("ascii")
        clean = _BOOT_META.sub(b"", document)
        closing = _HEAD_CLOSE.search(clean)
        if closing is None:
            return tags + clean
        return clean[:closing.start()] + tags + clean[closing.start():]

    def _source_hash(self) -> str:
        """Hash bytes, not revision metadata, so local uncommitted edits count."""
        entries: list[tuple[str, Path]] = []
        entries.extend((f"python/{p.relative_to(self._source_dir).as_posix()}", p)
                       for p in self._source_dir.rglob("*.py") if p.is_file())
        entries.extend((f"viewer/{p.relative_to(self._viewer_dir).as_posix()}", p)
                       for p in self._viewer_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in _CLIENT_SUFFIXES)
        digest = hashlib.sha256()
        for name, path in sorted(entries):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._file_hashes.hash(path).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


class BankSourceChanged(RuntimeError):
    """A named bank no longer matches the content captured at server boot."""


class BankSourceRegistry:
    """Keep boot hashes and reread a source only when its file stamp changes."""

    def __init__(self, file_hashes: FileHashCache | None = None) -> None:
        self._sources: dict[str, tuple[tuple[tuple[str, Path | None], ...], str]] = {}
        self._file_hashes = file_hashes or FileHashCache()

    def capture(
        self,
        bank_id: str,
        tck_path: str | Path,
        sift2_path: str | Path | None,
    ) -> str:
        paths = (("tck", Path(tck_path)), ("sift2", None if sift2_path is None else Path(sift2_path)))
        value = self._hash_paths(paths)
        self._sources[bank_id] = (paths, value)
        return value

    def validate(self, bank_id: str) -> str:
        try:
            paths, expected = self._sources[bank_id]
            current = self._hash_paths(paths)
        except (KeyError, OSError) as exc:
            raise BankSourceChanged("named bank source is unavailable after server boot") from exc
        if current != expected:
            raise BankSourceChanged("named bank source changed after server boot")
        return expected

    def _hash_paths(self, paths: tuple[tuple[str, Path | None], ...]) -> str:
        digest = hashlib.sha256()
        for label, path in paths:
            digest.update(label.encode("ascii"))
            digest.update(b"\0")
            digest.update(b"absent" if path is None else self._file_hashes.hash(path).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
