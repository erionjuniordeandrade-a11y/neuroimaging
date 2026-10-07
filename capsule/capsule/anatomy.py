"""Automatic, unreviewed anatomy layers for the case-capsule/1 manifest."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class LabelSpec:
    source_values: tuple[int, ...]
    key: str
    name: str
    group: str
    color: str


@dataclass(frozen=True)
class TaskLabel:
    key: str
    name: str
    group: str
    color: str


SYNTHSEG_LABELS = (
    LabelSpec((4, 5), "lateral_ventricle_left", "Ventrículo lateral esquerdo", "Ventrículos", "#43D7DF"),
    LabelSpec((43, 44), "lateral_ventricle_right", "Ventrículo lateral direito", "Ventrículos", "#27B8C6"),
    LabelSpec((14,), "third_ventricle", "Terceiro ventrículo", "Ventrículos", "#54E2E8"),
    LabelSpec((15,), "fourth_ventricle", "Quarto ventrículo", "Ventrículos", "#33C3D0"),
    LabelSpec((16,), "brainstem", "Tronco encefálico", "Tronco e cerebelo", "#CF896C"),
    LabelSpec((7, 8), "cerebellum_left", "Cerebelo esquerdo", "Tronco e cerebelo", "#D79A7C"),
    LabelSpec((46, 47), "cerebellum_right", "Cerebelo direito", "Tronco e cerebelo", "#E5B18E"),
    LabelSpec((10,), "thalamus_left", "Tálamo esquerdo", "Núcleos profundos", "#B784A7"),
    LabelSpec((49,), "thalamus_right", "Tálamo direito", "Núcleos profundos", "#C89ABA"),
    LabelSpec((11,), "caudate_left", "Núcleo caudado esquerdo", "Núcleos profundos", "#9A83B7"),
    LabelSpec((50,), "caudate_right", "Núcleo caudado direito", "Núcleos profundos", "#AF98CC"),
    LabelSpec((12,), "putamen_left", "Putâmen esquerdo", "Núcleos profundos", "#C77B72"),
    LabelSpec((51,), "putamen_right", "Putâmen direito", "Núcleos profundos", "#D9968C"),
    LabelSpec((13,), "pallidum_left", "Globo pálido esquerdo", "Núcleos profundos", "#8C776D"),
    LabelSpec((52,), "pallidum_right", "Globo pálido direito", "Núcleos profundos", "#A88E81"),
    LabelSpec((17,), "hippocampus_left", "Hipocampo esquerdo", "Estruturas límbicas", "#D46FA8"),
    LabelSpec((53,), "hippocampus_right", "Hipocampo direito", "Estruturas límbicas", "#E28BB9"),
    LabelSpec((18,), "amygdala_left", "Amígdala esquerda", "Estruturas límbicas", "#E3866B"),
    LabelSpec((54,), "amygdala_right", "Amígdala direita", "Estruturas límbicas", "#F1A18A"),
)


# Reviewed Brazilian Portuguese names and lobe groups for every SynthSeg DK parcel.
# Source values and colors remain sourced from FreeSurfer's model list and color LUT.
DK_PARCEL_PTBR: dict[str, tuple[str, str]] = {
    "bankssts_left": ("Banco do sulco temporal superior esquerdo", "Lobo temporal"),
    "bankssts_right": ("Banco do sulco temporal superior direito", "Lobo temporal"),
    "caudalanteriorcingulate_left": ("Giro do cíngulo anterior caudal esquerdo", "Lobo límbico (cíngulo)"),
    "caudalanteriorcingulate_right": ("Giro do cíngulo anterior caudal direito", "Lobo límbico (cíngulo)"),
    "caudalmiddlefrontal_left": ("Giro frontal médio caudal esquerdo", "Lobo frontal"),
    "caudalmiddlefrontal_right": ("Giro frontal médio caudal direito", "Lobo frontal"),
    "cuneus_left": ("Cúneo esquerdo", "Lobo occipital"),
    "cuneus_right": ("Cúneo direito", "Lobo occipital"),
    "entorhinal_left": ("Córtex entorrinal esquerdo", "Lobo temporal"),
    "entorhinal_right": ("Córtex entorrinal direito", "Lobo temporal"),
    "fusiform_left": ("Giro fusiforme esquerdo", "Lobo temporal"),
    "fusiform_right": ("Giro fusiforme direito", "Lobo temporal"),
    "inferiorparietal_left": ("Lóbulo parietal inferior esquerdo", "Lobo parietal"),
    "inferiorparietal_right": ("Lóbulo parietal inferior direito", "Lobo parietal"),
    "inferiortemporal_left": ("Giro temporal inferior esquerdo", "Lobo temporal"),
    "inferiortemporal_right": ("Giro temporal inferior direito", "Lobo temporal"),
    "isthmuscingulate_left": ("Istmo do giro do cíngulo esquerdo", "Lobo límbico (cíngulo)"),
    "isthmuscingulate_right": ("Istmo do giro do cíngulo direito", "Lobo límbico (cíngulo)"),
    "lateraloccipital_left": ("Córtex occipital lateral esquerdo", "Lobo occipital"),
    "lateraloccipital_right": ("Córtex occipital lateral direito", "Lobo occipital"),
    "lateralorbitofrontal_left": ("Giro orbitofrontal lateral esquerdo", "Lobo frontal"),
    "lateralorbitofrontal_right": ("Giro orbitofrontal lateral direito", "Lobo frontal"),
    "lingual_left": ("Giro lingual esquerdo", "Lobo occipital"),
    "lingual_right": ("Giro lingual direito", "Lobo occipital"),
    "medialorbitofrontal_left": ("Giro orbitofrontal medial esquerdo", "Lobo frontal"),
    "medialorbitofrontal_right": ("Giro orbitofrontal medial direito", "Lobo frontal"),
    "middletemporal_left": ("Giro temporal médio esquerdo", "Lobo temporal"),
    "middletemporal_right": ("Giro temporal médio direito", "Lobo temporal"),
    "parahippocampal_left": ("Giro parahipocampal esquerdo", "Lobo temporal"),
    "parahippocampal_right": ("Giro parahipocampal direito", "Lobo temporal"),
    "paracentral_left": ("Lóbulo paracentral esquerdo", "Lobo frontal"),
    "paracentral_right": ("Lóbulo paracentral direito", "Lobo frontal"),
    "parsopercularis_left": ("Porção opercular do giro frontal inferior esquerdo", "Lobo frontal"),
    "parsopercularis_right": ("Porção opercular do giro frontal inferior direito", "Lobo frontal"),
    "parsorbitalis_left": ("Porção orbital do giro frontal inferior esquerdo", "Lobo frontal"),
    "parsorbitalis_right": ("Porção orbital do giro frontal inferior direito", "Lobo frontal"),
    "parstriangularis_left": ("Porção triangular do giro frontal inferior esquerdo", "Lobo frontal"),
    "parstriangularis_right": ("Porção triangular do giro frontal inferior direito", "Lobo frontal"),
    "pericalcarine_left": ("Córtex pericalcarino esquerdo", "Lobo occipital"),
    "pericalcarine_right": ("Córtex pericalcarino direito", "Lobo occipital"),
    "postcentral_left": ("Giro pós-central esquerdo", "Lobo parietal"),
    "postcentral_right": ("Giro pós-central direito", "Lobo parietal"),
    "posteriorcingulate_left": ("Giro do cíngulo posterior esquerdo", "Lobo límbico (cíngulo)"),
    "posteriorcingulate_right": ("Giro do cíngulo posterior direito", "Lobo límbico (cíngulo)"),
    "precentral_left": ("Giro pré-central esquerdo", "Lobo frontal"),
    "precentral_right": ("Giro pré-central direito", "Lobo frontal"),
    "precuneus_left": ("Pré-cúneo esquerdo", "Lobo parietal"),
    "precuneus_right": ("Pré-cúneo direito", "Lobo parietal"),
    "rostralanteriorcingulate_left": ("Giro do cíngulo anterior rostral esquerdo", "Lobo límbico (cíngulo)"),
    "rostralanteriorcingulate_right": ("Giro do cíngulo anterior rostral direito", "Lobo límbico (cíngulo)"),
    "rostralmiddlefrontal_left": ("Giro frontal médio rostral esquerdo", "Lobo frontal"),
    "rostralmiddlefrontal_right": ("Giro frontal médio rostral direito", "Lobo frontal"),
    "superiorfrontal_left": ("Giro frontal superior esquerdo", "Lobo frontal"),
    "superiorfrontal_right": ("Giro frontal superior direito", "Lobo frontal"),
    "superiorparietal_left": ("Lóbulo parietal superior esquerdo", "Lobo parietal"),
    "superiorparietal_right": ("Lóbulo parietal superior direito", "Lobo parietal"),
    "superiortemporal_left": ("Giro temporal superior esquerdo", "Lobo temporal"),
    "superiortemporal_right": ("Giro temporal superior direito", "Lobo temporal"),
    "supramarginal_left": ("Giro supramarginal esquerdo", "Lobo parietal"),
    "supramarginal_right": ("Giro supramarginal direito", "Lobo parietal"),
    "frontalpole_left": ("Polo frontal esquerdo", "Lobo frontal"),
    "frontalpole_right": ("Polo frontal direito", "Lobo frontal"),
    "temporalpole_left": ("Polo temporal esquerdo", "Lobo temporal"),
    "temporalpole_right": ("Polo temporal direito", "Lobo temporal"),
    "transversetemporal_left": ("Giro temporal transverso esquerdo", "Lobo temporal"),
    "transversetemporal_right": ("Giro temporal transverso direito", "Lobo temporal"),
    "insula_left": ("Ínsula esquerda", "Ínsula"),
    "insula_right": ("Ínsula direita", "Ínsula"),
}


# CT carries bone and air only (owner ruling 2026-09-28): soft tissue, nerves, vessels and ventricles come from MR.
_TASK_LABELS: dict[str, dict[str, TaskLabel]] = {
    "craniofacial_structures": {
        "mandible": TaskLabel("mandible", "Mandíbula", "Osso", "#E8DCC4"),
        "teeth_lower": TaskLabel("teeth_lower", "Dentes inferiores", "Osso", "#D7C7A9"),
        "skull": TaskLabel("skull", "Crânio", "Osso", "#F0E6D2"),
        "sinus_maxillary": TaskLabel("sinus_maxillary", "Seio maxilar", "Seios e cavidades", "#8CC6B5"),
        "sinus_frontal": TaskLabel("sinus_frontal", "Seio frontal", "Seios e cavidades", "#72B5A2"),
        "teeth_upper": TaskLabel("teeth_upper", "Dentes superiores", "Osso", "#CDBB9B"),
    },
    "headneck_bones_vessels": {
        "larynx_air": TaskLabel("larynx_air", "Via aérea laríngea", "Seios e cavidades", "#70B6AA"),
        "thyroid_cartilage": TaskLabel("thyroid_cartilage", "Cartilagem tireóidea", "Osso", "#D9CBB3"),
        "hyoid": TaskLabel("hyoid", "Osso hióide", "Osso", "#E8DCC4"),
        "cricoid_cartilage": TaskLabel("cricoid_cartilage", "Cartilagem cricóidea", "Osso", "#CBBBA2"),
        "zygomatic_arch_right": TaskLabel("zygomatic_arch_right", "Arco zigomático direito", "Osso", "#F0E6D2"),
        "zygomatic_arch_left": TaskLabel("zygomatic_arch_left", "Arco zigomático esquerdo", "Osso", "#E8DCC4"),
        "styloid_process_right": TaskLabel("styloid_process_right", "Processo estiloide direito", "Osso", "#D9CBB3"),
        "styloid_process_left": TaskLabel("styloid_process_left", "Processo estiloide esquerdo", "Osso", "#CBBBA2"),
    },
    "head_glands_cavities": {
        "nasopharynx": TaskLabel("nasopharynx", "Nasofaringe", "Seios e cavidades", "#A889C8"),
        "oropharynx": TaskLabel("oropharynx", "Orofaringe", "Seios e cavidades", "#9473B4"),
        "hypopharynx": TaskLabel("hypopharynx", "Hipofaringe", "Seios e cavidades", "#B39AD2"),
        "nasal_cavity_right": TaskLabel("nasal_cavity_right", "Cavidade nasal direita", "Seios e cavidades", "#75BDA9"),
        "nasal_cavity_left": TaskLabel("nasal_cavity_left", "Cavidade nasal esquerda", "Seios e cavidades", "#5AA38F"),
        "auditory_canal_right": TaskLabel("auditory_canal_right", "Canal auditivo direito", "Seios e cavidades", "#8FA7D8"),
        "auditory_canal_left": TaskLabel("auditory_canal_left", "Canal auditivo esquerdo", "Seios e cavidades", "#748CC1"),
        "hard_palate": TaskLabel("hard_palate", "Palato duro", "Osso", "#BE7890"),
    },
}

_TASK_CLASS_ORDER = {
    "craniofacial_structures": tuple(_TASK_LABELS["craniofacial_structures"]),
    "headneck_bones_vessels": tuple(_TASK_LABELS["headneck_bones_vessels"]),
    "head_glands_cavities": tuple(_TASK_LABELS["head_glands_cavities"]),
}

_TOTALSEG_TASKS = tuple(_TASK_CLASS_ORDER)
_FREESURFER_LICENSE = "FreeSurfer Software License Agreement (code and SynthSeg model weights; research use only)"
_TOTALSEG_LICENSE = "Apache-2.0 (TotalSegmentator code and open-task weights)"
_SYNTHSEG_NATIVE_PACKAGES = (
    "tensorflow==2.17.1", "tf-keras==2.17.0", "surfa", "numpy==1.26.4", "scipy", "nibabel",
)


def encode_label_map(source: np.ndarray, definitions: tuple[LabelSpec, ...] | list[LabelSpec],
                     spacing_mm: tuple[float, float, float] | list[float]) -> tuple[np.ndarray, list[dict]]:
    """Map source label IDs to compact uint8 values and derive volumes on the capsule grid."""
    source = np.asarray(source)
    spacing = np.asarray(spacing_mm, dtype=float)
    if source.ndim != 3:
        raise ValueError("anatomy label arrays must be 3D")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("anatomy grid spacing must contain three positive finite values")
    if len(definitions) > 255:
        raise ValueError("an anatomy item cannot contain more than 255 labels")
    claimed_sources: set[int] = set()
    output = np.zeros(source.shape, dtype=np.uint8)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    labels = []
    for value, spec in enumerate(definitions, start=1):
        overlap = claimed_sources.intersection(spec.source_values)
        if overlap:
            raise ValueError(f"source label values assigned more than once: {sorted(overlap)}")
        claimed_sources.update(spec.source_values)
        selected = np.isin(source, spec.source_values)
        output[selected] = value
        labels.append({"value": value, "key": spec.key, "name": spec.name, "group": spec.group,
                       "color": spec.color, "volume_ml": round(float(np.count_nonzero(selected)) * voxel_ml, 6)})
    return output, labels


def synthseg_dk_label_specs(home: Path | None = None) -> tuple[LabelSpec, ...]:
    """Resolve DK IDs and colors from the installed SynthSeg list and FreeSurfer LUT."""
    home = home or _freesurfer_home()
    labels_path = home / "models" / "synthseg_parcellation_labels.npy"
    lut_path = home / "FreeSurferColorLUT.txt"
    if not labels_path.is_file():
        raise FileNotFoundError(f"SynthSeg parcellation label list is missing: {labels_path}")
    if not lut_path.is_file():
        raise FileNotFoundError(f"FreeSurfer color LUT is missing: {lut_path}")
    values = np.load(labels_path, allow_pickle=False)
    if values.ndim != 1:
        raise RuntimeError(f"SynthSeg parcellation label list must be one-dimensional: {labels_path}")
    parcel_values = [int(value) for value in values if int(value) != 0]
    if len(parcel_values) != len(set(parcel_values)):
        raise RuntimeError(f"SynthSeg parcellation label list contains duplicate IDs: {labels_path}")

    requested = set(parcel_values)
    lut: dict[int, tuple[str, tuple[int, int, int]]] = {}
    for line in lut_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6 or not fields[0].isdigit():
            continue
        try:
            label_value = int(fields[0])
            rgb = tuple(int(channel) for channel in fields[2:5])
        except ValueError:
            continue
        if label_value in requested and fields[1].startswith(("ctx-lh-", "ctx-rh-")):
            if label_value in lut:
                raise RuntimeError(f"FreeSurfer color LUT contains duplicate DK ID {label_value}: {lut_path}")
            if any(channel < 0 or channel > 255 for channel in rgb):
                raise RuntimeError(f"FreeSurfer color LUT has an invalid RGB color for DK ID {label_value}")
            lut[label_value] = (fields[1], rgb)

    missing_values = sorted(requested - set(lut))
    if missing_values:
        raise RuntimeError(f"FreeSurfer color LUT lacks SynthSeg DK IDs: {missing_values}")

    specs = []
    for value in parcel_values:
        lut_name, rgb = lut[value]
        if lut_name.startswith("ctx-lh-"):
            region, side = lut_name.removeprefix("ctx-lh-"), "left"
        else:
            region, side = lut_name.removeprefix("ctx-rh-"), "right"
        key = f"{region.replace('-', '_')}_{side}"
        if key not in DK_PARCEL_PTBR:
            raise RuntimeError(f"DK parcel {lut_name} has no reviewed pt-BR name and lobe group")
        name, group = DK_PARCEL_PTBR[key]
        color = "#" + "".join(f"{channel:02X}" for channel in rgb)
        specs.append(LabelSpec((value,), key, name, group, color))

    observed_keys = {spec.key for spec in specs}
    if observed_keys != set(DK_PARCEL_PTBR):
        missing = sorted(observed_keys ^ set(DK_PARCEL_PTBR))
        raise RuntimeError(f"DK pt-BR localization table does not match the installed SynthSeg list: {missing}")
    return tuple(specs)


def resample_labels(image: sitk.Image, grid: sitk.Image, transform: sitk.Transform) -> np.ndarray:
    """Nearest-neighbor map of a tool label volume onto the capsule grid."""
    if image.GetDimension() != 3 or grid.GetDimension() != 3:
        raise ValueError("anatomy resampling requires 3D source and target images")
    values = sitk.GetArrayViewFromImage(image)
    if values.size and (values.min() < 0 or values.max() > 255):
        raise ValueError("anatomy source labels must fit in uint8")
    source = sitk.Cast(image, sitk.sitkUInt8)
    aligned = sitk.Resample(source, grid, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(aligned)


def _resample_synthseg_labels(image: sitk.Image, grid: sitk.Image, transform: sitk.Transform) -> np.ndarray:
    """Resample SynthSeg's wide label IDs without truncating DK values above uint8."""
    if image.GetDimension() != 3 or grid.GetDimension() != 3:
        raise ValueError("anatomy resampling requires 3D source and target images")
    values = sitk.GetArrayViewFromImage(image)
    if values.size and (values.min() < 0 or values.max() > np.iinfo(np.uint16).max):
        raise ValueError("SynthSeg source labels must fit in uint16")
    source = sitk.Cast(image, sitk.sitkUInt16)
    aligned = sitk.Resample(source, grid, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    return sitk.GetArrayFromImage(aligned)


def task_label_specs(task: str, task_info: dict) -> tuple[LabelSpec, ...]:
    """Resolve only explicitly supported class names against the installed task class map."""
    if task not in _TASK_CLASS_ORDER:
        raise ValueError(f"TotalSegmentator task {task!r} is not in the capsule anatomy task set")
    classes = {str(name): int(index) for index, name in task_info.get("classes", {}).items()}
    missing = sorted(set(_TASK_CLASS_ORDER[task]) - set(classes))
    if missing:
        raise RuntimeError(f"TotalSegmentator {task} class map lacks expected labels: {', '.join(missing)}")
    return tuple(LabelSpec((classes[class_name],), _TASK_LABELS[task][class_name].key,
                           _TASK_LABELS[task][class_name].name, _TASK_LABELS[task][class_name].group,
                           _TASK_LABELS[task][class_name].color)
                 for class_name in _TASK_CLASS_ORDER[task])


def totalsegmentator_info(executable: str | None = None) -> dict:
    executable = executable or shutil.which("totalseg_info")
    if not executable:
        raise FileNotFoundError("totalseg_info not found; install TotalSegmentator outside the project environment")
    result = subprocess.run([executable, "--json"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"totalseg_info --json exited {result.returncode}: {_error_text(result.stderr or result.stdout)}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("totalseg_info --json did not return valid JSON") from exc


def task_specs_from_info(info: dict) -> list[tuple[str, tuple[LabelSpec, ...]]]:
    tasks = info.get("tasks", {})
    result = []
    for task in _TOTALSEG_TASKS:
        task_info = tasks.get(task)
        if task_info is None:
            continue
        if task_info.get("modality") != "CT":
            raise RuntimeError(f"TotalSegmentator task {task} is {task_info.get('modality')}, expected CT")
        if task_info.get("license_required"):
            continue
        result.append((task, task_label_specs(task, task_info)))
    return result


def licensed_head_tasks(info: dict) -> list[str]:
    """List installed CT head tasks that require a licence the owner has not registered."""
    tasks = info.get("tasks", {})
    return [task for task in ("brain_structures", "face")
            if tasks.get(task, {}).get("modality") == "CT" and tasks[task].get("license_required")]


def clean_nifti(image: sitk.Image, destination: Path) -> None:
    """Write pixels and physical geometry only, without source metadata."""
    clean = sitk.GetImageFromArray(sitk.GetArrayFromImage(image).astype(np.float32))
    clean.CopyInformation(image)
    sitk.WriteImage(clean, str(destination), True)


def _error_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    shared_memory = re.search(
        r"RuntimeError: unable to open shared memory object <[^>]+> in read-write mode: "
        r"Operation not permitted \(\d+\)", normalized)
    if shared_memory:
        return shared_memory.group(0)
    if len(normalized) > 1200:
        return f"{normalized[:600]} ... {normalized[-600:]}"
    return normalized or "no diagnostic output"


def _freesurfer_home() -> Path:
    return Path(os.environ.get("FREESURFER_HOME", Path.home() / "freesurfer"))


def freesurfer_version(home: Path | None = None) -> str:
    stamp = (home or _freesurfer_home()) / "build-stamp.txt"
    if stamp.is_file():
        match = re.search(r"-(\d+\.\d+(?:\.\d+)?)\D", stamp.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return "unknown"


def _synthseg_command(source: Path, destination: Path, home: Path) -> tuple[list[str], dict[str, str]]:
    executable = home / "bin" / "mri_synthseg"
    script = home / "python" / "scripts" / "mri_synthseg"
    if not executable.is_file() or not script.is_file():
        raise FileNotFoundError(f"mri_synthseg or its FreeSurfer script is missing under {home}")
    env = dict(os.environ, FREESURFER_HOME=str(home))
    stamp = (home / "build-stamp.txt").read_text(encoding="utf-8") if (home / "build-stamp.txt").is_file() else ""
    if platform.machine().lower() in {"arm64", "aarch64"} and "x86_64" in stamp:
        env["TF_USE_LEGACY_KERAS"] = "1"
        uv = shutil.which("uv")
        if not uv:
            raise FileNotFoundError("uv is required to run the x86 FreeSurfer SynthSeg script natively on Apple Silicon")
        command = [uv, "run", "--no-project", "--python", "3.12"]
        for package in _SYNTHSEG_NATIVE_PACKAGES:
            command += ["--with", package]
        bootstrap = ("import runpy, sys, numpy as np; "
                     "setattr(np, 'float128', getattr(np, 'float128', np.longdouble)); "
                     "script = sys.argv.pop(1); sys.argv[0] = script; "
                     "runpy.run_path(script, run_name='__main__')")
        command += ["python", "-c", bootstrap, str(script)]
    else:
        command = [str(executable)]
    volumes = destination.with_name("volumes.csv")
    return command + ["--i", str(source), "--o", str(destination), "--vol", str(volumes),
                      "--robust", "--cpu", "--threads", "1", "--parc"], env


def _synthseg_soft_volumes(path: Path, specs: tuple[LabelSpec, ...]) -> dict[str, float] | None:
    """Read SynthSeg parcel mm³, mapping FreeSurfer names to curated label keys."""
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, skipinitialspace=True)
            if not reader.fieldnames:
                return None
            names = [name.strip() if name else "" for name in reader.fieldnames]
            if len(names) != len(set(names)) or not names[0]:
                return None
            reader.fieldnames = names
            row = next(reader, None)
            if row is None:
                return None
            lut_keys = {}
            for spec in specs:
                region, side = spec.key.rsplit("_", 1)
                lut_keys[f"ctx-{'lh' if side == 'left' else 'rh'}-{region.replace('_', '-')}"] = spec.key
            values = {}
            for name in names[1:]:
                if name not in lut_keys:
                    continue
                raw = row.get(name)
                if raw is None:
                    return None
                value = float(raw.strip())
                if not np.isfinite(value) or value < 0:
                    return None
                values[lut_keys[name]] = value / 1000.0
            if set(values) != {spec.key for spec in specs}:
                return None
            return values
    except (OSError, csv.Error, ValueError, TypeError):
        return None


def run_synthseg(image: sitk.Image, grid: sitk.Image, transform: sitk.Transform,
                 volume_id: str) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    """Run SynthSeg 2.0 with DK parcels once, then emit compact MR and gyri items."""
    home = _freesurfer_home()
    version = freesurfer_version(home)
    gyri_specs = synthseg_dk_label_specs(home)
    with tempfile.TemporaryDirectory(prefix="case-capsule-synthseg-") as temporary:
        work = Path(temporary)
        source, target, volume_csv = work / "input.nii.gz", work / "synthseg.nii.gz", work / "volumes.csv"
        clean_nifti(image, source)
        command, env = _synthseg_command(source, target, home)
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
        seconds = round(time.monotonic() - started, 1)
        if result.returncode or not target.is_file():
            details = _error_text(result.stderr or result.stdout)
            raise RuntimeError(f"mri_synthseg failed (exit {result.returncode}): {details}")
        source_labels = sitk.ReadImage(str(target))
        soft_volumes = _synthseg_soft_volumes(volume_csv, gyri_specs)
        source_data = sitk.GetArrayViewFromImage(source_labels)
        source_voxel_ml = float(np.prod(source_labels.GetSpacing())) / 1000.0
        source_hard_ml = {
            spec.key: float(np.count_nonzero(source_data == spec.source_values[0])) * source_voxel_ml
            for spec in gyri_specs
        }
        grid_labels = _resample_synthseg_labels(source_labels, grid, transform)
    packed, labels = encode_label_map(grid_labels, SYNTHSEG_LABELS, grid.GetSpacing())
    source_values, source_counts = np.unique(grid_labels, return_counts=True)
    counts = {int(value): int(count) for value, count in zip(source_values, source_counts)}
    voxel_ml = float(np.prod(grid.GetSpacing())) / 1000.0
    observed_gyri = tuple(
        spec for spec in gyri_specs
        if round(counts.get(spec.source_values[0], 0) * voxel_ml, 6) > 0
    )
    gyri_packed, gyri_labels = encode_label_map(grid_labels, observed_gyri, grid.GetSpacing())
    qc = {"method": "synthseg hard-vs-soft parcel agreement", "ratio_bounds": [0.5, 2.0],
          "max_failing": 2, "calibration": "pilot: 1 template + 2 patient scans"}
    if soft_volumes is None:
        qc.update({"n_failing": None, "failing": [], "verdict": "UNAVAILABLE"})
    else:
        failing = [spec.key for spec in gyri_specs
                   if soft_volumes[spec.key] > 0 and not (
                       0.5 <= source_hard_ml[spec.key] / soft_volumes[spec.key] <= 2.0)]
        qc.update({"n_failing": len(failing), "failing": failing,
                   "verdict": "FAIL" if len(failing) >= 3 else "PASS"})
    for label in gyri_labels:
        label["soft_volume_ml"] = (round(soft_volumes[label["key"]], 6)
                                   if soft_volumes is not None else None)
    method = f"synthseg 2.0 (FreeSurfer {version}, --robust, --parc)"
    mr_item = {"id": "anat_mr", "blob": "anatomy_anat_mr", "for_volume": volume_id, "source": "auto",
               "method": method, "licence": _FREESURFER_LICENSE, "reviewed": False, "labels": labels}
    gyri_item = {"id": "anat_mr_gyri", "blob": "anatomy_anat_mr_gyri", "for_volume": volume_id,
                 "source": "auto", "method": method, "licence": _FREESURFER_LICENSE,
                 "reviewed": False, "labels": gyri_labels, "qc": qc}
    record = {"seconds": seconds, "version": version,
              "command": "mri_synthseg --parc --robust --cpu --vol"}
    return [mr_item, gyri_item], {mr_item["blob"]: packed, gyri_item["blob"]: gyri_packed}, record


def run_totalsegmentator(image: sitk.Image, grid: sitk.Image, transform: sitk.Transform, volume_id: str,
                         task: str, definitions: tuple[LabelSpec, ...], info: dict) -> tuple[dict, np.ndarray, dict]:
    executable = shutil.which("TotalSegmentator")
    if not executable:
        raise FileNotFoundError("TotalSegmentator executable not found")
    version = str(info.get("totalsegmentator_version", "unknown"))
    task_info = info["tasks"][task]
    with tempfile.TemporaryDirectory(prefix=f"case-capsule-{task}-") as temporary:
        work = Path(temporary)
        source, target, report = work / "input.nii.gz", work / "labels.nii.gz", work / "run-report.json"
        clean_nifti(image, source)
        command = [executable, "-i", str(source), "-o", str(target), "-ml", "-ta", task,
                   "-d", "cpu", "-rp", str(report)]
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        seconds = round(time.monotonic() - started, 1)
        if result.returncode or not target.is_file():
            details = _error_text(result.stderr or result.stdout)
            raise RuntimeError(f"TotalSegmentator {task} failed (exit {result.returncode}): {details}")
        source_labels = sitk.ReadImage(str(target))
        grid_labels = resample_labels(source_labels, grid, transform)
        observed = set(int(value) for value in np.unique(grid_labels) if value)
        expected = {int(value) for value in task_info.get("classes", {})}
        unexpected = observed - expected
        if unexpected:
            raise RuntimeError(f"TotalSegmentator {task} returned label IDs absent from its class map: {sorted(unexpected)}")
    anatomy_id = f"anat_ct_{task}"
    packed, labels = encode_label_map(grid_labels, definitions, grid.GetSpacing())
    # A class the tool found nowhere on this case (e.g. larynx outside a head FOV) is not listed as a 0 mL structure.
    labels = [label for label in labels if label["volume_ml"] > 0]
    if not labels:
        raise RuntimeError(f"TotalSegmentator {task} found none of its selected structures on this case")
    item = {"id": anatomy_id, "blob": f"anatomy_{anatomy_id}", "for_volume": volume_id, "source": "auto",
            "method": f"totalsegmentator {version} task {task}", "licence": _TOTALSEG_LICENSE,
            "reviewed": False, "labels": labels}
    return item, packed, {"seconds": seconds, "version": version, "task": task,
                          "device": "cpu", "classes": len(labels)}


def produce_anatomy(volumes: list[dict], grid: sitk.Image) -> tuple[list[dict], dict[str, np.ndarray], list[str], dict]:
    """Run MR SynthSeg and available open CT head tasks, continuing after a task-specific failure."""
    items: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    blocked: list[str] = []
    records: dict[str, dict] = {}
    mr = [volume for volume in volumes if volume["kind"] == "MR"]
    if mr:
        source = next((volume for volume in mr if volume["registration"].get("reference")), mr[0])
        print(f"anatomy anat_mr: starting SynthSeg on {source['id']}", flush=True)
        try:
            mr_items, mr_arrays, record = run_synthseg(source["image"], grid, source["transform"], source["id"])
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
            blocked.append(f"BLOCKED MR anatomy: {exc}")
        else:
            items.extend(mr_items)
            arrays.update(mr_arrays)
            records.update({item["id"]: record for item in mr_items})

    ct = [volume for volume in volumes if volume["kind"] == "CT"]
    if ct:
        source = next((volume for volume in ct if volume["registration"].get("reference")), ct[0])
        try:
            info = totalsegmentator_info()
            task_specs = task_specs_from_info(info)
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
            blocked.append(f"BLOCKED CT anatomy: {exc}")
        else:
            for task in licensed_head_tasks(info):
                blocked.append(f"SKIPPED CT anatomy [{task}]: TotalSegmentator licence required")
            for task, definitions in task_specs:
                print(f"anatomy anat_ct_{task}: starting TotalSegmentator {info.get('totalsegmentator_version', 'unknown')}"
                      f" task {task}", flush=True)
                try:
                    item, array, record = run_totalsegmentator(source["image"], grid, source["transform"],
                                                               source["id"], task, definitions, info)
                except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
                    blocked.append(f"BLOCKED CT anatomy [{task}]: {exc}")
                    continue
                items.append(item)
                arrays[item["blob"]] = array
                records[item["id"]] = record
    return items, arrays, blocked, records
