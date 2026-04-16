"""Optimize resting-state model parameters with differential evolution.

This script replaces brute-force grid search with SciPy differential evolution
for resting-state fit-mode optimization.

Key behaviors:
- Test mode uses run folder 0.
- Non-test runs never use folder 0 (run IDs start at 1).
- Outputs are organized by run ID and pair subfolder:
  hetero-{hetero_label}_aniso-{aniso_label}
- Cache and eval artifacts are local to each pair folder.
- Only fit evaluation is implemented in this version.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import reduce_by_labels
from scipy.optimize import differential_evolution
from scipy.stats import zscore

from heteromodes.restingstate import analyze_bold, calc_node_fc, evaluate_model
from heteromodes.utils import get_project_root, load_hmap
from neuromodes.eigen import EigenSolver
from nsbutils.plotting_pyvista import plot_surf
from nsbutils.utils import unmask

PROJ_DIR = get_project_root()

OBJECTIVE_VERSION = "model_rest_de_fit_v1"

DEFAULT_ALPHA = None
DEFAULT_BETA = None
DEFAULT_ANISO_CURV1 = None
DEFAULT_ANISO_CURV2 = None
DEFAULT_R = None
DEFAULT_GAMMA = None

METRIC_CHOICES = ("edge_fc_corr", "node_fc_corr", "cpc1_corr")
PARAM_ORDER = ("alpha", "beta", "aniso_curv1", "aniso_curv2", "r", "gamma")


@dataclass(frozen=True)
class GridSpec:
    min: float
    max: float
    step: float


@dataclass
class ObjectiveEvaluator:
    surf: str
    medmask: np.ndarray
    parc: Optional[np.ndarray]
    hetero_map: Optional[np.ndarray]
    aniso_map: Optional[np.ndarray]
    emp_outputs: Dict[str, np.ndarray]
    metrics: Sequence[str]
    band_freq: Tuple[float, float]
    scaling: str
    n_modes: int
    n_runs: int
    nt_emp: int
    dt_emp: float
    dt_model: float
    tsteady: int
    param_specs: Dict[str, GridSpec]
    fixed_params: Dict[str, Any]
    cache_dir: Path
    eval_dir: Path
    meta_base: Dict[str, Any]
    cache: Dict[str, float]

    def _resolve_params(self, x: Sequence[float]) -> Dict[str, Optional[float]]:
        params: Dict[str, Optional[float]] = {name: None for name in PARAM_ORDER}
        params.update(self.fixed_params)
        for i, name in enumerate(self.param_specs.keys()):
            spec = self.param_specs[name]
            params[name] = _snap_to_grid(float(x[i]), spec.min, spec.max, spec.step)
        return params

    def _cache_key_and_path(self, params: Dict[str, Optional[float]]) -> Tuple[str, Path]:
        meta = dict(self.meta_base)
        meta.update(params)
        key = _hash_key(meta)
        return key, self.cache_dir / f"eval_{key}.npz"

    def evaluate_params(
        self,
        params: Dict[str, Optional[float]],
        *,
        return_model_outputs: bool = False,
    ) -> Dict[str, Any]:
        cache_key, cache_path = self._cache_key_and_path(params)
        model_cache_path = self.cache_dir / f"eval_{cache_key}_model_outputs.npz"

        has_model_cache = model_cache_path.exists()
        if cache_path.exists() and (not return_model_outputs or has_model_cache):
            with np.load(cache_path, allow_pickle=False) as cached:
                metric_vals = {m: float(cached[m]) for m in self.metrics if m in cached.files}
                score = float(cached["score"])
                objective = float(cached["objective"])
            breadcrumb = {
                "cache_key": cache_key,
                "objective": objective,
                "score": score,
                **params,
                **metric_vals,
            }
            _safe_write_json_once(self.eval_dir / f"{cache_key}.json", breadcrumb)
            out = {
                "cache_key": cache_key,
                "objective": objective,
                "score": score,
                "metrics": metric_vals,
            }
            if return_model_outputs:
                with np.load(model_cache_path, allow_pickle=False) as model_cached:
                    out["model_outputs"] = {k: model_cached[k] for k in model_cached.files}
            return out

        bold_data = _simulate_bold(
            surf=self.surf,
            medmask=self.medmask,
            parc=self.parc,
            hetero_map=self.hetero_map,
            aniso_map=self.aniso_map,
            alpha=params.get("alpha"),
            beta=params.get("beta"),
            aniso_curv1=params.get("aniso_curv1"),
            aniso_curv2=params.get("aniso_curv2"),
            r=params.get("r"),
            gamma=params.get("gamma"),
            scaling=self.scaling,
            n_modes=self.n_modes,
            n_runs=self.n_runs,
            nt_emp=self.nt_emp,
            dt_emp=self.dt_emp,
            dt_model=self.dt_model,
            tsteady=self.tsteady,
        )

        model_outputs = analyze_bold(
            bold_data,
            dt_emp=self.dt_emp,
            band_freq=self.band_freq,
            metrics=list(self.metrics),
        )
        metric_vals = evaluate_model(model_outputs, self.emp_outputs, metrics=list(self.metrics))
        metric_vals = {k: float(v) for k, v in metric_vals.items()}

        score = float(sum(metric_vals[m] for m in self.metrics if m in metric_vals))
        objective = float(-score)
        if not np.isfinite(objective):
            objective = 1e6

        payload: Dict[str, Any] = {
            **self.meta_base,
            **params,
            **metric_vals,
            "cache_key": cache_key,
            "objective": objective,
            "score": score,
        }
        _atomic_savez(cache_path, **payload)
        if return_model_outputs:
            model_payload = {
                k: np.asarray(v)
                for k, v in model_outputs.items()
                if isinstance(v, np.ndarray)
            }
            if model_payload:
                _atomic_savez(model_cache_path, **model_payload)

        breadcrumb = {
            "cache_key": cache_key,
            "objective": objective,
            "score": score,
            **params,
            **metric_vals,
        }
        _safe_write_json_once(self.eval_dir / f"{cache_key}.json", breadcrumb)

        out = {
            "cache_key": cache_key,
            "objective": objective,
            "score": score,
            "metrics": metric_vals,
        }
        if return_model_outputs:
            out["model_outputs"] = model_outputs
        return out

    def __call__(self, x: np.ndarray) -> float:
        params = self._resolve_params(x)
        cache_key, _ = self._cache_key_and_path(params)

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            result = self.evaluate_params(params, return_model_outputs=False)
            objective = float(result["objective"])
            self.cache[cache_key] = objective
            return objective
        except Exception as exc:
            param_str = ", ".join([f"{k}={v}" for k, v in params.items() if v is not None])
            print(f"  ERROR at {param_str}: {type(exc).__name__}: {exc}")
            return 1e6


class TimingCallback:
    def __init__(self, param_specs: Dict[str, GridSpec]) -> None:
        self.param_specs = param_specs
        self.param_names = list(param_specs.keys())
        self.iteration_times: List[float] = []

    def __call__(self, xk: np.ndarray, convergence: float) -> None:
        self.iteration_times.append(time.time())
        param_vals = {}
        for i, name in enumerate(self.param_names):
            spec = self.param_specs[name]
            param_vals[name] = _snap_to_grid(float(xk[i]), spec.min, spec.max, spec.step)

        param_str = ", ".join([f"{name}={val:.4g}" for name, val in param_vals.items()]) or "no free params"
        if len(self.iteration_times) > 1:
            elapsed = self.iteration_times[-1] - self.iteration_times[-2]
            print(f"  Iteration {len(self.iteration_times)}: {elapsed/60:.3f}min | {param_str}, convergence={convergence:.4f}")
        else:
            print(f"  Iteration 1 (initial): {param_str}, convergence={convergence:.4f}")


def _hash_key(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]


def _snap_to_grid(x: float, min_val: float, max_val: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    k = round((x - min_val) / step)
    snapped = min_val + k * step
    return float(np.clip(snapped, min_val, max_val))


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _safe_write_json_once(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except FileExistsError:
        return


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _parse_grid3(values: Tuple[float, float, float], name: str) -> GridSpec:
    min_val, max_val, step = [float(v) for v in values]
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    if step <= 0:
        raise ValueError(f"{name} step must be > 0")
    return GridSpec(min=min_val, max=max_val, step=step)


def _next_non_test_run_id(results_dir: Path) -> int:
    run_ids: List[int] = []
    if results_dir.exists():
        for child in results_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                run_id = int(child.name)
                if run_id > 0:
                    run_ids.append(run_id)
    return (max(run_ids) + 1) if run_ids else 1


def _validate_pair_component(label: Optional[str], arg_name: str) -> str:
    token = str(label)
    if token in {".", ".."}:
        raise ValueError(f"--{arg_name} cannot be '.' or '..'")
    path_obj = Path(token)
    if path_obj.is_absolute() or len(path_obj.parts) != 1:
        raise ValueError(f"--{arg_name} must be a single folder-safe name (no path separators)")
    return token


def _normalize_config_for_id_check(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    normalized.pop("run_hash", None)
    normalized.pop("maxiter", None)
    normalized.pop("popsize", None)
    normalized.pop("n_jobs", None)
    normalized.pop("pair_name", None)
    normalized.pop("pair_dir", None)
    normalized.pop("hetero_label", None)
    normalized.pop("aniso_label", None)
    normalized.pop("optimization_parameters", None)
    normalized.pop("id_config_file", None)
    normalized.pop("config_file", None)
    return normalized


def _collect_config_mismatches(expected: Any, actual: Any, prefix: str = "") -> List[str]:
    mismatches: List[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                mismatches.append(f"{child_prefix}: unexpected key in current config")
                continue
            if key not in actual:
                mismatches.append(f"{child_prefix}: missing from current config")
                continue
            mismatches.extend(_collect_config_mismatches(expected[key], actual[key], child_prefix))
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(f"{prefix}: expected list length {len(expected)}, got {len(actual)}")
            return mismatches
        for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
            mismatches.extend(_collect_config_mismatches(exp_item, act_item, f"{prefix}[{i}]"))
        return mismatches
    if expected != actual:
        mismatches.append(f"{prefix}: expected {expected!r}, got {actual!r}")
    return mismatches


def _resolved_label(aniso_label: Optional[str], hetero_label: Optional[str]) -> Optional[str]:
    if aniso_label is None and hetero_label is not None:
        return hetero_label
    return aniso_label


def _build_param_specs(args: argparse.Namespace) -> Tuple[Dict[str, GridSpec], Dict[str, Any], Dict[str, Any], str]:
    has_beta = args.beta is not None
    has_curv1 = args.aniso_curv1 is not None
    has_curv2 = args.aniso_curv2 is not None

    if has_curv1 != has_curv2:
        raise ValueError("--aniso_curv1 and --aniso_curv2 must be provided together")
    if has_beta and (has_curv1 or has_curv2):
        raise ValueError("beta+aniso_map and aniso_curv1+aniso_curv2 are mutually exclusive")

    if has_beta:
        aniso_mode = "map"
    elif has_curv1 and has_curv2:
        aniso_mode = "curv"
    else:
        aniso_mode = "none"

    defaults = {
        "alpha": DEFAULT_ALPHA,
        "beta": DEFAULT_BETA,
        "aniso_curv1": DEFAULT_ANISO_CURV1,
        "aniso_curv2": DEFAULT_ANISO_CURV2,
        "r": DEFAULT_R,
        "gamma": DEFAULT_GAMMA,
    }

    active_param_names = ["alpha", "r", "gamma"]
    if aniso_mode == "map":
        active_param_names.append("beta")
    elif aniso_mode == "curv":
        active_param_names.extend(["aniso_curv1", "aniso_curv2"])

    specs: Dict[str, GridSpec] = {}
    for name in active_param_names:
        values = getattr(args, name)
        if values is not None:
            specs[name] = _parse_grid3(tuple(values), name)

    fixed_params = {
        name: defaults[name]
        for name in active_param_names
        if name not in specs and defaults[name] is not None
    }

    return specs, fixed_params, defaults, aniso_mode


def _species_constants(species: str, n_subjs: int) -> Tuple[int, float, float, str, int]:
    nt_emp_map = {"human": 1200, "macaque": 500, "marmoset": 510}
    tr_map = {"human": 0.72, "macaque": 2.6, "marmoset": 2.0}
    dt_model_map = {"human": 0.09, "macaque": 0.1, "marmoset": 0.1}
    data_desc_map = {
        "human": f"hcp-s1200_nsubj-{n_subjs}",
        "macaque": f"macaque-awake_nsubj-{n_subjs}",
        "marmoset": f"mbm-v4_nsubj-{n_subjs}",
    }

    nt_emp = nt_emp_map[species]
    dt_emp = tr_map[species]
    dt_model = dt_model_map[species]
    data_desc = data_desc_map[species]
    tsteady = 550

    return nt_emp, dt_emp, dt_model, data_desc, tsteady


def _setup_surface_and_masks(args: argparse.Namespace) -> Tuple[str, Optional[np.ndarray], np.ndarray, str]:
    parc = None
    if args.parc is not None:
        if args.density != "32k":
            raise ValueError("Parcel-based models must be run at 32k density.")
        if args.species != "human":
            raise ValueError("Parcellation is only valid for human species.")

        parc = nib.load(
            str(
                Path(PROJ_DIR)
                / "data"
                / "parcellations"
                / f"parc-{args.parc}_space-fsLR_den-32k_hemi-L.label.gii"
            )
        ).darrays[0].data.astype(int)

        space_desc = f"space-fsLR_den-32k_parc-{args.parc}"
        medmask = parc != 0
    else:
        space_desc = f"space-fsLR_den-{args.density}"
        medmask = nib.load(
            str(
                Path(PROJ_DIR)
                / "data"
                / "empirical"
                / args.species
                / f"space-fsLR_den-{args.density}_hemi-L_desc-nomedialwall.func.gii"
            )
        ).darrays[0].data.astype(bool)

    surf = str(
        Path(PROJ_DIR)
        / "data"
        / "empirical"
        / args.species
        / f"space-fsLR_den-{args.density}_hemi-L_desc-midthickness.surf.gii"
    )

    return surf, parc, medmask, space_desc


def _load_maps(args: argparse.Namespace, medmask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if args.hetero_label is None:
        hetero_map = None
    else:
        if args.hetero_label.startswith("null"):
            split = args.hetero_label.split("-")
            if len(split) != 3:
                raise ValueError("Null map format must be null-{hmap_label}-{null_id}")
            hmap_label = split[1]
            null_id = int(split[2])
            hetero_map = np.load(
                str(
                    Path(PROJ_DIR)
                    / "data"
                    / "nulls"
                    / args.species
                    / f"data-{hmap_label}_space-fsLR_den-{args.density}_hemi-L_nmodes-500_nnulls-1000_nulls_resample-True.npy"
                )
            )[null_id, :]
        else:
            hetero_map = load_hmap(args.hetero_label, species=args.species, density=args.density)

        p_lower, p_upper = np.percentile(hetero_map[medmask], [2, 98])
        hetero_map = np.clip(hetero_map, p_lower, p_upper)

    if args.aniso_label is None:
        aniso_map = None
    else:
        aniso_map = load_hmap(args.aniso_label, species=args.species, density=args.density)

    return hetero_map, aniso_map


def _simulate_bold(
    *,
    surf: str,
    medmask: np.ndarray,
    parc: Optional[np.ndarray],
    hetero_map: Optional[np.ndarray],
    aniso_map: Optional[np.ndarray],
    alpha: Optional[float],
    beta: Optional[float],
    aniso_curv1: Optional[float],
    aniso_curv2: Optional[float],
    r: Optional[float],
    gamma: Optional[float],
    scaling: str,
    n_modes: int,
    n_runs: int,
    nt_emp: int,
    dt_emp: float,
    dt_model: float,
    tsteady: int,
) -> np.ndarray:
    solver_kwargs: Dict[str, Any] = {
        "surf": surf,
        "mask": medmask,
        "scaling": scaling,
    }

    if alpha is not None:
        if hetero_map is None:
            raise ValueError("alpha was set but hetero_map is None")
        solver_kwargs["hetero"] = hetero_map
        solver_kwargs["alpha"] = float(alpha)

    if beta is not None:
        if aniso_map is None:
            raise ValueError("beta was set but aniso_map is None")
        solver_kwargs["aniso_map"] = aniso_map
        solver_kwargs["beta"] = float(beta)
    elif aniso_curv1 is not None and aniso_curv2 is not None:
        solver_kwargs["aniso_curv"] = (float(aniso_curv1), float(aniso_curv2))

    solver = EigenSolver(**solver_kwargs)
    solver.solve(n_modes=int(n_modes), fix_mode1=True, standardize=False, seed=365)

    # Reuse deterministic external inputs across evaluations via neuromodes cache.
    ext_input_cache_dir = Path(PROJ_DIR) / "results" / "human" / "model_rest" / "_cache_ext_input"
    ext_input_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CACHE_DIR"] = str(ext_input_cache_dir)

    downsample_factor = int(dt_emp / dt_model)
    nt_model = int(nt_emp * downsample_factor) + int(tsteady)

    if parc is None:
        n_regions = int(np.sum(medmask))
    else:
        n_regions = len(np.unique(parc[medmask]))
    bold = np.empty((n_regions, nt_emp, n_runs), dtype=np.float32)

    for i in range(n_runs):
        sim_kwargs: Dict[str, Any] = {
            "dt": dt_model,
            "nt": nt_model,
            "seed": i,
            "cache_input": True,
            "bold_out": True,
            "decomp_method": "project",
            "pde_method": "fourier",
        }
        if r is not None:
            sim_kwargs["r"] = float(r)
        if gamma is not None:
            sim_kwargs["gamma"] = float(gamma)

        bold_i = solver.simulate_waves(**sim_kwargs).astype(np.float32)
        bold_i = bold_i[:, tsteady:]
        bold_i = bold_i[:, ::downsample_factor]

        if parc is not None:
            bold_i = reduce_by_labels(bold_i, parc[medmask], axis=1)

        bold[:, :, i] = zscore(bold_i, axis=1).astype(np.float32)

    return bold


def _load_empirical_fit_outputs(
    *,
    species: str,
    metrics: Sequence[str],
    data_desc: str,
    space_desc: str,
    nt_emp: int,
    band_freq: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    outputs: Dict[str, np.ndarray] = {}

    if "edge_fc_corr" in metrics or "node_fc_corr" in metrics:
        fc_file = (
            Path(PROJ_DIR)
            / "data"
            / "empirical"
            / species
            / f"{data_desc}_desc-fc_{space_desc}_hemi-L_nt-{nt_emp}.h5"
        )
        with h5py.File(fc_file, "r") as f:
            outputs["fc"] = np.asarray(f["fc_group"], dtype=np.float32)

    if "cpc1_corr" in metrics:
        cpcs_file = (
            Path(PROJ_DIR)
            / "data"
            / "empirical"
            / species
            / (
                f"{data_desc}_desc-cpcs_{space_desc}_hemi-L_"
                f"freql-{band_freq[0]}_freqh-{band_freq[1]}_nt-{nt_emp}.h5"
            )
        )
        with h5py.File(cpcs_file, "r") as f:
            outputs["cpcs"] = np.asarray(f["cpcs_group"], dtype=np.complex64)

    return outputs


def _plot_fc_heatmap(
    *,
    save_path: Path,
    model_fc: np.ndarray,
    emp_fc: Optional[np.ndarray],
) -> None:
    if emp_fc is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(model_fc, cmap="seismic", vmin=-1.0, vmax=1.0)
        ax.set_title("Model FC")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    im0 = axs[0].imshow(emp_fc, cmap="seismic", vmin=-1.0, vmax=1.0)
    axs[0].set_title("Empirical FC")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(model_fc, cmap="seismic", vmin=-1.0, vmax=1.0)
    axs[1].set_title("Model FC")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("Vertex")
        ax.set_ylabel("Vertex")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_plot_surf_dict(surf_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    surf_gii = nib.load(surf_path)
    vertices = np.asarray(surf_gii.darrays[0].data, dtype=float)
    faces = np.asarray(surf_gii.darrays[1].data, dtype=int)
    return {"lh": {"v": vertices, "t": faces}}


def _plot_brain_model_emp(
    *,
    save_path: Path,
    surf_dict: Dict[str, Dict[str, np.ndarray]],
    medmask: np.ndarray,
    model_vals_masked: np.ndarray,
    emp_vals_masked: np.ndarray,
    title: str,
    cmap: str,
) -> None:

    stacked_masked = np.column_stack([emp_vals_masked, model_vals_masked])
    stacked_full = unmask(stacked_masked, medmask)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_surf(
        surf=surf_dict,
        data={"lh": stacked_full},
        views=["lateral", "medial"],
        layout_indiv="row",
        layout_group="row",
        cmap=cmap,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"{title}: empirical | model")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pairwise_landscape(
    *,
    run_dir: Path,
    free_param_names: Sequence[str],
) -> List[Path]:
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"manifest.csv not found in {run_dir}; skipping landscape plot.")
        return []

    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"manifest.csv is empty in {run_dir}; skipping landscape plot.")
        return []

    if not free_param_names:
        print("No free optimization parameters; skipping landscape plot.")
        return []

    objective = np.asarray([float(r["objective"]) for r in rows], dtype=float)
    best_idx = int(np.argmin(objective))
    saved_paths: List[Path] = []

    if len(free_param_names) == 1:
        p0 = free_param_names[0]
        x = np.asarray([float(r[p0]) for r in rows], dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
        sc = ax.scatter(x, objective, c=objective, cmap="viridis_r", s=45, alpha=0.8, edgecolors="black", linewidth=0.4)
        ax.plot([x[best_idx]], [objective[best_idx]], "r*", markersize=16, label=f"Best objective={objective[best_idx]:.4f}")
        ax.set_xlabel(p0)
        ax.set_ylabel("Objective")
        ax.set_title("Landscape")
        ax.legend(loc="best")
        fig.colorbar(sc, ax=ax).set_label("Objective")
        save_path = run_dir / f"landscape_{p0}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)
        return saved_paths

    for i, p1 in enumerate(free_param_names):
        x = np.asarray([float(r[p1]) for r in rows], dtype=float)
        for j in range(i + 1, len(free_param_names)):
            p2 = free_param_names[j]
            y = np.asarray([float(r[p2]) for r in rows], dtype=float)

            fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
            sc = ax.scatter(
                x,
                y,
                c=objective,
                cmap="viridis_r",
                s=40,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.3,
            )
            ax.plot([x[best_idx]], [y[best_idx]], "r*", markersize=14, label=f"Best objective={objective[best_idx]:.4f}")
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            ax.set_title(f"Landscape: {p1} vs {p2}")
            ax.legend(loc="best")
            fig.colorbar(sc, ax=ax).set_label("Objective")

            save_path = run_dir / f"landscape_{p1}-{p2}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(save_path)

    return saved_paths


def _build_manifest(eval_dir: Path, save_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(eval_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue

    rows = sorted(rows, key=lambda d: float(d.get("objective", np.inf)))

    if not rows:
        return []

    fieldnames = ["cache_key", "objective", "score"] + list(PARAM_ORDER)
    for m in METRIC_CHOICES:
        if m not in fieldnames:
            fieldnames.append(m)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize resting-state model parameters with differential evolution.")
    parser.add_argument("--test", action="store_true", help="Run in test mode using folder 0.")
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Optional run ID for intentional continuation. Non-test mode does not allow ID 0.",
    )

    parser.add_argument("--species", type=str, choices=["human", "macaque", "marmoset"], default="human")
    parser.add_argument("--density", "--den", dest="density", type=str, default="4k", help="Surface density.")

    parser.add_argument("--hetero_label", type=lambda x: None if x.lower() == "none" else x, default=None)
    parser.add_argument("--aniso_label", type=lambda x: None if x.lower() == "none" else x, default=None)

    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_modes", type=int, default=500)
    parser.add_argument("--n_subjs", type=int, default=255)
    parser.add_argument("--band_freq", type=float, nargs=2, default=[0.04, 0.07])
    parser.add_argument("--scaling", type=str, default="sigmoid")
    parser.add_argument("--parc", type=lambda x: None if x.lower() == "none" else x, default=None)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=list(METRIC_CHOICES),
        default=["edge_fc_corr", "node_fc_corr", "cpc1_corr"],
        help="Fit metrics to optimize (fcd_ks is intentionally not supported).",
    )

    parser.add_argument(
        "--evaluation",
        type=str,
        choices=["fit", "crossval"],
        default="fit",
        help="Evaluation mode. Only fit is implemented in this version.",
    )

    parser.add_argument("--alpha", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--beta", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--aniso_curv1", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--aniso_curv2", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--r", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--gamma", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))

    parser.add_argument("--maxiter", type=int, default=50, help="Maximum differential-evolution iterations.")
    parser.add_argument("--popsize", type=int, default=16, help="Population size multiplier for differential evolution.")
    parser.add_argument("--seed", type=int, default=365, help="Seed for differential evolution initialization.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers for differential evolution.")
    parser.add_argument("--polish", action="store_true")

    args = parser.parse_args()

    # Deduplicate metrics while preserving order.
    args.metrics = list(dict.fromkeys(args.metrics))
    return args


def main() -> None:
    t0 = time.time()
    args = parse_args()

    if args.evaluation != "fit":
        raise NotImplementedError("Only --evaluation fit is implemented in optimisation_de.py")

    if args.id is not None and int(args.id) < 0:
        raise ValueError("--id must be >= 0")
    if args.test and args.id is not None:
        raise ValueError("--id cannot be used with --test")
    if (not args.test) and args.id == 0:
        raise ValueError("Run ID 0 is reserved for --test mode; use --id >= 1")

    args.aniso_label = _resolved_label(args.aniso_label, args.hetero_label)

    if args.metrics is None or len(args.metrics) == 0:
        raise ValueError("At least one metric must be supplied via --metrics")

    param_specs, fixed_params, defaults, aniso_mode = _build_param_specs(args)
    if aniso_mode == "map" and args.aniso_label is None:
        raise ValueError("beta optimization requires --aniso_label (or --hetero_label so aniso can resolve from it)")

    free_param_names = list(param_specs.keys())
    if not free_param_names:
        print("No free optimization parameters provided; evaluating the fixed default model only.")

    hetero_token = _validate_pair_component(args.hetero_label, "hetero_label")
    aniso_token = _validate_pair_component(args.aniso_label, "aniso_label")
    pair_name = f"hetero-{hetero_token}-aniso-{aniso_token}"

    results_dir = Path(PROJ_DIR) / "results" / args.species / "model_rest" / "de"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.test:
        print("Running in test mode with fixed run ID 0. Existing contents of this folder will be deleted.")
        run_id = 0
        run_parent = results_dir / "0"
        if run_parent.exists():
            shutil.rmtree(run_parent)
        run_parent.mkdir(parents=True, exist_ok=False)
    else:
        if args.id is None:
            run_id = _next_non_test_run_id(results_dir)
            run_parent = results_dir / str(run_id)
            while True:
                try:
                    run_parent.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    run_id = _next_non_test_run_id(results_dir)
                    run_parent = results_dir / str(run_id)
        else:
            run_id = int(args.id)
            if run_id == 0:
                raise ValueError("Run ID 0 is reserved for --test mode")
            run_parent = results_dir / str(run_id)
            run_parent.mkdir(parents=True, exist_ok=True)

        print(f"Using run ID {run_id} with parent folder {run_parent}")

    pair_dir = run_parent / pair_name
    if not pair_dir.exists():
        pair_dir.mkdir(parents=True, exist_ok=False)

    cache_dir = pair_dir / "_cache"
    eval_dir = pair_dir / "evals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    nt_emp, dt_emp, dt_model, data_desc, tsteady = _species_constants(args.species, args.n_subjs)
    surf, parc, medmask, space_desc = _setup_surface_and_masks(args)
    hetero_map, aniso_map = _load_maps(args, medmask)

    print("Loading empirical fit outputs...")
    band_freq = (float(args.band_freq[0]), float(args.band_freq[1]))
    emp_outputs = _load_empirical_fit_outputs(
        species=args.species,
        metrics=args.metrics,
        data_desc=data_desc,
        space_desc=space_desc,
        nt_emp=nt_emp,
        band_freq=band_freq,
    )

    id_config = {
        "schema_version": 1,
        "objective_version": OBJECTIVE_VERSION,
        "run_id": int(run_id),
        "test_mode": bool(args.test),
        "species": args.species,
        "density": args.density,
        "evaluation": args.evaluation,
        "metrics": list(args.metrics),
        "n_runs": int(args.n_runs),
        "n_modes": int(args.n_modes),
        "n_subjs": int(args.n_subjs),
        "band_freq": [float(v) for v in band_freq],
        "scaling": args.scaling,
        "parc": args.parc,
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "seed": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "polish": bool(args.polish),
        "anisotropy_mode": aniso_mode,
        "defaults": defaults,
        "active_parameter_names": free_param_names,
        "fixed_params": fixed_params,
    }

    id_config_path = run_parent / "id_config.json"
    if id_config_path.exists() and not args.test:
        saved = json.loads(id_config_path.read_text(encoding="utf-8"))
        mismatches = _collect_config_mismatches(
            _normalize_config_for_id_check(saved),
            _normalize_config_for_id_check(id_config),
        )
        if mismatches:
            mismatch_msg = "\n  - " + "\n  - ".join(mismatches[:12])
            raise ValueError(
                f"Provided --id {run_id} has parameter mismatches against {id_config_path}:"
                f"{mismatch_msg}"
            )

    run_config = {
        **id_config,
        "pair_name": pair_name,
        "pair_dir": str(pair_dir),
        "hetero_label": args.hetero_label,
        "aniso_label": args.aniso_label,
        "optimization_parameters": {name: asdict(spec) for name, spec in param_specs.items()},
        "id_config_file": str(id_config_path),
        "config_file": str(pair_dir / "config.json"),
    }
    run_config["run_hash"] = _hash_key(run_config)

    _atomic_write_json(id_config_path, id_config)
    _atomic_write_json(pair_dir / "config.json", run_config)
    (pair_dir / "run_hash.txt").write_text(f"{run_config['run_hash']}\n", encoding="utf-8")

    print("Starting optimization with differential evolution...")
    evaluator = ObjectiveEvaluator(
        surf=surf,
        medmask=medmask,
        parc=parc,
        hetero_map=hetero_map,
        aniso_map=aniso_map,
        emp_outputs=emp_outputs,
        metrics=args.metrics,
        band_freq=band_freq,
        scaling=args.scaling,
        n_modes=int(args.n_modes),
        n_runs=int(args.n_runs),
        nt_emp=int(nt_emp),
        dt_emp=float(dt_emp),
        dt_model=float(dt_model),
        tsteady=int(tsteady),
        param_specs=param_specs,
        fixed_params=fixed_params,
        cache_dir=cache_dir,
        eval_dir=eval_dir,
        meta_base={
            "objective_version": OBJECTIVE_VERSION,
            "run_hash": run_config["run_hash"],
            "species": args.species,
            "density": args.density,
            "evaluation": args.evaluation,
            "metrics": list(args.metrics),
            "n_runs": int(args.n_runs),
            "n_modes": int(args.n_modes),
            "n_subjs": int(args.n_subjs),
            "band_freq": [float(v) for v in band_freq],
            "scaling": args.scaling,
            "parc": args.parc,
            "hetero_label": args.hetero_label,
            "aniso_label": args.aniso_label,
            "anisotropy_mode": aniso_mode,
        },
        cache={},
    )

    free_bounds = [(spec.min, spec.max) for spec in param_specs.values()]
    single_point_de_result: Optional[Dict[str, Any]] = None
    single_point_best_eval: Optional[Dict[str, Any]] = None
    if free_bounds:
        is_single_point = all(spec.min == spec.max for spec in param_specs.values())
        if is_single_point:
            print("Single-point mode: skipping differential_evolution and evaluating once.")
            best_params = {name: None for name in PARAM_ORDER}
            best_params.update(fixed_params)
            for name, spec in param_specs.items():
                best_params[name] = spec.min

            x_fixed = np.array([best_params[name] for name in param_specs.keys()], dtype=float)
            single_point_best_eval = evaluator.evaluate_params(best_params, return_model_outputs=True)
            fun = float(single_point_best_eval["objective"])
            evaluator.cache[single_point_best_eval["cache_key"]] = fun
            single_point_de_result = {
                "x": [float(v) for v in x_fixed],
                "fun": fun,
                "nfev": 1,
                "nit": 0,
                "success": True,
                "message": "Single-point evaluation (all bounds fixed).",
            }
            result = None
        else:
            timing_callback = TimingCallback(param_specs)
            result = differential_evolution(
                evaluator,
                bounds=free_bounds,
                seed=int(args.seed),
                maxiter=int(args.maxiter),
                popsize=int(args.popsize),
                polish=bool(args.polish),
                workers=int(args.n_jobs),
                updating="deferred" if int(args.n_jobs) != 1 else "immediate",
                callback=timing_callback,
                disp=True,
            )
            best_params = {name: None for name in PARAM_ORDER}
            best_params.update(fixed_params)
            for i, name in enumerate(param_specs.keys()):
                spec = param_specs[name]
                best_params[name] = _snap_to_grid(float(result.x[i]), spec.min, spec.max, spec.step)
    else:
        result = None
        best_params = {name: None for name in PARAM_ORDER}
        best_params.update(fixed_params)

    if single_point_best_eval is not None:
        best_eval = single_point_best_eval
    else:
        best_eval = evaluator.evaluate_params(best_params, return_model_outputs=True)
    best_metrics = dict(best_eval["metrics"])
    best_objective = float(best_eval["objective"])
    best_score = float(best_eval["score"])

    best_json = {
        "run_hash": run_config["run_hash"],
        "cache_key": best_eval["cache_key"],
        "objective": best_objective,
        "score": best_score,
        **best_metrics,
        **best_params,
    }
    (pair_dir / "best.json").write_text(json.dumps(best_json, indent=2, sort_keys=True), encoding="utf-8")

    if single_point_de_result is not None:
        de_result = single_point_de_result
    else:
        de_result = {
            "x": [float(v) for v in result.x] if result is not None else [],
            "fun": float(result.fun) if result is not None else best_objective,
            "nfev": int(result.nfev) if result is not None else 1,
            "nit": int(result.nit) if result is not None else 0,
            "success": bool(result.success) if result is not None else True,
            "message": str(result.message) if result is not None else "evaluated fixed default parameters",
        }
    (pair_dir / "de_result.json").write_text(json.dumps(de_result, indent=2, sort_keys=True), encoding="utf-8")

    rows = _build_manifest(eval_dir, pair_dir / "manifest.csv")
    if rows:
        print(f"Saved manifest with {len(rows)} rows")

    model_outputs = best_eval.get("model_outputs", {})

    if "edge_fc_corr" in args.metrics and "fc" in model_outputs:
        _plot_fc_heatmap(
            save_path=pair_dir / "edge_fc_corr_fc_matrix.png",
            model_fc=np.asarray(model_outputs["fc"], dtype=float),
            emp_fc=np.asarray(emp_outputs.get("fc"), dtype=float) if "fc" in emp_outputs else None,
        )

    if parc is None:
        surf_dict = _load_plot_surf_dict(surf)

        if "node_fc_corr" in args.metrics and "fc" in model_outputs and "fc" in emp_outputs:
            model_node = calc_node_fc(np.asarray(model_outputs["fc"], dtype=float))
            emp_node = calc_node_fc(np.asarray(emp_outputs["fc"], dtype=float))
            _plot_brain_model_emp(
                save_path=pair_dir / "node_fc_corr_brain_map.png",
                surf_dict=surf_dict,
                medmask=medmask,
                model_vals_masked=np.asarray(model_node, dtype=float),
                emp_vals_masked=np.asarray(emp_node, dtype=float),
                title="Node FC",
                cmap="turbo",
            )

        if "cpc1_corr" in args.metrics and "cpcs" in model_outputs and "cpcs" in emp_outputs:
            model_cpc1 = np.imag(np.asarray(model_outputs["cpcs"])[:, 0])
            emp_cpc1 = np.imag(np.asarray(emp_outputs["cpcs"])[:, 0])
            _plot_brain_model_emp(
                save_path=pair_dir / "cpc1_corr_brain_map.png",
                surf_dict=surf_dict,
                medmask=medmask,
                model_vals_masked=np.asarray(model_cpc1, dtype=float),
                emp_vals_masked=np.asarray(emp_cpc1, dtype=float),
                title="CPC1 (imag)",
                cmap="turbo",
            )
    else:
        if "node_fc_corr" in args.metrics or "cpc1_corr" in args.metrics:
            print("Skipping node/cpc brain maps because --parc was used (not vertex-level data).")

    _ = _plot_pairwise_landscape(run_dir=pair_dir, free_param_names=free_param_names)

    print(f"Run parent folder (ID={run_id}): {run_parent}")
    print(f"Pair folder: {pair_dir}")
    print(f"Total optimisation time: {(time.time() - t0)/3600:.3f} hrs")


if __name__ == "__main__":
    main()
