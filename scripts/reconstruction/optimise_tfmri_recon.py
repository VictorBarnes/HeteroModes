"""Optimize tfMRI reconstruction AUC with optional anisotropic search.

This script keeps the TTP-style parameter handling, caching, and run-folder
bookkeeping, but swaps the objective for tfMRI reconstruction-error AUC.
K@target is reported as a diagnostic only. Isotropic baseline results are cached in
a persistent folder so they can be reused across runs.
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
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from heteromodes.utils import get_project_root, load_hmap
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf

# TODO: add a weighted-AUC objective later if we want to emphasize early-mode performance more strongly.

PROJ_DIR = get_project_root()
DEFAULT_TFMRI_FILE = Path(PROJ_DIR) / "data" / "empirical" / "human" / "S255_tfMRI_ALLTASKS_raw_lh.mat"
DEFAULT_ALPHA = None
DEFAULT_BETA = None
DEFAULT_ANISO_CURV1 = None
DEFAULT_ANISO_CURV2 = None
OBJECTIVE_VERSION = "tfmri_recon_auc_v2_error_target"


def _hash_key(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]


def _snap_to_grid(x: float, min: float, max: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    k = round((x - min) / step)
    snapped = min + k * step
    return float(np.clip(snapped, min, max))


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


@dataclass(frozen=True)
class GridSpec:
    min: float
    max: float
    step: float


def _parse_grid3(values: Tuple[float, float, float], name: str) -> GridSpec:
    min, max, step = [float(v) for v in values]
    if max < min:
        min, max = max, min
    if step <= 0:
        raise ValueError(f"{name} step must be > 0")
    return GridSpec(min=min, max=max, step=step)


def _next_run_id(results_dir: Path) -> int:
    run_ids = []
    if results_dir.exists():
        for child in results_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                run_ids.append(int(child.name))
    return (max(run_ids) + 1) if run_ids else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize tfMRI reconstruction AUC with optional anisotropic parameters.")
    parser.add_argument("--test", action="store_true", help="Run in test mode using folder 0 and test caches.")
    parser.add_argument("--density", default="32k", help="Surface density passed to neuromodes.io.fetch_surf.")
    parser.add_argument("--contrast", default="motor_cue_avg", help="Task contrast name inside the mat file.")
    parser.add_argument("--n_subj", type=int, default=255, help="Number of subjects to use.")
    parser.add_argument("--n_modes", type=int, default=100, help="Maximum number of modes to compute.")
    parser.add_argument("--mode_step", type=int, default=1, help="Spacing between mode counts in the AUC curve.")
    parser.add_argument("--error_target", type=float, default=0.5, help="Absolute reconstruction-error target for parsimony mode count K@target (target >= 0).")
    parser.add_argument("--hetero_label", default=None, help="Heterogeneity map label.")
    parser.add_argument("--aniso_label", default=None, help="Anisotropy map label.")
    parser.add_argument("--alpha", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Alpha optimization range: MIN MAX STEP")
    parser.add_argument("--beta", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Beta optimization range: MIN MAX STEP")
    parser.add_argument("--aniso_curv1", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Aniso_curv1 optimization range: MIN MAX STEP")
    parser.add_argument("--aniso_curv2", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Aniso_curv2 optimization range: MIN MAX STEP")
    parser.add_argument("--maxiter", type=int, default=50, help="Maximum differential-evolution iterations.")
    parser.add_argument("--popsize", type=int, default=16, help="Population size multiplier for differential evolution.")
    parser.add_argument("--seed", type=int, default=365, help="Random seed for differential evolution.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers for differential evolution.")
    parser.add_argument("--polish", action="store_true", help="Enable differential evolution polish step.")
    parser.add_argument("--plot", action="store_true", help="Plot the isotropic and optimized anisotropic curves.")
    # parser.add_argument("--output_json", default=None, help="Optional path to save optimization results as JSON.")
    return parser.parse_args()


def load_task_maps(contrast: str, medmask: np.ndarray, n_subj: int) -> np.ndarray:
    with h5py.File(DEFAULT_TFMRI_FILE, "r") as f:
        data = f["zstat"][contrast][:, medmask].T
        if data.shape[1] < n_subj:
            raise ValueError(f"Requested task maps from {n_subj} but only {data.shape[1]} are available")
        task_maps = np.asarray(data[:, :n_subj], dtype=float)
    if task_maps.ndim == 1:
        task_maps = task_maps[:, np.newaxis]
    return task_maps


def curve_auc(mode_counts: np.ndarray, curve: np.ndarray) -> float:
    return float(np.trapezoid(curve, mode_counts))


def k_target_mode(mode_counts: np.ndarray, curve: np.ndarray, target: float) -> int:
    hit_idx = np.where(curve <= float(target))[0]
    return int(mode_counts[hit_idx[0]]) if len(hit_idx) else int(mode_counts[-1])


def _cached_k_target(cached: Any, target: float, mode_counts: np.ndarray, curve: np.ndarray) -> int:
    if "k_target" in cached:
        return int(cached["k_target"])
    if "k95" in cached:
        return int(cached["k95"])
    return int(k_target_mode(mode_counts, curve, target))


def _resolved_label(aniso_label: str | None, hetero_label: str) -> str:
    return hetero_label if aniso_label is None else aniso_label


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
    }

    active_param_names = ["alpha"]
    if aniso_mode == "map":
        active_param_names.append("beta")
    elif aniso_mode == "curv":
        active_param_names.extend(["aniso_curv1", "aniso_curv2"])

    specs: Dict[str, GridSpec] = {}
    for name in active_param_names:
        values = getattr(args, name)
        if values is not None:
            specs[name] = _parse_grid3(tuple(values), name)

    fixed_params = {name: defaults[name] for name in active_param_names if name not in specs and defaults[name] is not None}
    return specs, fixed_params, defaults, aniso_mode


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
        param_str = ", ".join([f"{name}={val:.2f}" for name, val in param_vals.items()]) or "no free params"
        if len(self.iteration_times) > 1:
            elapsed = self.iteration_times[-1] - self.iteration_times[-2]
            print(f"  Iteration {len(self.iteration_times)}: {elapsed/60:.3f}min | {param_str}, convergence={convergence:.4f}")
        else:
            print(f"  Iteration 1 (initial): {param_str}, convergence={convergence:.4f}")


@dataclass
class ObjectiveEvaluator:
    mesh: Any
    medmask: np.ndarray
    task_maps: np.ndarray
    mode_counts: np.ndarray
    hetero_map: np.ndarray
    aniso_map: np.ndarray
    param_specs: Dict[str, GridSpec]
    fixed_params: Dict[str, Any]
    cache_dir: Path
    eval_dir: Path
    meta_base: Dict[str, Any]
    cache: Dict[str, float]
    error_target: float

    def _resolve_params(self, x: Sequence[float]) -> Dict[str, Any]:
        param_values = dict(self.fixed_params)
        for i, name in enumerate(self.param_specs.keys()):
            spec = self.param_specs[name]
            param_values[name] = _snap_to_grid(float(x[i]), spec.min, spec.max, spec.step)
        return param_values

    def _cache_key_and_path(self, param_values: Dict[str, Any]) -> Tuple[str, Path]:
        meta = dict(self.meta_base)
        meta.update(param_values)
        key = _hash_key(meta)
        return key, self.cache_dir / f"eval_{key}.npz"

    def __call__(self, x: np.ndarray) -> float:
        param_values = self._resolve_params(x)
        cache_key, cache_path = self._cache_key_and_path(param_values)

        if cache_key in self.cache:
            return self.cache[cache_key]

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            objective = float(cached["objective"])
            self.cache[cache_key] = objective
            cached_curve = np.asarray(cached["curve"], dtype=float) if "curve" in cached else np.asarray([])
            k_target = _cached_k_target(cached, self.error_target, self.mode_counts, cached_curve if cached_curve.size else np.zeros_like(self.mode_counts, dtype=float))
            breadcrumb = dict(param_values)
            breadcrumb.update({
                "cache_key": cache_key,
                "objective": objective,
                "auc": objective,
                "k_target": k_target,
                "error_target": float(self.error_target),
            })
            _safe_write_json_once(self.eval_dir / f"{cache_key}.json", breadcrumb)
            return objective

        try:
            solver_kwargs: Dict[str, Any] = {
                "surf": self.mesh,
                "mask": self.medmask,
            }
            alpha_val = param_values.get("alpha")
            beta_val = param_values.get("beta")
            curv1_val = param_values.get("aniso_curv1")
            curv2_val = param_values.get("aniso_curv2")
            if alpha_val is not None:
                solver_kwargs["hetero"] = self.hetero_map
                solver_kwargs["alpha"] = float(alpha_val)
            if beta_val is not None:
                solver_kwargs["aniso_map"] = self.aniso_map
                solver_kwargs["beta"] = float(beta_val)
            elif curv1_val is not None and curv2_val is not None:
                solver_kwargs["aniso_curv"] = (float(curv1_val), float(curv2_val))
            solver = EigenSolver(**solver_kwargs)
            solver.solve(n_modes=int(self.mode_counts.max()))
            _, recon_error, _ = solver.reconstruct(data=self.task_maps, mode_counts=self.mode_counts)
            curve = np.mean(recon_error, axis=1)
            objective = curve_auc(self.mode_counts, curve)
            k_target = k_target_mode(self.mode_counts, curve, self.error_target)

            payload = dict(self.meta_base)
            payload.update(param_values)
            payload.update({
                "cache_key": cache_key,
                "objective": objective,
                "auc": objective,
                "k_target": k_target,
                "error_target": float(self.error_target),
                "curve": curve.astype(np.float32),
                "mode_counts": self.mode_counts.astype(np.int32),
            })
            _atomic_savez(cache_path, **payload)
            _safe_write_json_once(self.eval_dir / f"{cache_key}.json", {
                **param_values,
                "cache_key": cache_key,
                "objective": objective,
                "auc": objective,
                "k_target": k_target,
                "error_target": float(self.error_target),
            })
            self.cache[cache_key] = objective
            return objective
        except Exception as exc:
            param_str = ", ".join([f"{name}={val:.4f}" for name, val in param_values.items()])
            print(f"  ERROR at {param_str}: {type(exc).__name__}: {exc}")
            return 1e6


def evaluate_solver(solver: EigenSolver, task_maps: np.ndarray, mode_counts: np.ndarray, error_target: float) -> Tuple[np.ndarray, float, int]:
    _, recon_error, _ = solver.reconstruct(data=task_maps, mode_counts=mode_counts)
    curve = np.mean(recon_error, axis=1)
    auc = curve_auc(mode_counts, curve)
    k_target = k_target_mode(mode_counts, curve, error_target)
    return curve, auc, k_target


def evaluate_solver_stats(
    solver: EigenSolver,
    task_maps: np.ndarray,
    mode_counts: np.ndarray,
    error_target: float,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    _, recon_error, _ = solver.reconstruct(data=task_maps, mode_counts=mode_counts)
    recon_error = np.asarray(recon_error, dtype=float)
    mean_curve = np.mean(recon_error, axis=1)
    std_curve = np.std(recon_error, axis=1)
    auc = curve_auc(mode_counts, mean_curve)
    k_target = k_target_mode(mode_counts, mean_curve, error_target)
    return mean_curve, std_curve, auc, k_target


def plot_auc_landscape(
    *,
    run_dir: Path,
    free_param_names: Sequence[str],
    contrast: str,
    error_target: float,
) -> Path | None:
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"manifest.csv not found in {run_dir}; skipping landscape plot.")
        return None

    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"manifest.csv is empty in {run_dir}; skipping landscape plot.")
        return None

    if not free_param_names:
        print("No free optimization parameters; skipping landscape plot.")
        return None

    auc_vals = np.asarray([float(r["auc"]) for r in rows], dtype=float)
    k_target_vals = np.asarray([float(r["k_target"]) for r in rows], dtype=float)
    fig = None

    if len(free_param_names) == 1:
        p0 = free_param_names[0]
        x = np.asarray([float(r[p0]) for r in rows], dtype=float)
        fig, (ax_auc, ax_k) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        scatter_auc = ax_auc.scatter(
            x,
            auc_vals,
            c=auc_vals,
            cmap="turbo",
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )
        best_idx = int(np.argmin(auc_vals))
        ax_auc.plot(x[best_idx], auc_vals[best_idx], "r*", markersize=16, label=f"Best AUC={auc_vals[best_idx]:.4f}")
        ax_auc.set_xlabel(p0)
        ax_auc.set_ylabel("AUC")
        ax_auc.set_title("AUC Landscape")
        ax_auc.legend(loc="best")
        fig.colorbar(scatter_auc, ax=ax_auc).set_label("AUC")

        scatter_k = ax_k.scatter(
            x,
            k_target_vals,
            c=k_target_vals,
            cmap="viridis",
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )
        ax_k.plot(x[best_idx], k_target_vals[best_idx], "r*", markersize=16, label=f"Best K={k_target_vals[best_idx]:.0f}")
        ax_k.set_xlabel(p0)
        ax_k.set_ylabel(f"K@err<={float(error_target):.3f}")
        ax_k.set_title("Modes to Error Target")
        ax_k.legend(loc="best")
        fig.colorbar(scatter_k, ax=ax_k).set_label("Modes")
    elif len(free_param_names) == 2:
        p0, p1 = free_param_names[:2]
        x = np.asarray([float(r[p0]) for r in rows], dtype=float)
        y = np.asarray([float(r[p1]) for r in rows], dtype=float)
        fig, (ax_auc, ax_k) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        scatter_auc = ax_auc.scatter(
            x,
            y,
            c=auc_vals,
            cmap="turbo",
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )
        best_idx = int(np.argmin(auc_vals))
        ax_auc.plot(x[best_idx], y[best_idx], "r*", markersize=18, label=f"Best AUC={auc_vals[best_idx]:.4f}")
        ax_auc.set_xlabel(p0)
        ax_auc.set_ylabel(p1)
        ax_auc.set_title("AUC Landscape")
        ax_auc.legend(loc="best")
        fig.colorbar(scatter_auc, ax=ax_auc).set_label("AUC")

        scatter_k = ax_k.scatter(
            x,
            y,
            c=k_target_vals,
            cmap="viridis",
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )
        ax_k.plot(x[best_idx], y[best_idx], "r*", markersize=18, label=f"Best K={k_target_vals[best_idx]:.0f}")
        ax_k.set_xlabel(p0)
        ax_k.set_ylabel(p1)
        ax_k.set_title("Modes to Error Target")
        ax_k.legend(loc="best")
        fig.colorbar(scatter_k, ax=ax_k).set_label("Modes")
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        p0, p1, p2 = free_param_names[:3]
        x = np.asarray([float(r[p0]) for r in rows], dtype=float)
        y = np.asarray([float(r[p1]) for r in rows], dtype=float)
        z = np.asarray([float(r[p2]) for r in rows], dtype=float)
        fig = plt.figure(figsize=(14, 6))
        ax_auc = fig.add_subplot(1, 2, 1, projection="3d")
        ax_k = fig.add_subplot(1, 2, 2, projection="3d")

        scatter_auc = ax_auc.scatter(
            x,
            y,
            z,
            c=auc_vals,
            cmap="turbo",
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )
        best_idx = int(np.argmin(auc_vals))
        ax_auc.scatter([x[best_idx]], [y[best_idx]], [z[best_idx]], color="red", s=450, marker="*")
        ax_auc.set_xlabel(p0)
        ax_auc.set_ylabel(p1)
        ax_auc.set_zlabel(p2)
        ax_auc.set_title("AUC Landscape")
        fig.colorbar(scatter_auc, ax=ax_auc, shrink=0.75).set_label("AUC")

        scatter_k = ax_k.scatter(
            x,
            y,
            z,
            c=k_target_vals,
            cmap="viridis",
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )
        ax_k.scatter([x[best_idx]], [y[best_idx]], [z[best_idx]], color="red", s=450, marker="*")
        ax_k.set_xlabel(p0)
        ax_k.set_ylabel(p1)
        ax_k.set_zlabel(p2)
        ax_k.set_title("Modes to Error Target")
        fig.colorbar(scatter_k, ax=ax_k, shrink=0.75).set_label("Modes")

    if fig is None:
        return None

    title_params = ", ".join(list(free_param_names[:3]))
    fig.suptitle(f"Landscape Overview ({title_params})", y=1.02)
    save_path = run_dir / f"{contrast}_landscape.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_best_reconstruction_curves(
    *,
    run_dir: Path,
    contrast: str,
    mode_counts: np.ndarray,
    iso_mean: np.ndarray,
    iso_std: np.ndarray,
    aniso_mean: np.ndarray,
    aniso_std: np.ndarray,
    isotropic_auc: float,
    isotropic_k_target: int,
    anisotropic_auc: float,
    anisotropic_k_target: int,
    error_target: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    k_label = f"K@err<={float(error_target):.3f}"

    ax.plot(mode_counts, iso_mean, color="tab:blue", label=f"Isotropic AUC={isotropic_auc:.3f}, {k_label}={isotropic_k_target}")
    ax.plot(mode_counts, aniso_mean, color="tab:red", label=f"Anisotropic AUC={anisotropic_auc:.3f}, {k_label}={anisotropic_k_target}")

    ax.fill_between(mode_counts, iso_mean - iso_std, iso_mean + iso_std, color="tab:blue", alpha=0.2)
    ax.fill_between(mode_counts, aniso_mean - aniso_std, aniso_mean + aniso_std, color="tab:red", alpha=0.2)

    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Reconstruction error")
    ax.legend(loc="upper right")
    ax.set_title("Best Reconstruction Curves")
    fig.tight_layout()

    save_path = run_dir / f"{contrast}_best-recon-curv.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _compute_isotropic_reconstruction(
    *,
    results_dir: Path,
    mesh: Any,
    medmask: np.ndarray,
    task_maps: np.ndarray,
    mode_counts: np.ndarray,
    meta_base: Dict[str, Any],
    error_target: float,
    cache_subdir: str = "_iso_cache",
) -> Dict[str, Any]:
    isotropic_dir = results_dir / cache_subdir
    isotropic_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _hash_key({**meta_base, "baseline": "isotropic"})
    cache_path = isotropic_dir / f"iso_{cache_key}.npz"
    json_path = isotropic_dir / f"iso_{cache_key}.json"

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        curve = np.asarray(cached["curve"], dtype=float)
        result = {
            "cache_key": cache_key,
            "cache_path": str(cache_path),
            "curve": curve.tolist(),
            "auc": float(cached["auc"]),
            "k_target": _cached_k_target(cached, error_target, mode_counts, curve),
            "error_target": float(error_target),
            "objective": float(cached["objective"]),
        }
        _safe_write_json_once(json_path, result)
        return result

    solver = EigenSolver(surf=mesh, mask=medmask)
    solver.solve(n_modes=int(mode_counts.max()))
    curve, auc, k_target = evaluate_solver(solver, task_maps, mode_counts, error_target)
    payload = {
        **meta_base,
        "baseline": "isotropic",
        "cache_key": cache_key,
        "objective": auc,
        "auc": auc,
        "k_target": k_target,
        "error_target": float(error_target),
        "curve": curve.astype(np.float32),
        "mode_counts": mode_counts.astype(np.int32),
    }
    _atomic_savez(cache_path, **payload)
    _safe_write_json_once(json_path, {
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "objective": auc,
        "auc": auc,
        "k_target": k_target,
        "error_target": float(error_target),
    })
    return {
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "curve": curve.tolist(),
        "auc": auc,
        "k_target": k_target,
        "error_target": float(error_target),
        "objective": auc,
    }


def main() -> None:
    t1 = time.time()
    args = parse_args()
    if float(args.error_target) < 0.0:
        raise ValueError("--error-target must satisfy target >= 0")
    args.aniso_label = _resolved_label(args.aniso_label, args.hetero_label)

    results_dir = Path(PROJ_DIR) / "results" / "human" / "reconstruction"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.test:
        run_id = 0
        run_dir = results_dir / "0"
        cache_dir = run_dir / "_cache_test"
        iso_cache_subdir = "_iso_cache_test"
        if run_dir.exists():
            shutil.rmtree(run_dir)
    else:
        run_id = _next_run_id(results_dir)
        run_dir = results_dir / str(run_id)
        cache_dir = results_dir / "_cache"
        iso_cache_subdir = "_iso_cache"
        while True:
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                run_id = _next_run_id(results_dir)
                run_dir = results_dir / str(run_id)

    eval_dir = run_dir / "evals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    param_specs, fixed_params, defaults, aniso_mode = _build_param_specs(args)
    free_param_names = list(param_specs.keys())

    if not free_param_names:
        print("No free optimization parameters provided; evaluating the fixed default model only.")

    # Must start at 2 modes since the first mode is constant
    mode_counts = np.arange(2, args.n_modes + 1, args.mode_step, dtype=int)
    mesh, medmask = fetch_surf(density=args.density)
    task_maps = load_task_maps(args.contrast, medmask, args.n_subj)

    if args.hetero_label is not None:
        hetero_map = load_hmap(args.hetero_label, density=args.density)
    else:
        hetero_map = None
    if args.aniso_label is not None:
        aniso_map = load_hmap(args.aniso_label, density=args.density)
    else:
        aniso_map = None

    run_config = {
        "schema_version": 1,
        "objective_version": OBJECTIVE_VERSION,
        "run_id": int(run_id),
        "test_mode": bool(args.test),
        "density": args.density,
        "contrast": args.contrast,
        "n_subj": int(args.n_subj),
        "n_modes": int(args.n_modes),
        "mode_step": int(args.mode_step),
        "error_target": float(args.error_target),
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "seed": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "polish": bool(args.polish),
        "tfmri_file": str(DEFAULT_TFMRI_FILE),
        "hetero_label": args.hetero_label,
        "aniso_label": args.aniso_label,
        "anisotropy_mode": aniso_mode,
        "defaults": defaults,
        "optimization_parameters": {name: asdict(spec) for name, spec in param_specs.items()},
    }

    isotropic_meta = {
        "objective_version": OBJECTIVE_VERSION,
        "density": args.density,
        "contrast": args.contrast,
        "n_subj": int(args.n_subj),
        "n_modes": int(args.n_modes),
        "mode_step": int(args.mode_step),
        "error_target": float(args.error_target),
        "tfmri_file": str(DEFAULT_TFMRI_FILE),
    }
    isotropic = _compute_isotropic_reconstruction(
        results_dir=run_dir if args.test else results_dir,
        mesh=mesh,
        medmask=medmask,
        task_maps=task_maps,
        mode_counts=mode_counts,
        meta_base=isotropic_meta,
        error_target=float(args.error_target),
        cache_subdir=iso_cache_subdir,
    )
    run_config["isotropic_cache_key"] = isotropic["cache_key"]
    run_config["run_hash"] = _hash_key(run_config)

    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "run_hash.txt").write_text(f"{run_config['run_hash']}\n", encoding="utf-8")

    objective = ObjectiveEvaluator(
        mesh=mesh,
        medmask=medmask,
        task_maps=task_maps,
        mode_counts=mode_counts,
        hetero_map=hetero_map,
        aniso_map=aniso_map,
        param_specs=param_specs,
        fixed_params=fixed_params,
        cache_dir=cache_dir,
        eval_dir=eval_dir,
        meta_base={
            "objective_version": OBJECTIVE_VERSION,
            "run_hash": run_config["run_hash"],
            "density": args.density,
            "contrast": args.contrast,
            "n_subj": int(args.n_subj),
            "n_modes": int(args.n_modes),
            "mode_step": int(args.mode_step),
            "error_target": float(args.error_target),
            "tfmri_file": str(DEFAULT_TFMRI_FILE),
            "hetero_label": args.hetero_label,
            "aniso_label": args.aniso_label,
        },
        cache={},
        error_target=float(args.error_target),
    )

    free_bounds = [(spec.min, spec.max) for spec in param_specs.values()]
    if free_bounds:
        timing_callback = TimingCallback(param_specs)
        result = differential_evolution(
            objective,
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
        best_params = dict(fixed_params)
        for i, name in enumerate(param_specs.keys()):
            spec = param_specs[name]
            best_params[name] = _snap_to_grid(float(result.x[i]), spec.min, spec.max, spec.step)
    else:
        result = None
        best_params = dict(fixed_params)

    best_key, best_cache_path = objective._cache_key_and_path(best_params)
    if best_cache_path.exists():
        cached = np.load(best_cache_path, allow_pickle=False)
        best_objective = float(cached["objective"])
        best_auc = float(cached["auc"])
        cached_curve = np.asarray(cached["curve"], dtype=float) if "curve" in cached else np.asarray([])
        best_k_target = _cached_k_target(cached, float(args.error_target), mode_counts, cached_curve if cached_curve.size else np.zeros_like(mode_counts, dtype=float))
    else:
        best_objective = float(objective(np.array([best_params[name] for name in free_param_names], dtype=float))) if free_param_names else float(objective(np.array([], dtype=float)))
        best_auc = best_objective
        best_solver_kwargs: Dict[str, Any] = {
            "surf": mesh,
            "mask": medmask,
        }
        if best_params.get("alpha") is not None:
            best_solver_kwargs["hetero"] = hetero_map
            best_solver_kwargs["alpha"] = float(best_params["alpha"])
        if best_params.get("beta") is not None:
            best_solver_kwargs["aniso_map"] = aniso_map
            best_solver_kwargs["beta"] = float(best_params["beta"])
        elif best_params.get("aniso_curv1") is not None and best_params.get("aniso_curv2") is not None:
            best_solver_kwargs["aniso_curv"] = (float(best_params["aniso_curv1"]), float(best_params["aniso_curv2"]))
        aniso_solver_for_best = EigenSolver(**best_solver_kwargs)
        aniso_solver_for_best.solve(n_modes=int(args.n_modes))
        best_curve, best_auc, best_k_target = evaluate_solver(aniso_solver_for_best, task_maps, mode_counts, float(args.error_target))
        _atomic_savez(
            best_cache_path,
            objective=best_auc,
            auc=best_auc,
            k_target=best_k_target,
            error_target=float(args.error_target),
            curve=best_curve.astype(np.float32),
            mode_counts=mode_counts.astype(np.int32),
            **best_params,
        )

    iso_solver = EigenSolver(surf=mesh, mask=medmask)
    iso_solver.solve(n_modes=int(args.n_modes))
    iso_curve_mean, iso_curve_std, iso_auc, iso_k_target = evaluate_solver_stats(
        iso_solver,
        task_maps,
        mode_counts,
        float(args.error_target),
    )

    aniso_solver_kwargs: Dict[str, Any] = {
        "surf": mesh,
        "mask": medmask,
    }
    if best_params.get("alpha") is not None:
        aniso_solver_kwargs["hetero"] = hetero_map
        aniso_solver_kwargs["alpha"] = float(best_params["alpha"])
    if best_params.get("beta") is not None:
        aniso_solver_kwargs["aniso_map"] = aniso_map
        aniso_solver_kwargs["beta"] = float(best_params["beta"])
    elif best_params.get("aniso_curv1") is not None and best_params.get("aniso_curv2") is not None:
        aniso_solver_kwargs["aniso_curv"] = (float(best_params["aniso_curv1"]), float(best_params["aniso_curv2"]))
    aniso_solver = EigenSolver(**aniso_solver_kwargs)
    aniso_solver.solve(n_modes=int(args.n_modes))
    aniso_curve_mean, aniso_curve_std, aniso_auc, aniso_k_target = evaluate_solver_stats(
        aniso_solver,
        task_maps,
        mode_counts,
        float(args.error_target),
    )

    summary = {
        "density": args.density,
        "contrast": args.contrast,
        "n_subj": int(args.n_subj),
        "n_modes": int(args.n_modes),
        "mode_step": int(args.mode_step),
        "error_target": float(args.error_target),
        "run_id": int(run_id),
        "run_hash": run_config["run_hash"],
        "anisotropy_mode": aniso_mode,
        "hetero_label": args.hetero_label,
        "aniso_label": args.aniso_label,
        "isotropic_cache": {
            "cache_key": isotropic["cache_key"],
            "cache_path": isotropic["cache_path"],
            "auc": iso_auc,
            "k_target": iso_k_target,
        },
        "isotropic": {
            "auc": iso_auc,
            "k_target": iso_k_target,
        },
        "anisotropic": {
            "alpha": (float(best_params.get("alpha")) if best_params.get("alpha") is not None else None),
            "beta": (float(best_params.get("beta")) if best_params.get("beta") is not None else None),
            "aniso_curv1": (float(best_params.get("aniso_curv1")) if best_params.get("aniso_curv1") is not None else None),
            "aniso_curv2": (float(best_params.get("aniso_curv2")) if best_params.get("aniso_curv2") is not None else None),
            "auc": aniso_auc,
            "k_target": aniso_k_target,
            "objective": best_objective,
            "success": bool(result.success) if result is not None else True,
            "message": str(result.message) if result is not None else "evaluated fixed default parameters",
            "nit": int(result.nit) if result is not None else 0,
            "nfev": int(result.nfev) if result is not None else 1,
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "best.json").write_text(
        json.dumps(
            {
                "run_hash": run_config["run_hash"],
                "cache_key": best_key,
                "objective": best_objective,
                "auc": best_auc,
                "k_target": best_k_target,
                "error_target": float(args.error_target),
                **best_params,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    de_result = {
        "x": [float(v) for v in result.x] if result is not None else [],
        "fun": float(result.fun) if result is not None else best_objective,
        "nfev": int(result.nfev) if result is not None else 1,
        "nit": int(result.nit) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "evaluated fixed default parameters",
    }
    (run_dir / "de_result.json").write_text(json.dumps(de_result, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for p in sorted(eval_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows = sorted(rows, key=lambda d: float(d.get("objective", np.inf)))
    if rows:
        fieldnames = ["cache_key", "objective", "auc", "k_target", "error_target"] + list(free_param_names)
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with (run_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # print(json.dumps(summary, indent=2))
    print(f"Run folder: {run_dir}")

    # if args.output_json:
    #     output_path = Path(args.output_json)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     with output_path.open("w", encoding="utf-8") as f:
    #         json.dump(summary, f, indent=2)

    landscape_path = plot_auc_landscape(
        run_dir=run_dir,
        free_param_names=free_param_names,
        contrast=args.contrast,
        error_target=float(args.error_target),
    )
    curves_path = plot_best_reconstruction_curves(
        run_dir=run_dir,
        contrast=args.contrast,
        mode_counts=mode_counts,
        iso_mean=iso_curve_mean,
        iso_std=iso_curve_std,
        aniso_mean=aniso_curve_mean,
        aniso_std=aniso_curve_std,
        isotropic_auc=float(iso_auc),
        isotropic_k_target=int(iso_k_target),
        anisotropic_auc=float(aniso_auc),
        anisotropic_k_target=int(aniso_k_target),
        error_target=float(args.error_target),
    )

    if args.plot:
        if landscape_path is not None:
            landscape_img = plt.imread(str(landscape_path))
            fig_land, ax_land = plt.subplots(figsize=(8, 6))
            ax_land.imshow(landscape_img)
            ax_land.axis("off")
            fig_land.tight_layout()

        curves_img = plt.imread(str(curves_path))
        fig_curv, ax_curv = plt.subplots(figsize=(8, 6))
        ax_curv.imshow(curves_img)
        ax_curv.axis("off")
        fig_curv.tight_layout()

    if landscape_path is not None:
        print(f"Saved landscape figure: {landscape_path}")
    print(f"Saved reconstruction curve figure: {curves_path}")

    print(f"Total optimisation time: {(time.time() - t1)/60:.2f}min")

if __name__ == "__main__":
    main()