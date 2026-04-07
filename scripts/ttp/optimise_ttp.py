from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from pathlib import Path

import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import reduce_by_labels
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from heteromodes.utils import get_project_root, load_hmap
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf


PROJ_DIR = get_project_root()

# Update this if the objective function changes in a non-backwards-compatible way
OBJECTIVE_VERSION = "ttp_spearman_abs_v1"

def _hash_key(d: Dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


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

class TimingCallback:
    def __init__(self, param_specs: Dict[str, GridSpec]) -> None:
        """Initialize with only the parameters being optimized.
        
        Parameters
        ----------
        param_specs : Dict[str, GridSpec]
            Dictionary mapping parameter name to GridSpec (e.g. {"alpha": GridSpec(...), "beta": GridSpec(...)})
        """
        self.param_specs = param_specs
        self.param_names = list(param_specs.keys())
        self.iteration_times: list[float] = []

    def __call__(self, xk: np.ndarray, convergence: float) -> None:
        self.iteration_times.append(time.time())
        param_vals = {}
        for i, name in enumerate(self.param_names):
            spec = self.param_specs[name]
            param_vals[name] = _snap_to_grid(float(xk[i]), spec.min, spec.max, spec.step)
        
        param_str = ", ".join([f"{name}={val:.2f}" for name, val in param_vals.items()])
        if len(self.iteration_times) > 1:
            elapsed = self.iteration_times[-1] - self.iteration_times[-2]
            print(f"  Iteration {len(self.iteration_times)}: {elapsed/60:.3f}min | {param_str}, convergence={convergence:.4f}")
        else:
            print(f"  Iteration 1 (initial): {param_str}, convergence={convergence:.4f}")


class ObjectiveEvaluator:
    def __init__(
        self,
        *,
        mesh: Any,
        medmask: np.ndarray,
        parc_labels_masked: np.ndarray,
        vis_hierarchy_roi_labels: np.ndarray,
        proxy_hierarchy: np.ndarray,
        hetero_map: np.ndarray,
        aniso_map: np.ndarray,
        param_specs: Dict[str, GridSpec],
        n_modes: int,
        dt: float,
        nt: int,
        r: float,
        gamma: float,
        ext_input: np.ndarray,
        cache_dir: Path,
        eval_dir: Path,
        meta_base: Dict[str, Any],
    ) -> None:
        self.mesh = mesh
        self.medmask = medmask
        self.parc_labels_masked = parc_labels_masked
        self.vis_hierarchy_roi_labels = vis_hierarchy_roi_labels
        self.proxy_hierarchy = proxy_hierarchy
        self.hetero_map = hetero_map
        self.aniso_map = aniso_map
        self.param_specs = param_specs
        self.param_names = list(param_specs.keys())
        self.n_modes = int(n_modes)
        self.dt = float(dt)
        self.nt = int(nt)
        self.r = float(r)
        self.gamma = float(gamma)
        self.ext_input = ext_input
        self.cache_dir = cache_dir
        self.eval_dir = eval_dir
        self.meta_base = meta_base

    def _cache_key_and_path(self, param_values: Dict[str, float]) -> Tuple[str, Path]:
        meta = dict(self.meta_base)
        meta.update(param_values)
        key = _hash_key(meta)
        return key, (self.cache_dir / f"eval_{key}.npz")

    def __call__(self, x: np.ndarray) -> float:
        param_values = {}
        for i, name in enumerate(self.param_names):
            spec = self.param_specs[name]
            param_values[name] = _snap_to_grid(float(x[i]), spec.min, spec.max, spec.step)
        
        cache_key, cache_path = self._cache_key_and_path(param_values)

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            abs_rho = float(cached["abs_rho"])
            json_data = dict(param_values)
            json_data["cache_key"] = cache_key
            json_data["abs_rho"] = abs_rho
            _safe_write_json_once(self.eval_dir / f"{cache_key}.json", json_data)
            return -abs_rho

        try:
            solver_kwargs = {"surf": self.mesh, "mask": self.medmask}
            
            if "alpha" in param_values:
                alpha = param_values.get("alpha", 0)
                solver_kwargs["hetero"] = self.hetero_map if alpha != 0 else None
                solver_kwargs["alpha"] = alpha if alpha != 0 else None
            
            if "beta" in param_values:
                beta = param_values.get("beta", 0)
                solver_kwargs["aniso_map"] = self.aniso_map if beta != 0 else None
                solver_kwargs["beta"] = beta if beta != 0 else None
            
            if "aniso_curv1" in param_values or "aniso_curv2" in param_values:
                aniso_curv1 = param_values.get("aniso_curv1", 0)
                aniso_curv2 = param_values.get("aniso_curv2", 0)
                if aniso_curv1 != 0 or aniso_curv2 != 0:
                    solver_kwargs["aniso_curv"] = (aniso_curv1, aniso_curv2)
            
            solver = EigenSolver(**solver_kwargs).solve(n_modes=self.n_modes)
            neural = solver.simulate_waves(
                ext_input=self.ext_input,
                nt=self.nt,
                dt=self.dt,
                r=self.r,
                gamma=self.gamma,
            )

            neural_parc = reduce_by_labels(neural, self.parc_labels_masked, axis=1)
            neural_hierarchy = neural_parc[self.vis_hierarchy_roi_labels - 1, :]
            ttp_hierarchy = np.argmax(neural_hierarchy, axis=1) * self.dt
            rho = float(spearmanr(self.proxy_hierarchy, ttp_hierarchy).correlation)

            if not np.isfinite(rho):
                param_str = ", ".join([f"{name}={val:.4f}" for name, val in param_values.items()])
                print(f"  WARNING: Non-finite rho at {param_str}. Assigning large penalty.")
                return 1e6

            abs_rho = float(abs(rho))
            meta = dict(self.meta_base)
            meta.update(param_values)
            meta.update({"cache_key": cache_key})
            save_arrays = {
                "rho": rho,
                "abs_rho": abs_rho,
                "ttp_hierarchy": np.asarray(ttp_hierarchy, dtype=np.float32),
                "meta_json": json.dumps(meta, sort_keys=True),
            }
            save_arrays.update(param_values)
            _atomic_savez(cache_path, **save_arrays)
            
            json_data = dict(param_values)
            json_data["cache_key"] = cache_key
            json_data["abs_rho"] = abs_rho
            _safe_write_json_once(self.eval_dir / f"{cache_key}.json", json_data)
            return -abs_rho
        except Exception as e:
            param_str = ", ".join([f"{name}={val:.4f}" for name, val in param_values.items()])
            print(f"  ERROR at {param_str}: {type(e).__name__}: {e}")
            return 1e6


def plot_optimization_landscape(run_dir: Path, save_path: Path | None = None) -> None:
    """
    Plot optimization landscape from manifest.csv.
    Handles 2D (alpha vs beta) or 3D (alpha vs beta vs aniso_curv1) plots.
    
    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing manifest.csv
    save_path : Path, optional
        If provided, save the figure here instead of displaying
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    manifest_path = Path(run_dir) / "manifest.csv"
    if not manifest_path.exists():
        print(f"manifest.csv not found in {run_dir}")
        return
    
    df = pd.read_csv(manifest_path)
    
    # Determine which parameters we have
    params = [col for col in df.columns if col not in ["cache_key", "abs_rho"]]
    
    if len(params) == 2:
        # 2D plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            df[params[0]],
            df[params[1]],
            c=df["abs_rho"],
            cmap="turbo",
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5
        )
        ax.set_xlabel(params[0].capitalize(), fontsize=12)
        ax.set_ylabel(params[1].capitalize(), fontsize=12)
        
        # Mark the best point
        best_idx = df["abs_rho"].idxmax()
        best_x = df.loc[best_idx, params[0]]
        best_y = df.loc[best_idx, params[1]]
        best_rho = df.loc[best_idx, "abs_rho"]
        ax.plot(best_x, best_y, "r*", markersize=20, label=f"Best: rho={best_rho:.4f}")
        
    elif len(params) >= 3:
        # 3D plot (or higher dimensional - just use first 3)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            df[params[0]],
            df[params[1]],
            df[params[2]],
            c=df["abs_rho"],
            cmap="turbo",
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5
        )
        ax.set_xlabel(params[0].capitalize(), fontsize=11)
        ax.set_ylabel(params[1].capitalize(), fontsize=11)
        ax.set_zlabel(params[2].capitalize(), fontsize=11)
        
        # Mark the best point
        best_idx = df["abs_rho"].idxmax()
        best_x = df.loc[best_idx, params[0]]
        best_y = df.loc[best_idx, params[1]]
        best_z = df.loc[best_idx, params[2]]
        best_rho = df.loc[best_idx, "abs_rho"]
        ax.scatter([best_x], [best_y], [best_z], color="red", s=500, marker="*", label=f"Best: rho={best_rho:.4f}")
    else:
        print("Insufficient parameters in manifest.csv for plotting")
        return
    
    ax.set_title(f"Optimization Landscape (Run {Path(run_dir).name})", fontsize=14)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("|Spearman rho|", fontsize=12)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimise TTP alpha/beta to maximise |Spearman rho|.")
    p.add_argument("--test", action="store_true", help="Run in test mode using folder 0 and _cache_test.")
    p.add_argument("--hetero_label", type=str, default="SAaxis", help="Heterogeneity map label.")
    p.add_argument("--aniso_label", type=str, default="SAaxis", help="Anisotropy map label")
    p.add_argument("--density", type=str, default="32k", help="Surface density for mesh and parcellation.")
    p.add_argument("--n_modes", type=int, default=1000, help="Number of eigenmodes.")
    p.add_argument("--alpha", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Alpha optimization range: MIN MAX STEP")
    p.add_argument("--beta", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Beta optimization range: MIN MAX STEP")
    p.add_argument("--aniso_curv1", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Aniso_curv1 optimization range: MIN MAX STEP")
    p.add_argument("--aniso_curv2", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Aniso_curv2 optimization range: MIN MAX STEP")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers for DE (-1 = all).")
    p.add_argument("--maxiter", type=int, default=50, help="DE max iterations.")
    p.add_argument("--popsize", type=int, default=16, help="DE population size multiplier.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--polish", action="store_true", help="Enable DE polish step (usually off).")
    return p.parse_args()


def main() -> None:
    t1 = time.time()
    args = parse_args()

    results_dir = Path(PROJ_DIR) / "results" / "human" / "ttp"
    if args.test:
        run_id = 0
        run_dir = results_dir / "0"
        cache_dir = run_dir / "_cache_test"
        if run_dir.exists():
            shutil.rmtree(run_dir)
    else:
        run_id = _next_run_id(results_dir)
        run_dir = results_dir / str(run_id)
        cache_dir = results_dir / "_cache"
        while True:
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                run_id = _next_run_id(results_dir)
                run_dir = results_dir / str(run_id)

    eval_dir = run_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter specs dict with only the parameters provided
    param_specs = {}
    bounds = []
    
    if args.alpha is not None:
        alpha_spec = _parse_grid3(tuple(args.alpha), "alpha")
        print(f"Optimising with alpha in [{alpha_spec.min}, {alpha_spec.max}] step {alpha_spec.step}")
        param_specs["alpha"] = alpha_spec
        bounds.append((alpha_spec.min, alpha_spec.max))
    else:
        print("Not optimising alpha (using default: 0)")
    
    if args.beta is not None:
        beta_spec = _parse_grid3(tuple(args.beta), "beta")
        print(f"Optimising with beta in [{beta_spec.min}, {beta_spec.max}] step {beta_spec.step}")
        param_specs["beta"] = beta_spec
        bounds.append((beta_spec.min, beta_spec.max))
    else:
        print("Not optimising beta (using default: 0)")
    
    if args.aniso_curv1 is not None:
        aniso_curv1_spec = _parse_grid3(tuple(args.aniso_curv1), "aniso_curv1")
        print(f"Optimising with aniso_curv1 in [{aniso_curv1_spec.min}, {aniso_curv1_spec.max}] step {aniso_curv1_spec.step}")
        param_specs["aniso_curv1"] = aniso_curv1_spec
        bounds.append((aniso_curv1_spec.min, aniso_curv1_spec.max))
    else:
        print("Not optimising aniso_curv1 (using default: 0)")
    
    if args.aniso_curv2 is not None:
        aniso_curv2_spec = _parse_grid3(tuple(args.aniso_curv2), "aniso_curv2")
        print(f"Optimising with aniso_curv2 in [{aniso_curv2_spec.min}, {aniso_curv2_spec.max}] step {aniso_curv2_spec.step}")
        param_specs["aniso_curv2"] = aniso_curv2_spec
        bounds.append((aniso_curv2_spec.min, aniso_curv2_spec.max))
    else:
        print("Not optimising aniso_curv2 (using default: 0)")
    
    if not param_specs:
        raise ValueError("At least one parameter must be provided for optimization (--alpha, --beta, --aniso_curv1, or --aniso_curv2)")
    
    print(f"\nTotal optimization dimensions: {len(param_specs)}")

    # Simulation constants (keep simple; edit here if needed)
    dt = 1e-4   # seconds
    nt = 1000
    r = 28.9        # mm
    gamma = 116     # seconds^-1
    stimulation_amplitude = 20.0
    stim_start = 10
    stim_stop = 20

    mesh, medmask = fetch_surf(density=args.density)

    parc_path = Path(PROJ_DIR) / "data" / "parcellations" / f"parc-hcpmmp1_space-fsLR_den-{args.density}_hemi-L.label.gii"
    parc = nib.load(str(parc_path))
    parc_labels = parc.darrays[0].data.astype(int)
    parc_labels_masked = parc_labels[medmask]

    label_to_key = {lab.label: lab.key for lab in parc.labeltable.labels}
    v1_mask = parc_labels == label_to_key["L_V1_ROI"]

    vis_hierarchy_path = Path(PROJ_DIR) / "data" / "parcellations" / "17_visual_cortical_hierarchy_rois.npy"
    vis_hierarchy_roi_labels = np.load(vis_hierarchy_path).astype(int)

    hetero_label = args.hetero_label
    aniso_label = args.aniso_label if args.aniso_label is not None else hetero_label
    hetero_map = load_hmap(hetero_label, density=args.density)
    aniso_map = load_hmap(aniso_label, density=args.density)

    # Proxy hierarchy: myelinmap parcellated on the same masked labels (so indices are label_key-1)
    myelin = load_hmap("myelinmap", density=args.density)
    myelin_parc = reduce_by_labels(myelin[medmask], parc_labels_masked, axis=0)
    proxy_hierarchy = myelin_parc[vis_hierarchy_roi_labels - 1]

    # External input (vertex x time) in masked space
    ext_input = np.zeros((mesh.vertices.shape[0], nt), dtype=np.float32)
    ext_input[v1_mask, stim_start:stim_stop] = float(stimulation_amplitude)
    ext_input = ext_input[medmask, :]

    config = {
        "schema_version": 1,    # Increment if config structure changes in a non-backwards-compatible way
        "objective_version": OBJECTIVE_VERSION, # Increment if objective function changes in a non-backwards-compatible way
        "run_id": run_id,
        "test_mode": bool(args.test),
        "hetero_label": hetero_label,
        "aniso_label": aniso_label,
        "n_modes": int(args.n_modes),
        "n_jobs": int(args.n_jobs),
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "seed": int(args.seed),
        "polish": bool(args.polish),
        "dt": float(dt),
        "nt": int(nt),
        "r": float(r),
        "gamma": float(gamma),
        "stimulation_amplitude": float(stimulation_amplitude),
        "stim_start": int(stim_start),
        "stim_stop": int(stim_stop),
        "density": args.density,
    }
    for name, spec in param_specs.items():
        config[name] = asdict(spec)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    meta_base = {
        "objective_version": OBJECTIVE_VERSION,
        "hetero_label": hetero_label,
        "aniso_label": aniso_label,
        "n_modes": int(args.n_modes),
        "dt": float(dt),
        "nt": int(nt),
        "r": float(r),
        "gamma": float(gamma),
        "stimulation_amplitude": float(stimulation_amplitude),
        "stim_start": int(stim_start),
        "stim_stop": int(stim_stop),
        "density": args.density,
    }

    evaluator = ObjectiveEvaluator(
        mesh=mesh,
        medmask=medmask,
        parc_labels_masked=parc_labels_masked,
        vis_hierarchy_roi_labels=vis_hierarchy_roi_labels,
        proxy_hierarchy=proxy_hierarchy,
        hetero_map=hetero_map,
        aniso_map=aniso_map,
        param_specs=param_specs,
        n_modes=int(args.n_modes),
        dt=dt,
        nt=nt,
        r=r,
        gamma=gamma,
        ext_input=ext_input,
        cache_dir=cache_dir,
        eval_dir=eval_dir,
        meta_base=meta_base,
    )

    # Use a custom callback to track iteration times and print progress
    timing_callback = TimingCallback(param_specs)
    
    res = differential_evolution(
        evaluator,
        bounds=bounds,
        maxiter=int(args.maxiter),
        popsize=int(args.popsize),
        seed=int(args.seed),
        workers=int(args.n_jobs),
        updating="deferred",
        polish=bool(args.polish),
        callback=timing_callback,
        disp=True,
    )

    best_params = {}
    for i, name in enumerate(param_specs.keys()):
        spec = param_specs[name]
        best_params[name] = _snap_to_grid(float(res.x[i]), spec.min, spec.max, spec.step)
    
    best_key, best_cache_path = evaluator._cache_key_and_path(best_params)
    cached = np.load(best_cache_path, allow_pickle=False)

    best = dict(best_params)
    best["rho"] = float(cached["rho"])
    best["abs_rho"] = float(cached["abs_rho"])
    best["cache_key"] = best_key
    (run_dir / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

    de_result = {
        "x": [float(v) for v in res.x],
        "fun": float(res.fun),
        "nfev": int(res.nfev),
        "nit": int(res.nit),
        "success": bool(res.success),
        "message": str(res.message),
    }
    (run_dir / "de_result.json").write_text(
        json.dumps(de_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Build manifest from per-eval JSON breadcrumbs.
    rows = []
    for p in sorted(eval_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows = sorted(rows, key=lambda d: float(d.get("abs_rho", -np.inf)), reverse=True)
    if rows:
        import csv
        
        # Build fieldnames from the current optimization parameters plus any
        # extra keys already present in older eval JSON rows.
        fieldnames = ["cache_key"] + list(param_specs.keys()) + ["abs_rho"]
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with (run_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    param_str = ", ".join([f"{name}={val:.4f}" for name, val in best_params.items()])
    print(f"\nBest |rho| = {best['abs_rho']:.4f} at {param_str}")
    print(f"Run folder: {run_dir}")
    print(f"Total optimisation time: {(time.time() - t1)/60:.2f}min")
    
    # Generate landscape plot
    plot_optimization_landscape(run_dir, save_path=run_dir / "landscape.png")


if __name__ == "__main__":
    main()

