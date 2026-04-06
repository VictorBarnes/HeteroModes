from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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


def _snap_to_grid(x: float, lo: float, hi: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    k = round((x - lo) / step)
    snapped = lo + k * step
    return float(np.clip(snapped, lo, hi))


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
    lo: float
    hi: float
    step: float


def _parse_grid3(values: Tuple[float, float, float], name: str) -> GridSpec:
    lo, hi, step = [float(v) for v in values]
    if hi < lo:
        lo, hi = hi, lo
    if step <= 0:
        raise ValueError(f"{name} step must be > 0")
    return GridSpec(lo=lo, hi=hi, step=step)

class TimingCallback:
    def __init__(self, alpha_spec: GridSpec, beta_spec: GridSpec) -> None:
        self.alpha_spec = alpha_spec
        self.beta_spec = beta_spec
        self.iteration_times: list[float] = []

    def __call__(self, xk: np.ndarray, convergence: float) -> None:
        self.iteration_times.append(time.time())
        alpha_val = _snap_to_grid(float(xk[0]), self.alpha_spec.lo, self.alpha_spec.hi, self.alpha_spec.step)
        beta_val = _snap_to_grid(float(xk[1]), self.beta_spec.lo, self.beta_spec.hi, self.beta_spec.step)
        if len(self.iteration_times) > 1:
            elapsed = self.iteration_times[-1] - self.iteration_times[-2]
            print(f"  Iteration {len(self.iteration_times)}: {elapsed/60:.3f}min | alpha={alpha_val:.2f}, beta={beta_val:.2f}, convergence={convergence:.4f}")
        else:
            print(f"  Iteration 1 (initial): alpha={alpha_val:.2f}, beta={beta_val:.2f}, convergence={convergence:.4f}")


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
        alpha_spec: GridSpec,
        beta_spec: GridSpec,
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
        self.alpha_spec = alpha_spec
        self.beta_spec = beta_spec
        self.n_modes = int(n_modes)
        self.dt = float(dt)
        self.nt = int(nt)
        self.r = float(r)
        self.gamma = float(gamma)
        self.ext_input = ext_input
        self.cache_dir = cache_dir
        self.eval_dir = eval_dir
        self.meta_base = meta_base

    def _cache_key_and_path(self, alpha: float, beta: float) -> Tuple[str, Path]:
        meta = dict(self.meta_base)
        meta.update({"alpha": float(alpha), "beta": float(beta)})
        key = _hash_key(meta)
        return key, (self.cache_dir / f"eval_{key}.npz")

    def __call__(self, x: np.ndarray) -> float:
        alpha = _snap_to_grid(float(x[0]), self.alpha_spec.lo, self.alpha_spec.hi, self.alpha_spec.step)
        beta = _snap_to_grid(float(x[1]), self.beta_spec.lo, self.beta_spec.hi, self.beta_spec.step)
        cache_key, cache_path = self._cache_key_and_path(alpha, beta)

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            abs_rho = float(cached["abs_rho"])
            _safe_write_json_once(
                self.eval_dir / f"{cache_key}.json",
                {"cache_key": cache_key, "alpha": alpha, "beta": beta, "abs_rho": abs_rho},
            )
            return -abs_rho

        try:
            solver = EigenSolver(
                surf=self.mesh,
                mask=self.medmask,
                hetero=self.hetero_map if alpha != 0 else None,
                alpha=alpha if alpha != 0 else None,
                aniso_map=self.aniso_map if beta != 0 else None,
                beta=beta if beta != 0 else None,
            ).solve(n_modes=self.n_modes)
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
                print(f"  WARNING: Non-finite rho at alpha={alpha:.4f}, beta={beta:.4f}. Assigning large penalty.")
                return 1e6

            abs_rho = float(abs(rho))
            meta = dict(self.meta_base)
            meta.update({"alpha": alpha, "beta": beta, "cache_key": cache_key})
            _atomic_savez(
                cache_path,
                rho=rho,
                abs_rho=abs_rho,
                alpha=alpha,
                beta=beta,
                ttp_hierarchy=np.asarray(ttp_hierarchy, dtype=np.float32),
                meta_json=json.dumps(meta, sort_keys=True),
            )
            _safe_write_json_once(
                self.eval_dir / f"{cache_key}.json",
                {"cache_key": cache_key, "alpha": alpha, "beta": beta, "abs_rho": abs_rho},
            )
            return -abs_rho
        except Exception as e:
            print(f"  ERROR at alpha={alpha:.4f}, beta={beta:.4f}: {type(e).__name__}: {e}")
            return 1e6


def plot_optimization_landscape(run_dir: Path, save_path: Path | None = None) -> None:
    """
    Plot alpha vs beta colored by abs_rho from manifest.csv.
    
    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing manifest.csv
    save_path : Path, optional
        If provided, save the figure here instead of displaying
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import pandas as pd
    
    manifest_path = Path(run_dir) / "manifest.csv"
    if not manifest_path.exists():
        print(f"manifest.csv not found in {run_dir}")
        return
    
    df = pd.read_csv(manifest_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with alpha on x, beta on y, color = abs_rho
    scatter = ax.scatter(
        df["alpha"],
        df["beta"],
        c=df["abs_rho"],
        cmap="turbo",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5
    )
    
    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel("Beta", fontsize=12)
    ax.set_title(f"Optimization Landscape (Run {Path(run_dir).name})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("|Spearman rho|", fontsize=12)
    
    # Mark the best point
    best_idx = df["abs_rho"].idxmax()
    best_alpha = df.loc[best_idx, "alpha"]
    best_beta = df.loc[best_idx, "beta"]
    best_rho = df.loc[best_idx, "abs_rho"]
    
    ax.plot(best_alpha, best_beta, "r*", markersize=20, label=f"Best: rho={best_rho:.4f}")
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimise TTP alpha/beta to maximise |Spearman rho|.")
    p.add_argument("--id", type=int, default=0, help="Run ID (integer). Default: 0.")
    p.add_argument("--hetero_label", type=str, default="SAaxis", help="Heterogeneity map label.")
    p.add_argument("--aniso_label", type=str, default="SAaxis", help="Anisotropy map label")
    p.add_argument("--density", type=str, default="32k", help="Surface density for mesh and parcellation.")
    p.add_argument("--n_modes", type=int, default=1000, help="Number of eigenmodes.")
    p.add_argument("--alpha", type=float, nargs=3, required=True, metavar=("MIN", "MAX", "STEP"))
    p.add_argument("--beta", type=float, nargs=3, required=True, metavar=("MIN", "MAX", "STEP"))
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers for DE (-1 = all).")
    p.add_argument("--maxiter", type=int, default=50, help="DE max iterations.")
    p.add_argument("--popsize", type=int, default=16, help="DE population size multiplier.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--polish", action="store_true", help="Enable DE polish step (usually off).")
    return p.parse_args()


def main() -> None:
    t1 = time.time()
    args = parse_args()

    run_id = str(args.id)
    results_dir = Path(PROJ_DIR) / "results" / "ttp"
    run_dir = results_dir / run_id
    cache_dir = results_dir / "_cache"
    eval_dir = run_dir / "evals"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    alpha_spec = _parse_grid3(tuple(args.alpha), "alpha")
    print(f"Optimising with alpha in [{alpha_spec.lo}, {alpha_spec.hi}] step {alpha_spec.step}")
    beta_spec = _parse_grid3(tuple(args.beta), "beta")
    print(f"Optimising with beta in [{beta_spec.lo}, {beta_spec.hi}] step {beta_spec.step}")

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
        "hetero_label": hetero_label,
        "aniso_label": aniso_label,
        "n_modes": int(args.n_modes),
        "alpha": asdict(alpha_spec),
        "beta": asdict(beta_spec),
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
        alpha_spec=alpha_spec,
        beta_spec=beta_spec,
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

    bounds = [(alpha_spec.lo, alpha_spec.hi), (beta_spec.lo, beta_spec.hi)]
    
    # Use a custom callback to track iteration times and print progress
    timing_callback = TimingCallback(alpha_spec, beta_spec)
    
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

    best_alpha = _snap_to_grid(float(res.x[0]), alpha_spec.lo, alpha_spec.hi, alpha_spec.step)
    best_beta = _snap_to_grid(float(res.x[1]), beta_spec.lo, beta_spec.hi, beta_spec.step)
    best_key, best_cache_path = evaluator._cache_key_and_path(best_alpha, best_beta)
    cached = np.load(best_cache_path, allow_pickle=False)

    best = {
        "alpha": best_alpha,
        "beta": best_beta,
        "rho": float(cached["rho"]),
        "abs_rho": float(cached["abs_rho"]),
        "cache_key": best_key,
    }
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

        with (run_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cache_key", "alpha", "beta", "abs_rho"])
            w.writeheader()
            w.writerows(rows)

    print(f"\nBest |rho| = {best['abs_rho']:.4f} at alpha={best_alpha}, beta={best_beta}")
    print(f"Run folder: {run_dir}")
    print(f"Total optimisation time: {(time.time() - t1)/60:.2f}min")
    
    # Generate landscape plot
    plot_optimization_landscape(run_dir, save_path=run_dir / "landscape.png")


if __name__ == "__main__":
    main()

