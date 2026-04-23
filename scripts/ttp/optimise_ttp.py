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

import nibabel as nib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from brainspace.utils.parcellation import reduce_by_labels
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr, rankdata

from heteromodes.utils import get_project_root, load_hmap
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf
from nsbutils.utils import unmask
from nsbutils.plotting_pyvista import plot_surf, plot_surf_video


PROJ_DIR = get_project_root()

# Update this if the objective function changes in a non-backwards-compatible way
OBJECTIVE_VERSION = "ttp_spearman_abs_v1"
DEFAULT_R = 28.9
DEFAULT_GAMMA = 116.0

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


def _next_non_test_run_id(results_dir: Path) -> int:
    run_ids = []
    if results_dir.exists():
        for child in results_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                run_id = int(child.name)
                if run_id > 0:
                    run_ids.append(run_id)
    return (max(run_ids) + 1) if run_ids else 1


def _normalize_config_for_id_check(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
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
    normalized.pop("active_parameter_names", None)
    return normalized


def _collect_config_mismatches(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    mismatches: list[str] = []
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


def _validate_pair_component(label: str | None, arg_name: str) -> str:
    token = str(label)
    if token in {".", ".."}:
        raise ValueError(f"--{arg_name} cannot be '.' or '..'")
    path_obj = Path(token)
    if path_obj.is_absolute() or len(path_obj.parts) != 1:
        raise ValueError(f"--{arg_name} must be a single folder-safe name (no path separators)")
    return token

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
        fixed_params: Dict[str, float],
        n_modes: int,
        dt: float,
        nt: int,
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
        self.fixed_params = dict(fixed_params)
        self.n_modes = int(n_modes)
        self.dt = float(dt)
        self.nt = int(nt)
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

        resolved_params = dict(self.fixed_params)
        resolved_params.update(param_values)

        cache_key, cache_path = self._cache_key_and_path(resolved_params)

        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if "meta_json" not in cached.files:
                    raise ValueError("Missing meta_json in cache")

                cached_meta = json.loads(str(cached["meta_json"]))
                expected_meta = dict(self.meta_base)
                expected_meta.update(resolved_params)
                expected_meta.update({"cache_key": cache_key})

                # Validate cache metadata before reuse; mismatch means stale cache.
                for key, expected in expected_meta.items():
                    if key not in cached_meta:
                        raise ValueError(f"Missing '{key}' in cached metadata")

                    got = cached_meta[key]
                    if isinstance(expected, float):
                        if abs(float(got) - expected) > 1e-6:
                            raise ValueError(
                                f"Metadata mismatch for '{key}': got {got}, expected {expected}"
                            )
                    else:
                        if got != expected:
                            raise ValueError(
                                f"Metadata mismatch for '{key}': got {got}, expected {expected}"
                            )

                abs_rho = float(cached["abs_rho"])
                json_data = dict(resolved_params)
                json_data["cache_key"] = cache_key
                json_data["abs_rho"] = abs_rho
                _safe_write_json_once(self.eval_dir / f"{cache_key}.json", json_data)
                return -abs_rho
            except Exception as e:
                print(f"  WARNING: Ignoring stale/invalid cache {cache_path.name}: {e}")

        try:
            solver_kwargs = {"surf": self.mesh, "mask": self.medmask}
            
            if "alpha" in resolved_params:
                alpha = resolved_params.get("alpha", 0)
                solver_kwargs["hetero"] = self.hetero_map if alpha != 0 else None
                solver_kwargs["alpha"] = alpha if alpha != 0 else None
            
            if "beta" in resolved_params:
                beta = resolved_params.get("beta", 0)
                solver_kwargs["aniso_map"] = self.aniso_map if beta != 0 else None
                solver_kwargs["beta"] = beta if beta != 0 else None
            
            if "aniso_curv1" in resolved_params or "aniso_curv2" in resolved_params:
                aniso_curv1 = resolved_params.get("aniso_curv1", 0)
                aniso_curv2 = resolved_params.get("aniso_curv2", 0)
                if aniso_curv1 != 0 or aniso_curv2 != 0:
                    solver_kwargs["aniso_curv"] = (aniso_curv1, aniso_curv2)

            if "r" not in resolved_params or "gamma" not in resolved_params:
                raise ValueError("Resolved parameters must include r and gamma")
            
            solver = EigenSolver(**solver_kwargs).solve(n_modes=self.n_modes)
            neural = solver.simulate_waves(
                ext_input=self.ext_input,
                nt=self.nt,
                dt=self.dt,
                r=float(resolved_params["r"]),
                gamma=float(resolved_params["gamma"]),
            )

            neural_parc = reduce_by_labels(neural, self.parc_labels_masked, axis=1)
            neural_hierarchy = neural_parc[self.vis_hierarchy_roi_labels, :]
            ttp_hierarchy = np.argmax(neural_hierarchy, axis=1) * self.dt
            rho = float(spearmanr(self.proxy_hierarchy, ttp_hierarchy).correlation)

            if not np.isfinite(rho):
                param_str = ", ".join([f"{name}={val:.4f}" for name, val in resolved_params.items()])
                print(f"  WARNING: Non-finite rho at {param_str}. Assigning large penalty.")
                return 1e6

            abs_rho = float(abs(rho))
            meta = dict(self.meta_base)
            meta.update(resolved_params)
            meta.update({"cache_key": cache_key})
            save_arrays = {
                "rho": rho,
                "abs_rho": abs_rho,
                "neural_hierarchy": np.asarray(neural_hierarchy, dtype=np.float32),
                "ttp_hierarchy": np.asarray(ttp_hierarchy, dtype=np.float32),
                "meta_json": json.dumps(meta, sort_keys=True),
            }
            save_arrays.update(resolved_params)
            _atomic_savez(cache_path, **save_arrays)
            
            json_data = dict(resolved_params)
            json_data["cache_key"] = cache_key
            json_data["abs_rho"] = abs_rho
            _safe_write_json_once(self.eval_dir / f"{cache_key}.json", json_data)
            return -abs_rho
        except Exception as e:
            param_str = ", ".join([f"{name}={val:.4f}" for name, val in resolved_params.items()])
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_best_summary(
    cache_path: Path,
    run_dir: Path,
    proxy_hierarchy: np.ndarray,
    vis_hierarchy_roi_labels: np.ndarray,
    parc_labels: np.ndarray,
    parc_labels_masked: np.ndarray,
    mesh: Any,
    medmask: np.ndarray,
    dt: float
) -> None:
    """
    Load the best cached result and create a summary plot.
    
    Parameters
    ----------
    cache_path : Path
        Path to the best model's .npz cache file
    run_dir : Path
        Output directory for the saved plot
    proxy_hierarchy : np.ndarray
        T1w/T2w values for the 17 visual hierarchy ROIs (from myelin_parc)
    vis_hierarchy_roi_labels : np.ndarray
        Parcel indices for the 17 visual hierarchy ROIs in parcel-space (label_key-1)
    parc_labels : np.ndarray
        Full-space parcel labels from the cortical parcellation
    parc_labels_masked : np.ndarray
        Masked-space parcel labels (medial wall removed)
    mesh : Any
        Cortical surface mesh object with vertices and faces
    medmask : np.ndarray
        Binary mask for medial wall exclusion
    dt : float
        Simulation timestep in seconds
    """
    
    # Load cached data
    cached = np.load(cache_path, allow_pickle=False)
    neural_hierarchy = cached["neural_hierarchy"]
    ttp_hierarchy = cached["ttp_hierarchy"]
    rho = float(cached["rho"])
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left panel: activity over time
    ttp_ms = np.asarray(ttp_hierarchy) * 1000.0
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(ttp_ms)), vmax=float(np.nanmax(ttp_ms)))
    cmap = plt.get_cmap("turbo")
    
    time_ms = np.arange(neural_hierarchy.shape[1]) * dt * 1000
    
    for k in range(neural_hierarchy.shape[0]):
        axs[0].plot(time_ms, neural_hierarchy[k, :], color=cmap(norm(ttp_ms[k])), linewidth=1.5, alpha=0.8)
    
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    _ = fig.colorbar(sm, ax=axs[0], label="Time to peak (ms)")
    
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Neural activity")
    axs[0].set_title("Activity over time for 17 visual hierarchy ROIs")
    
    # Right panel: rank-rank scatter
    x = np.asarray(proxy_hierarchy)
    y = np.asarray(ttp_hierarchy)
    
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    
    point_colors = cmap(norm(ttp_ms))
    axs[1].scatter(x_rank, y_rank, s=45, c=point_colors, edgecolors="black", linewidth=0.3)
    axs[1].set_xlabel("T1w/T2w (rank)")
    axs[1].set_ylabel("Time to peak (rank)")
    axs[1].set_title(f"Spearman rho = {rho:.4f}")
    
    plt.tight_layout()
    save_path = run_dir / "best_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Separate parcel-wise brain figure colored by parcel TTP values.
    n_parcels = int(np.max(parc_labels_masked))
    ttp_parcel_values = np.full(n_parcels, np.nan, dtype=np.float32)
    ttp_parcel_values[np.asarray(vis_hierarchy_roi_labels, dtype=int)] = np.asarray(ttp_hierarchy, dtype=np.float32)

    valid = (parc_labels_masked > 0) & (parc_labels_masked <= n_parcels)
    ttp_vertex_masked = np.full(parc_labels_masked.shape, np.nan, dtype=np.float32)
    ttp_vertex_masked[valid] = ttp_parcel_values[parc_labels_masked[valid] - 1]
    ttp_vertex_full = unmask(ttp_vertex_masked, medmask)

    fig_brain, ax_brain = plt.subplots(figsize=(7, 4.5))
    mesh_plot = {"lh": {"v": mesh.vertices, "t": mesh.faces}}
    plot_surf(
        surf=mesh_plot,
        data={"lh": ttp_vertex_full},
        rois={"lh": parc_labels},
        views=["lateral", "medial"],
        cmap="turbo",
        cbar=True,
        roi_outlines=True,
        ax=ax_brain,
    )
    ax_brain.set_title("Parcel time-to-peak (s)")
    fig_brain.tight_layout()
    fig_brain.savefig(run_dir / "ttp_parcel_brain.png", dpi=150, bbox_inches="tight")
    plt.close(fig_brain)


def create_best_video(
    mesh: Any,
    medmask: np.ndarray,
    hetero_map: np.ndarray | None,
    aniso_map: np.ndarray | None,
    best_params: Dict[str, float],
    n_modes: int,
    dt: float,
    nt: int,
    r: float,
    gamma: float,
    ext_input: np.ndarray,
    run_dir: Path
) -> None:
    """
    Reconstruct the best model and generate a wave propagation video.
    
    Parameters
    ----------
    mesh : Surface object
        Cortical surface mesh
    medmask : np.ndarray
        Binary mask for medial wall exclusion
    hetero_map : np.ndarray | None
        Heterogeneity map (or None if not used)
    aniso_map : np.ndarray | None
        Anisotropy map (or None if not used)
    best_params : Dict[str, float]
        Best parameter set from optimization
    n_modes : int
        Number of eigenmodes to solve for
    dt : float
        Simulation timestep in seconds
    nt : int
        Number of timesteps
    r : float
        Coupling strength (mm)
    gamma : float
        Damping rate (s^-1)
    ext_input : np.ndarray
        External input time series (vertices x time)
    run_dir : Path
        Output directory for the saved video
    """
    print("Generating best model video...")
    
    # Reconstruct solver with best parameters
    solver_kwargs = {"surf": mesh, "mask": medmask}
    
    if "alpha" in best_params:
        alpha = best_params.get("alpha", 0)
        solver_kwargs["hetero"] = hetero_map if alpha != 0 else None
        solver_kwargs["alpha"] = alpha if alpha != 0 else None
    
    if "beta" in best_params:
        beta = best_params.get("beta", 0)
        solver_kwargs["aniso_map"] = aniso_map if beta != 0 else None
        solver_kwargs["beta"] = beta if beta != 0 else None
    
    if "aniso_curv1" in best_params or "aniso_curv2" in best_params:
        aniso_curv1 = best_params.get("aniso_curv1", 0)
        aniso_curv2 = best_params.get("aniso_curv2", 0)
        if aniso_curv1 != 0 or aniso_curv2 != 0:
            solver_kwargs["aniso_curv"] = (aniso_curv1, aniso_curv2)
    
    solver = EigenSolver(**solver_kwargs).solve(n_modes=n_modes)
    
    # Run simulation
    neural = solver.simulate_waves(
        ext_input=ext_input,
        nt=nt,
        dt=dt,
        r=r,
        gamma=gamma,
    )
    
    # Prepare video data (skip first 10 timesteps)
    neural_ss = neural[:, 10:]
    clim = np.vstack([np.nanpercentile(neural_ss, 2, axis=0), np.nanpercentile(neural_ss, 98, axis=0)]).T
    
    # Build mesh dict for plotting
    mesh_plot = {"v": mesh.vertices, "t": mesh.faces}
    
    # Create video with improved rendering
    video_file = plot_surf_video(
        surf=mesh_plot,
        data_timeseries=unmask(neural_ss, medmask),
        filename=str(run_dir / "best_model.mp4"),
        framerate=100,
        cmap="viridis",
        clim=clim,
        views=["lateral", "medial"]
    )
    
    print(f"Video saved: {video_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimise TTP alpha/beta to maximise |Spearman rho|.")
    p.add_argument("--test", action="store_true", help="Run in test mode using folder 0 and _cache_test.")
    p.add_argument(
        "--id",
        type=int,
        default=None,
        help="Optional run ID for intentional continuation. Non-test mode does not allow ID 0.",
    )
    p.add_argument("--hetero_label", type=str, default=None, help="Heterogeneity map label.")
    p.add_argument("--aniso_label", type=str, default=None, help="Anisotropy map label")
    p.add_argument("--density", type=str, default="32k", help="Surface density for mesh and parcellation.")
    p.add_argument("--n_modes", type=int, default=1000, help="Number of eigenmodes.")
    p.add_argument("--alpha", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Alpha optimization range: MIN MAX STEP")
    p.add_argument("--beta", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Beta optimization range: MIN MAX STEP")
    p.add_argument("--aniso_curv1", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Aniso_curv1 optimization range: MIN MAX STEP")
    p.add_argument("--aniso_curv2", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="Aniso_curv2 optimization range: MIN MAX STEP")
    p.add_argument("--r", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="r optimization range: MIN MAX STEP")
    p.add_argument("--gamma", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"), help="gamma optimization range: MIN MAX STEP")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers for DE (-1 = all).")
    p.add_argument("--maxiter", type=int, default=50, help="DE max iterations.")
    p.add_argument("--popsize", type=int, default=16, help="DE population size multiplier.")
    p.add_argument("--seed", type=int, default=365, help="Random seed for differential evolution initialization.")
    p.add_argument("--polish", action="store_true", help="Enable DE polish step (usually off).")
    return p.parse_args()


def main() -> None:
    t1 = time.time()
    args = parse_args()

    if args.id is not None and int(args.id) < 0:
        raise ValueError("--id must be >= 0")
    if args.test and args.id is not None:
        raise ValueError("--id cannot be used with --test")
    if (not args.test) and args.id == 0:
        raise ValueError("Run ID 0 is reserved for --test mode; use --id >= 1")

    results_dir = Path(PROJ_DIR) / "results" / "human" / "ttp"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.test:
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

    hetero_token = _validate_pair_component(args.hetero_label, "hetero_label")
    aniso_token = _validate_pair_component(args.aniso_label, "aniso_label")
    pair_name = f"hetero-{hetero_token}_aniso-{aniso_token}"
    pair_dir = run_parent / pair_name
    if not pair_dir.exists():
        pair_dir.mkdir(parents=True, exist_ok=False)

    eval_dir = pair_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pair_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter specs dict with only the parameters provided
    param_specs = {}
    bounds = []
    defaults = {
        "alpha": None,
        "beta": None,
        "aniso_curv1": None,
        "aniso_curv2": None,
        "r": float(DEFAULT_R),
        "gamma": float(DEFAULT_GAMMA),
    }
    
    if args.alpha is not None:
        alpha_spec = _parse_grid3(tuple(args.alpha), "alpha")
        print(f"Optimising with alpha in [{alpha_spec.min}, {alpha_spec.max}] step {alpha_spec.step}")
        param_specs["alpha"] = alpha_spec
        bounds.append((alpha_spec.min, alpha_spec.max))
    else:
        print("Not optimising alpha (using default: None)")
    
    if args.beta is not None:
        beta_spec = _parse_grid3(tuple(args.beta), "beta")
        print(f"Optimising with beta in [{beta_spec.min}, {beta_spec.max}] step {beta_spec.step}")
        param_specs["beta"] = beta_spec
        bounds.append((beta_spec.min, beta_spec.max))
    else:
        print("Not optimising beta (using default: None)")
    
    if args.aniso_curv1 is not None:
        aniso_curv1_spec = _parse_grid3(tuple(args.aniso_curv1), "aniso_curv1")
        print(f"Optimising with aniso_curv1 in [{aniso_curv1_spec.min}, {aniso_curv1_spec.max}] step {aniso_curv1_spec.step}")
        param_specs["aniso_curv1"] = aniso_curv1_spec
        bounds.append((aniso_curv1_spec.min, aniso_curv1_spec.max))
    else:
        print("Not optimising aniso_curv1 (using default: None)")
    
    if args.aniso_curv2 is not None:
        aniso_curv2_spec = _parse_grid3(tuple(args.aniso_curv2), "aniso_curv2")
        print(f"Optimising with aniso_curv2 in [{aniso_curv2_spec.min}, {aniso_curv2_spec.max}] step {aniso_curv2_spec.step}")
        param_specs["aniso_curv2"] = aniso_curv2_spec
        bounds.append((aniso_curv2_spec.min, aniso_curv2_spec.max))
    else:
        print("Not optimising aniso_curv2 (using default: None)")

    if args.r is not None:
        r_spec = _parse_grid3(tuple(args.r), "r")
        print(f"Optimising with r in [{r_spec.min}, {r_spec.max}] step {r_spec.step}")
        param_specs["r"] = r_spec
        bounds.append((r_spec.min, r_spec.max))
    else:
        print(f"Not optimising r (using default: {DEFAULT_R})")

    if args.gamma is not None:
        gamma_spec = _parse_grid3(tuple(args.gamma), "gamma")
        print(f"Optimising with gamma in [{gamma_spec.min}, {gamma_spec.max}] step {gamma_spec.step}")
        param_specs["gamma"] = gamma_spec
        bounds.append((gamma_spec.min, gamma_spec.max))
    else:
        print(f"Not optimising gamma (using default: {DEFAULT_GAMMA})")
    
    if not param_specs:
        raise ValueError(
            "At least one parameter must be provided for optimization (--alpha, --beta, --aniso_curv1, --aniso_curv2, --r, or --gamma)"
        )
    
    print(f"\nTotal optimization dimensions: {len(param_specs)}")

    # Simulation constants (keep simple; edit here if needed)
    dt = 1e-4   # seconds
    nt = 1000
    fixed_params = {
        name: defaults[name]
        for name in defaults
        if name not in param_specs and defaults[name] is not None
    }
    active_parameter_names = list(param_specs.keys())
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
    # -1 because 0 label refers to medial wall which has been removed in our masked arrays
    vis_hierarchy_roi_labels = np.load(vis_hierarchy_path).astype(int) - 1 

    if args.hetero_label is not None:
        hetero_map = load_hmap(args.hetero_label, density=args.density)
    else:
        hetero_map = None
    if args.aniso_label is not None:
        aniso_map = load_hmap(args.aniso_label, density=args.density)
    else:
        aniso_map = None

    # Proxy hierarchy: myelinmap parcellated on the same masked labels (so indices are label_key-1)
    myelin = load_hmap("myelinmap", density=args.density)
    myelin_parc = reduce_by_labels(myelin[medmask], parc_labels_masked, axis=0)
    proxy_hierarchy = myelin_parc[vis_hierarchy_roi_labels]

    # External input (vertex x time) in masked space
    ext_input = np.zeros((mesh.vertices.shape[0], nt), dtype=np.float32)
    ext_input[v1_mask, stim_start:stim_stop] = float(stimulation_amplitude)
    ext_input = ext_input[medmask, :]

    id_config = {
        "schema_version": 1,
        "objective_version": OBJECTIVE_VERSION,
        "run_id": run_id,
        "test_mode": bool(args.test),
        "hetero_label": args.hetero_label,
        "aniso_label": args.aniso_label,
        "n_modes": int(args.n_modes),
        "n_jobs": int(args.n_jobs),
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "seed": int(args.seed),
        "polish": bool(args.polish),
        "dt": float(dt),
        "nt": int(nt),
        "stimulation_amplitude": float(stimulation_amplitude),
        "stim_start": int(stim_start),
        "stim_stop": int(stim_stop),
        "density": args.density,
        "defaults": defaults,
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

    config = {
        "schema_version": 1,    # Increment if config structure changes in a non-backwards-compatible way
        "objective_version": OBJECTIVE_VERSION, # Increment if objective function changes in a non-backwards-compatible way
        "run_id": run_id,
        "test_mode": bool(args.test),
        "hetero_label": args.hetero_label,
        "aniso_label": args.aniso_label,
        "n_modes": int(args.n_modes),
        "n_jobs": int(args.n_jobs),
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "seed": int(args.seed),
        "polish": bool(args.polish),
        "dt": float(dt),
        "nt": int(nt),
        "r": float(fixed_params["r"]) if "r" in fixed_params else None,
        "gamma": float(fixed_params["gamma"]) if "gamma" in fixed_params else None,
        "stimulation_amplitude": float(stimulation_amplitude),
        "stim_start": int(stim_start),
        "stim_stop": int(stim_stop),
        "density": args.density,
        "defaults": defaults,
        "active_parameter_names": active_parameter_names,
        "fixed_params": fixed_params,
        "id_config_file": str(id_config_path),
        "pair_name": pair_name,
        "pair_dir": str(pair_dir),
        "config_file": str(pair_dir / "config.json"),
    }
    for name, spec in param_specs.items():
        config[name] = asdict(spec)
    config["optimization_parameters"] = {name: asdict(spec) for name, spec in param_specs.items()}
    id_config_path.write_text(json.dumps(id_config, indent=2, sort_keys=True), encoding="utf-8")
    (pair_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    meta_base = {
        "objective_version": OBJECTIVE_VERSION,
        "hetero_label": args.hetero_label,
        "aniso_label": args.aniso_label,
        "n_modes": int(args.n_modes),
        "dt": float(dt),
        "nt": int(nt),
        "r": float(fixed_params["r"]) if "r" in fixed_params else None,
        "gamma": float(fixed_params["gamma"]) if "gamma" in fixed_params else None,
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
        fixed_params=fixed_params,
        n_modes=int(args.n_modes),
        dt=dt,
        nt=nt,
        ext_input=ext_input,
        cache_dir=cache_dir,
        eval_dir=eval_dir,
        meta_base=meta_base,
    )

    is_single_point = all(spec.min == spec.max for spec in param_specs.values())

    if is_single_point:
        print("Single-point mode: skipping differential_evolution and evaluating once.")
        best_params = dict(fixed_params)
        for name, spec in param_specs.items():
            best_params[name] = spec.min
        x_fixed = np.array([best_params[name] for name in param_specs.keys()], dtype=float)
        fun = float(evaluator(x_fixed))
        de_result = {
            "x": [float(v) for v in x_fixed],
            "fun": fun,
            "nfev": 1,
            "nit": 0,
            "success": True,
            "message": "Single-point evaluation (all bounds fixed).",
        }
    else:
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

        best_params = dict(fixed_params)
        for i, name in enumerate(param_specs.keys()):
            spec = param_specs[name]
            best_params[name] = _snap_to_grid(float(res.x[i]), spec.min, spec.max, spec.step)

        de_result = {
            "x": [float(v) for v in res.x],
            "fun": float(res.fun),
            "nfev": int(res.nfev),
            "nit": int(res.nit),
            "success": bool(res.success),
            "message": str(res.message),
        }
    
    best_key, best_cache_path = evaluator._cache_key_and_path(best_params)
    cached = np.load(best_cache_path, allow_pickle=False)
    
    # Validate that the cached result matches the expected parameters
    # (guards against hash collisions or stale cache reuse)
    cached_meta = json.loads(str(cached["meta_json"]))
    for param_name, param_value in best_params.items():
        if param_name not in cached_meta:
            print(f"WARNING: Parameter '{param_name}' not found in cached metadata")
        elif abs(float(cached_meta[param_name]) - param_value) > 1e-6:
            print(f"ERROR: Parameter mismatch for '{param_name}'")
            print(f"  Expected: {param_value}")
            print(f"  Got from cache: {cached_meta[param_name]}")
            raise ValueError(f"Cached result does not match expected parameters")
    
    best = dict(best_params)
    best["rho"] = float(cached["rho"])
    best["abs_rho"] = float(cached["abs_rho"])
    best["cache_key"] = best_key
    (pair_dir / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

    (pair_dir / "de_result.json").write_text(
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
        with (pair_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    param_str = ", ".join([f"{name}={val:.2f}" for name, val in best_params.items()])
    print(f"\nBest |rho| = {best['abs_rho']:.4f} at {param_str}")
    print(f"Run parent folder (ID={run_id}): {run_parent}")
    print(f"Pair folder: {pair_dir}")
    print(f"Total optimisation time: {(time.time() - t1)/60:.2f}min")
    
    # Generate landscape plot
    plot_optimization_landscape(pair_dir, save_path=pair_dir / "landscape.png")
    
    # Generate best model summary plot
    plot_best_summary(
        cache_path=best_cache_path,
        run_dir=pair_dir,
        proxy_hierarchy=proxy_hierarchy,
        vis_hierarchy_roi_labels=vis_hierarchy_roi_labels,
        parc_labels=parc_labels,
        parc_labels_masked=parc_labels_masked,
        mesh=mesh,
        medmask=medmask,
        dt=dt,
    )
    
    # Generate best model video
    create_best_video(
        mesh=mesh,
        medmask=medmask,
        hetero_map=hetero_map,
        aniso_map=aniso_map,
        best_params=best_params,
        n_modes=int(args.n_modes),
        dt=dt,
        nt=nt,
        r=float(best_params.get("r", DEFAULT_R)),
        gamma=float(best_params.get("gamma", DEFAULT_GAMMA)),
        ext_input=ext_input,
        run_dir=pair_dir
    )


if __name__ == "__main__":
    main()

