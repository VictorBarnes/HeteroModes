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
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import reduce_by_labels
from scipy.stats import zscore

from heteromodes.optimisation import (
    GridSpec,
    ObjectiveEvaluator,
    atomic_write_json,
    build_manifest,
    collect_config_mismatches,
    hash_payload,
    next_run_id,
    normalize_config_for_id_check,
    parse_grid3,
    run_differential_evolution,
    validate_path_component,
)
from heteromodes.restingstate import analyze_bold, calc_node_fc, evaluate_model
from heteromodes.utils import get_project_root, load_hmap
from neuromodes.eigen import EigenSolver
from nsbutils.plotting_pyvista import plot_surf
from nsbutils.utils import unmask

PROJ_DIR = get_project_root()

OBJECTIVE_VERSION = "model_rest_de_fit_v1"

# All default parameter values are None
DEFAULT_ALPHA = None
DEFAULT_BETA = None
DEFAULT_ANISO_CURV1 = None
DEFAULT_ANISO_CURV2 = None
DEFAULT_R = 18.0
DEFAULT_GAMMA = 116.0

METRIC_CHOICES = ("edge_fc_corr", "node_fc_corr", "cpc1_corr")
PARAM_ORDER = ("alpha", "beta", "aniso_curv1", "aniso_curv2", "r", "gamma")


def _validate_levels(
    surf_level: str, 
    hmap_level: str,
    fmri_level: str,
    data_desc: str
) -> str:
    """
    Validate that all individual-level arguments have the same subject ID, and return the subject ID or 'group'.
    """
    all_levels = {
        "surf": surf_level,
        "hmap": hmap_level,
        "fmri": fmri_level
    }
    subj_ids = {name: val for name, val in all_levels.items() if val != "group"}
    unique_ids = set(subj_ids.values())
    if len(unique_ids) > 1:
        raise ValueError(
            f"All individual-level arguments must have the same subject ID. "
            f"Found: {subj_ids}"
        )
    level = unique_ids.pop() if unique_ids else "group"

    if level == "group":
        return level
    else:   # individual-level, validate against subject list
        subj_list_path = (
            Path(PROJ_DIR)
            / "data"
            / "empirical"
            / "human"
            / f"{data_desc}_desc-subjects.txt"   # one sub-ID per line
        )
        if not subj_list_path.exists():
            raise FileNotFoundError(
                f"Subject list not found: {subj_list_path}. "
                "Cannot validate --subj_id."
            )
        valid_ids = {line.strip() for line in subj_list_path.read_text().splitlines() if line.strip()}
        if level not in valid_ids:
            raise ValueError(
                f"--subj_id {level!r} is not a valid subject for data_desc={data_desc!r}. "
                f"({len(valid_ids)} subjects in list.)"
            )
        return f"sub-{level}"

def _build_param_specs(
    args: argparse.Namespace
) -> Tuple[Dict[str, GridSpec], Dict[str, Any], Dict[str, Any], str]:

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
            specs[name] = parse_grid3(tuple(values), name)

    fixed_params = {
        name: defaults[name]
        for name in active_param_names
        if name not in specs and defaults[name] is not None
    }

    return specs, fixed_params, defaults, aniso_mode

def _fetch_empirical_constants(
    species: str,
    n_subjs: int,
    dataset: str,
    cohort: Optional[str],
) -> Tuple[int, float, float, str, int]:

    _CONSTANTS = {
        ("human",    "hcp-s1200"): (1200, 0.72, 0.09),
        ("human",    "hcp-ep"):    ( 820, 0.80, 0.10),
        ("macaque",  "default"):   ( 500, 2.60, 0.10),
        ("marmoset", "default"):   ( 510, 2.00, 0.10),
    }

    # Datasets that require a cohort label to construct data_desc
    _COHORT_REQUIRED = {"hcp-ep"}

    _DATA_DESC = {
        "human": {
            "hcp-s1200": f"hcp-s1200_nsubj-{n_subjs}",
            "hcp-ep":    f"hcp-ep-{cohort}_run-1-2_nsubj-{n_subjs}",
        },
        "macaque":  f"macaque-awake_nsubj-{n_subjs}",
        "marmoset": f"mbm-v4_nsubj-{n_subjs}",
    }

    constants_key = (species, dataset if species == "human" else "default")
    if constants_key not in _CONSTANTS:
        raise ValueError(f"Unknown species/dataset combination: {species!r}, {dataset!r}")

    # Validate cohort before attempting to use it
    if dataset in _COHORT_REQUIRED and not cohort:
        raise ValueError(f"--cohort is required for dataset {dataset!r}")

    nt_emp, dt_emp, dt_model = _CONSTANTS[constants_key]

    desc = _DATA_DESC[species]
    if isinstance(desc, dict):
        if dataset not in desc:
            raise ValueError(f"Unknown dataset for species {species!r}: {dataset!r}")
        data_desc = desc[dataset]
    else:
        data_desc = desc

    tsteady = 550

    return nt_emp, dt_emp, dt_model, data_desc, tsteady


def _setup_surface_and_masks(
    args: argparse.Namespace,
) -> Tuple[str, Optional[np.ndarray], np.ndarray, str]:

    is_individual = args.surf_level != "group"

    if is_individual:
        # Hard code HCP-EP directory for now
        HCPEP_DIR = Path("/fs03/kg98/vbarnes/hcp-ep")

        # Subject-specific surface registered to fsLR
        surf = str(
            HCPEP_DIR / f"sub-{args.surf_level}" / "MNINonLinear" / "Results"
            / f"space-fsLR_den-{args.density}_hemi-L_desc-midthickness.surf.gii"
        )
        medmask = nib.load(
            str(
                HCPEP_DIR / f"sub-{args.surf_level}" / "MNINonLinear" / "Results"
                / f"space-fsLR_den-{args.density}_hemi-L_desc-nomedialwall.func.gii"
            )
        ).darrays[0].data.astype(bool)
    else:
        # existing group-level paths (unchanged)
        surf = str(
            Path(PROJ_DIR)
            / "data" / "empirical" / args.species
            / f"space-fsLR_den-{args.density}_hemi-L_desc-midthickness.surf.gii"
        )
        medmask = nib.load(
            str(
                Path(PROJ_DIR)
                / "data" / "empirical" / args.species
                / f"{args.dataset}_space-fsLR_den-{args.density}_hemi-L_desc-nomedialwall.func.gii"
            )
        ).darrays[0].data.astype(bool)

    # parcellation logic is unchanged — parc is group-level only; raise if individual + parc
    if args.parc is not None and is_individual:
        raise ValueError("--parc is not supported with individual-level fitting.")
    parc = None

    space_desc = f"space-fsLR_den-{args.density}"
    return surf, parc, medmask, space_desc


def _load_maps(
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    is_individual = args.hmap_level != "group"

    if args.hetero_label is None:
        hetero_map = None
    elif is_individual:
        # hard code hcpep directory for now
        HCPEP_DIR = Path("/fs03/kg98/vbarnes/hcp-ep")
        
        hetero_map = nib.load(
            str(
                HCPEP_DIR
                / f"sub-{args.hmap_level}"
                / "MNINonLinear"
                / "Results"
                / f"space-fsLR_den-{args.density}_hemi-L_desc-{args.hetero_label}.func.gii"
            )
        ).darrays[0].data
    elif args.hetero_label.startswith("null"):
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

        # p_lower, p_upper = np.percentile(hetero_map[medmask], [2, 98])
        # hetero_map = np.clip(hetero_map, p_lower, p_upper)
    else:
        hetero_map = load_hmap(args.hetero_label, species=args.species, density=args.density)

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
    noise_seed: Optional[int],
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
            "seed": noise_seed + i,
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
    level: str,
    nt_emp: int,
    band_freq: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    outputs: Dict[str, np.ndarray] = {}

    is_individual = level != "group"

    if "edge_fc_corr" in metrics or "node_fc_corr" in metrics:
        fc_file = (
            Path(PROJ_DIR)
            / "data"
            / "empirical"
            / species
            / f"{data_desc}_desc-fc_{space_desc}_hemi-L_nt-{nt_emp}.h5"
        )
        if is_individual:
            with h5py.File(fc_file, "r") as f:
                # Get subject-specific index
                subj_ids = np.array(f["subj_ids"], dtype=str)
                # Find the index of the requested subject
                fc_index = np.where(subj_ids == level.removeprefix("sub-"))[0][0]
                # Load only the specific subject's FC
                outputs["fc"] = np.asarray(f[f"fc_indiv"][:, :, fc_index], dtype=np.float32)
        else:   
            with h5py.File(fc_file, "r") as f:
                outputs["fc"] = np.asarray(f["fc_group"], dtype=np.float32)

    if "cpc1_corr" in metrics:
        # TODO: individual level
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
    metrics: Sequence[str],
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

    metric_results = {metric: [float(r[metric]) for r in rows] for metric in metrics}
    objective = np.asarray([float(r["objective"]) for r in rows], dtype=float) * -1.0
    best_idx = int(np.argmax(objective))
    
    saved_paths: List[Path] = []

    if len(free_param_names) == 1:
        p0 = free_param_names[0]
        x = np.asarray([float(r[p0]) for r in rows], dtype=float)
        fig, axs = plt.subplots(1, len(metrics)+1, figsize=(7 * (len(metrics)+1), 5), constrained_layout=True)
        # Plot landscape for each metric
        for i, metric in enumerate(metrics):
            ax = axs[i]
            metric_vals = np.asarray(metric_results[metric], dtype=float)
            sc = ax.scatter(x, metric_vals, c=metric_vals, cmap="viridis", s=45, alpha=0.8, edgecolors="black", linewidth=0.4)
            ax.plot([x[best_idx]], [metric_vals[best_idx]], "r*", markersize=16, label=f"Best {metric}={metric_vals[best_idx]:.4f}")
            ax.set_xlabel(p0)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} Landscape")
            ax.legend(loc="best")
            fig.colorbar(sc, ax=ax).set_label(metric)
        # Plot landscape for objective
        sc = axs[-1].scatter(x, objective, c=objective, cmap="viridis", s=45, alpha=0.8, edgecolors="black", linewidth=0.4)
        axs[-1].plot([x[best_idx]], [objective[best_idx]], "r*", markersize=16, label=f"Best objective={objective[best_idx]:.4f}")
        axs[-1].set_xlabel(p0)
        axs[-1].set_ylabel("Objective")
        axs[-1].set_title("Objective Landscape")
        axs[-1].legend(loc="best")
        fig.colorbar(sc, ax=axs[-1]).set_label("Objective")
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

            fig, axs = plt.subplots(1, len(metrics) + 1, figsize=(7 * (len(metrics) + 1), 5), constrained_layout=True)
            # Plot landscape for each metric
            for k, metric in enumerate(metrics):
                ax = axs[k]
                metric_vals = np.asarray(metric_results[metric], dtype=float)
                sc = ax.scatter(
                    x,
                    y,
                    c=metric_vals,
                    cmap="viridis",
                    s=40,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.3,
                )
                ax.plot([x[best_idx]], [y[best_idx]], "r*", markersize=14, label=f"Best {metric}={metric_vals[best_idx]:.4f}")
                ax.set_xlabel(p1)
                ax.set_ylabel(p2)
                ax.set_title(f"{metric} Landscape: {p1} vs {p2}")
                ax.legend(loc="best")
                fig.colorbar(sc, ax=ax).set_label(metric)
            # Plot landscape for objective
            sc = axs[-1].scatter(
                x,
                y,
                c=objective,
                cmap="viridis",
                s=40,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.3,
            )
            axs[-1].plot([x[best_idx]], [y[best_idx]], "r*", markersize=14, label=f"Best objective={objective[best_idx]:.4f}")
            axs[-1].set_xlabel(p1)
            axs[-1].set_ylabel(p2)
            axs[-1].set_title(f"Objective Landscape: {p1} vs {p2}")
            axs[-1].legend(loc="best")
            fig.colorbar(sc, ax=axs[-1]).set_label("Objective")

            save_path = run_dir / f"landscape_{p1}-{p2}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(save_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize resting-state model parameters with differential evolution.")
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Optional run ID for intentional continuation. If not given, the next available run ID will be used. Use "
             "0 for a test run (results will always get overwritten on rerun).",
    )

    parser.add_argument("--species", type=str, choices=["human", "macaque", "marmoset"], default="human")
    parser.add_argument("--density", "--den", dest="density", type=str, default="4k", help="Surface density.")

    parser.add_argument("--hetero_label", type=lambda x: None if x.lower() == "none" else x, default=None)
    parser.add_argument("--aniso_label", type=lambda x: None if x.lower() == "none" else x, default=None)

    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_modes", type=int, default=500)
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="hcp-s1200",
        help="Empirical dataset label, e.g. 'hcp-s1200' or 'hcp-ep' (only human).",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Empirical human cohort label for hcp-ep dataset, e.g. 'hc', 'scz'",
    )
    parser.add_argument("--n_subjs", type=int, default=255)
    parser.add_argument(
        "--surf_level",
        type=str,
        default="group",
        help=(
            "Surface level for modelling. Use 'group' for the group-average surface. "
            "Pass a valid subject ID (e.g. '1001') to use an individual-specific surface. "
            "Only implemented for HCP-EP dataset."
        ),
    )
    parser.add_argument(
        "--hmap_level",
        type=str,
        default="group",
        help=(
            "Heterogeneity map level for modelling. Use 'group' for the group-average map. "
            "Pass a valid subject ID (e.g. '1001') to use an individual-specific map. "
            "Only implemented for HCP-EP dataset."
        ),
    )
    parser.add_argument(
        "--fmri_level",
        type=str,
        default="group",
        help=(
            "fMRI level for modelling. Use 'group' for the group-average fMRI data. "
            "Pass a valid subject ID (e.g. '1001') to use an individual-specific fMRI dataset. "
            "Only implemented for HCP-EP dataset."
        ),
    )

    parser.add_argument("--alpha", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--beta", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--aniso_curv1", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--aniso_curv2", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--r", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--gamma", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--noise_seed", type=int, default=0, help="Seed for noise generation in BOLD simulation.")

    parser.add_argument("--maxiter", type=int, default=50, help="Maximum differential-evolution iterations.")
    parser.add_argument("--popsize", type=int, default=16, help="Population size multiplier for differential evolution.")
    parser.add_argument("--de_seed", type=int, default=365, help="Seed for differential evolution initialization.")
    parser.add_argument(
        "--cpc_seed",
        type=int,
        default=365,
        help="Seed for CPC extraction randomness (fbpca) when using cpc1_corr.",
    )
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
    if (args.surf_level != "group" or args.hmap_level != "group" or args.fmri_level != "group") and args.dataset != "hcp-ep":
        raise NotImplementedError("Individual-level fitting is only implemented for --dataset hcp-ep at this time.")
    
    if args.dataset == "hcp-ep":
        if not args.cohort or args.cohort.lower() not in {"hc", "scz"}:
            raise ValueError("--cohort (e.g. 'hc' or 'scz') is required when --dataset hcp-ep is used.")
    
    if args.id is not None and int(args.id) < 0:
        raise ValueError("--id must be >= 0")

    if args.metrics is None or len(args.metrics) == 0:
        raise ValueError("At least one metric must be supplied via --metrics")

    param_specs, fixed_params, defaults, aniso_mode = _build_param_specs(args)
    if aniso_mode == "map" and args.aniso_label is None:
        raise ValueError("beta optimization requires --aniso_label (or --hetero_label so aniso can resolve from it)")

    free_param_names = list(param_specs.keys())
    if not free_param_names:
        print("No free optimization parameters provided; evaluating the fixed default model only.")

    hetero_token = validate_path_component(args.hetero_label, "hetero_label")
    aniso_token = validate_path_component(args.aniso_label, "aniso_label")
    pair_name = f"hetero-{hetero_token}_aniso-{aniso_token}"

    results_dir = (
        Path(PROJ_DIR) / "results" / "model_rest" / args.species / args.dataset
    )
    if args.cohort is not None:
        results_dir = results_dir / args.cohort
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch empirical constants and construct data description
    nt_emp, dt_emp, dt_model, data_desc, tsteady = _fetch_empirical_constants(
        args.species, args.n_subjs, args.dataset, args.cohort
    )

    # Level validation helper
    level = _validate_levels(args.surf_level, args.hmap_level, args.fmri_level, data_desc)
    results_dir = results_dir / level

    if args.id == 0:
        print("Using run ID 0 (scratch/test slot). Existing contents of this folder will be deleted.")
        run_id = 0
        run_parent = results_dir / "id-0"
        if run_parent.exists():
            shutil.rmtree(run_parent)
        run_parent.mkdir(parents=True, exist_ok=False)
    elif args.id is None:
        run_id = next_run_id(results_dir)
        run_parent = results_dir / f"id-{run_id}"
        while True:
            try:
                run_parent.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                run_id = next_run_id(results_dir)
                run_parent = results_dir / f"id-{run_id}"
    else:
        run_id = int(args.id)
        run_parent = results_dir / f"id-{run_id}"
        run_parent.mkdir(parents=True, exist_ok=True)
    print(f"Using run ID {run_id} with parent folder {run_parent}")

    pair_dir = run_parent / pair_name
    if not pair_dir.exists():
        pair_dir.mkdir(parents=True, exist_ok=False)

    eval_dir = pair_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load surface and masks, and load heterogeneity/anistropy maps if provided
    surf, parc, medmask, space_desc = _setup_surface_and_masks(args)
    hetero_map, aniso_map = _load_maps(args)
    if hetero_map is not None:
        hetero_map = hetero_map[medmask]
    if aniso_map is not None:
        aniso_map = aniso_map[medmask]

    print("Loading empirical fit outputs...")
    band_freq = (float(args.band_freq[0]), float(args.band_freq[1]))
    emp_outputs = _load_empirical_fit_outputs(
        species=args.species,
        metrics=args.metrics,
        data_desc=data_desc,
        level=args.fmri_level,
        space_desc=space_desc,
        nt_emp=nt_emp,
        band_freq=band_freq,
    )

    id_config = {
        "schema_version": 1,
        "objective_version": OBJECTIVE_VERSION,
        "run_id": int(run_id),
        "species": args.species,
        "dataset": args.dataset,
        "cohort": args.cohort,
        "density": args.density,
        "evaluation": args.evaluation,
        "metrics": list(args.metrics),
        "n_runs": int(args.n_runs),
        "n_modes": int(args.n_modes),
        "n_subjs": int(args.n_subjs),
        "surf_level": args.surf_level,
        "hmap_level": args.hmap_level,
        "fmri_level": args.fmri_level,
        "band_freq": [float(v) for v in band_freq],
        "noise_seed": int(args.noise_seed),
        "scaling": args.scaling,
        "parc": args.parc,
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "de_seed": int(args.de_seed),
        "cpc_seed": int(args.cpc_seed),
        "polish": bool(args.polish),
        "defaults": defaults,
        "fixed_params": fixed_params,
    }

    id_config_path = run_parent / "id_config.json"
    if id_config_path.exists() and args.id != 0:
        saved = json.loads(id_config_path.read_text(encoding="utf-8"))
        mismatches = collect_config_mismatches(
            normalize_config_for_id_check(saved),
            normalize_config_for_id_check(id_config),
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
    run_config["run_hash"] = hash_payload(run_config)

    atomic_write_json(id_config_path, id_config)
    atomic_write_json(pair_dir / "config.json", run_config)
    (pair_dir / "run_hash.txt").write_text(f"{run_config['run_hash']}\n", encoding="utf-8")

    ext_input_cache_dir = Path(PROJ_DIR) / "results" / "model_rest" / "_cache_ext_input"
    ext_input_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CACHE_DIR"] = str(ext_input_cache_dir)

    def model_callback(params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        bold_data = _simulate_bold(
            surf=surf, medmask=medmask, parc=parc, hetero_map=hetero_map,
            aniso_map=aniso_map, alpha=params.get("alpha"), beta=params.get("beta"),
            aniso_curv1=params.get("aniso_curv1"), aniso_curv2=params.get("aniso_curv2"),
            r=params.get("r"), gamma=params.get("gamma"), noise_seed=args.noise_seed,
            scaling=args.scaling, n_modes=int(args.n_modes), n_runs=int(args.n_runs),
            nt_emp=int(nt_emp), dt_emp=float(dt_emp), dt_model=float(dt_model),
            tsteady=int(tsteady),
        )
        return analyze_bold(
            bold_data, dt_emp=float(dt_emp), band_freq=band_freq,
            metrics=list(args.metrics), cpc_seed=int(args.cpc_seed),
        )

    def score_callback(model_outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        return evaluate_model(model_outputs, emp_outputs, metrics=list(args.metrics))

    evaluator = ObjectiveEvaluator(
        model_callback=model_callback,
        score_callback=score_callback,
        metrics=args.metrics,
        param_specs=param_specs,
        fixed_params={**defaults, **fixed_params},
        eval_dir=eval_dir,
        config=run_config,
        cache={}, save_model_outputs=bool(args.save),
    )

    print("Starting optimization with differential evolution...")
    best_params, de_result = run_differential_evolution(
        evaluator, maxiter=int(args.maxiter), popsize=int(args.popsize),
        seed=int(args.de_seed), workers=int(args.n_jobs), polish=bool(args.polish), disp=True,
    )
    best_params = {**{name: None for name in PARAM_ORDER}, **best_params}

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

    (pair_dir / "de_result.json").write_text(json.dumps(de_result, indent=2, sort_keys=True), encoding="utf-8")

    rows = build_manifest(
        eval_dir,
        pair_dir / "manifest.csv",
        parameter_names=PARAM_ORDER,
        metric_names=METRIC_CHOICES,
    )
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

    _ = _plot_pairwise_landscape(
        run_dir=pair_dir, 
        free_param_names=free_param_names,
        metrics=args.metrics
    )

    print(f"Run parent folder (ID={run_id}): {run_parent}")
    print(f"Pair folder: {pair_dir}")
    print(f"Total optimisation time: {(time.time() - t0)/3600:.3f} hrs")


if __name__ == "__main__":
    main()
