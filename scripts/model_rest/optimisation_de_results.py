from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from heteromodes.utils import get_project_root


DE_ID = 12
CONFIG_PATH = Path("data/heteromaps/human/heteromaps_config.json")
RESULTS_DIR = Path(f"results/human/model_rest/de/{DE_ID}")
OUT_PATH = Path(f"{RESULTS_DIR}/optimisation_results.png")


def _read_json(path: Path) -> dict[str, Any]:
	with path.open("r") as f:
		return json.load(f)


def _to_float(value: Any) -> float:
	if value is None:
		return float("nan")
	return float(value)


def main() -> None:
	proj_dir = Path(get_project_root())
	data = _read_json(proj_dir / CONFIG_PATH)
	config = {"None": {"label": "None"}, **data}
	# config.pop("SAaxis", None)
	# config.pop("curvature", None)

	map_ids = list(config.keys())
	map_labels = [config[k]["label"] for k in map_ids]
	n = len(map_ids)

	# Store all requested fields (even if we only plot edge/node for now).
	alpha = np.full((n, n), np.nan, dtype=float)
	beta = np.full((n, n), np.nan, dtype=float)
	r = np.full((n, n), np.nan, dtype=float)
	combined_metric = np.full((n, n), np.nan, dtype=float)
	r_edge = np.full((n, n), np.nan, dtype=float)
	r_node = np.full((n, n), np.nan, dtype=float)
	r_phase = np.full((n, n), np.nan, dtype=float)

	missing: list[str] = []
	for i, hetero_id in enumerate(map_ids):
		for j, aniso_id in enumerate(map_ids):
			folder = f"hetero-{hetero_id}-aniso-{aniso_id}"
			best_path = proj_dir / RESULTS_DIR / folder / "best.json"
			if not best_path.exists():
				missing.append(str(best_path.relative_to(proj_dir)))
				continue

			best = _read_json(best_path)
			alpha[i, j] = _to_float(best.get("alpha"))
			beta[i, j] = _to_float(best.get("beta"))
			r[i, j] = _to_float(best.get("r"))
			combined_metric[i, j] = -_to_float(best.get("objective"))
			r_edge[i, j] = _to_float(best.get("edge_fc_corr"))
			r_node[i, j] = _to_float(best.get("node_fc_corr"))
			r_phase[i, j] = _to_float(best.get("cpc1_corr"))

	if missing:
		# Keep output concise but visible during runs.
		preview = "\n".join(missing[:10])
		suffix = "" if len(missing) <= 10 else f"\n... (+{len(missing) - 10} more)"
		print(f"Warning: missing {len(missing)} best.json files:\n{preview}{suffix}")

	# Compute difference against homogeneous, isotropic baseline (first row and column).
	r_edge_diff = r_edge - r_edge[0, 0]
	r_node_diff = r_node - r_node[0, 0]
	r_phase_diff = r_phase - r_phase[0, 0]
	combined_diff = combined_metric - combined_metric[0, 0]

	# Define best hetero and aniso maps based on combined metric
	best_idx = np.unravel_index(np.nanargmax(combined_metric), combined_metric.shape)
	best_hetero = map_ids[best_idx[0]]
	best_aniso = map_ids[best_idx[1]]
	print(f"Best combined metric: {combined_metric[best_idx]:.4f} (hetero: {best_hetero}, aniso: {best_aniso})")

	fig, axs = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
	
	panels = [
		(r_edge_diff, r"$r_{\text{edge}}$", "Correlation", "RdBu_r"),
		(r_node_diff, r"$r_{\text{node}}$", "Correlation", "RdBu_r"),
		(r_phase_diff, r"$r_{\text{phase}}$", "Correlation", "RdBu_r"),
		(combined_diff, "Combined metric", "Edge + Node", "RdBu_r"),
		(r, r"$r$", "r", "viridis"),
		(alpha, r"$\alpha$", "alpha", "coolwarm"),
		(beta, r"$\beta$", "beta", "coolwarm"),
	]

	for ax, (mat, title, cbar_label, cmap) in zip(axs.ravel(), panels):
		if cbar_label != "r":
			vmin = -np.nanmax(np.abs(mat))
			vmax = np.nanmax(np.abs(mat))
		else:
			vmin = np.nanmin(mat)
			vmax = np.nanmax(mat)
			
		im = ax.imshow(mat, origin="upper", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
		ax.set_title(title)
		ax.set_xlabel("Anisotropy map")
		ax.set_ylabel("Heterogeneity map")

		ax.set_xticks(np.arange(n))
		ax.set_yticks(np.arange(n))
		ax.set_xticklabels(map_labels, rotation=45, ha="right")
		ax.set_yticklabels(map_labels)

		cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		cbar.ax.set_ylabel(cbar_label, rotation=90)

		# Draw black box around best model cell based on combined metric
		rect = plt.Rectangle(
			(best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1, 
			edgecolor="black", 
			facecolor="none", 
			linewidth=2
		)
		ax.add_patch(rect)

	out_path = proj_dir / OUT_PATH
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.01)
	plt.close(fig)

	print(f"Saved: {out_path.relative_to(proj_dir)}")


if __name__ == "__main__":
	main()

