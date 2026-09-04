"""Reusable differential-evolution optimization utilities.

The optimizer is model-agnostic. Callers provide a model-output callback and a
scoring callback, so the same machinery can optimize wave simulations,
connectomes, or other numerical model outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

ModelCallback = Callable[[Dict[str, Any]], Mapping[str, Any]]
ScoreCallback = Callable[[Mapping[str, Any]], Mapping[str, float]]


@dataclass(frozen=True)
class GridSpec:
    """A bounded, regularly spaced parameter domain."""

    min: float
    max: float
    step: float


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Return a stable short hash for JSON-compatible metadata."""

    data = json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]


def snap_to_grid(x: float, min_val: float, max_val: float, step: float) -> float:
    """Snap a value to a bounded grid."""

    if step <= 0:
        raise ValueError("step must be > 0")
    snapped = min_val + round((x - min_val) / step) * step
    return float(np.clip(snapped, min_val, max_val))


def parse_grid3(values: Tuple[float, float, float], name: str) -> GridSpec:
    """Parse ``(minimum, maximum, step)`` into a :class:`GridSpec`."""

    min_val, max_val, step = [float(value) for value in values]
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    if step <= 0:
        raise ValueError(f"{name} step must be > 0")
    return GridSpec(min=min_val, max=max_val, step=step)


def next_run_id(results_dir: Path, *, prefix: str = "id-") -> int:
    """Return the next positive run ID in a directory."""

    run_ids = []
    if results_dir.exists():
        for child in results_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name[len(prefix):] if prefix and child.name.startswith(prefix) else child.name
            if name.isdigit() and int(name) > 0:
                run_ids.append(int(name))
    return max(run_ids, default=0) + 1


def validate_path_component(value: Optional[str], arg_name: str) -> str:
    """Validate a value before using it as one directory-name component."""

    token = str(value)
    path_obj = Path(token)
    if token in {".", ".."} or path_obj.is_absolute() or len(path_obj.parts) != 1:
        raise ValueError(f"--{arg_name} must be a single folder-safe name (no path separators)")
    return token


def normalize_config_for_id_check(
    config: Mapping[str, Any],
    *,
    ignored_keys: Sequence[str] = (
        "run_hash", "maxiter", "popsize", "n_jobs", "pair_name", "pair_dir",
        "hetero_label", "aniso_label", "optimization_parameters",
        "id_config_file", "config_file",
    ),
) -> Dict[str, Any]:
    """Remove run-specific fields before comparing continuation configs."""

    normalized = dict(config)
    for key in ignored_keys:
        normalized.pop(key, None)
    return normalized


def collect_config_mismatches(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    """Return human-readable differences between nested config values."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches = []
        for key in sorted(set(expected) | set(actual)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                mismatches.append(f"{child_prefix}: unexpected key in current config")
            elif key not in actual:
                mismatches.append(f"{child_prefix}: missing from current config")
            else:
                mismatches.extend(collect_config_mismatches(expected[key], actual[key], child_prefix))
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{prefix}: expected list length {len(expected)}, got {len(actual)}"]
        mismatches = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatches.extend(collect_config_mismatches(expected_item, actual_item, f"{prefix}[{index}]"))
        return mismatches
    return [] if expected == actual else [f"{prefix}: expected {expected!r}, got {actual!r}"]


def build_manifest(
    eval_dir: Path,
    save_path: Path,
    *,
    parameter_names: Sequence[str],
    metric_names: Sequence[str],
) -> list[Dict[str, Any]]:
    """Write sorted objective JSON records to a CSV manifest."""

    rows = []
    for path in sorted(eval_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows.sort(key=lambda row: float(row.get("objective", np.inf)))
    if not rows:
        return []

    fieldnames = ["cache_key", "objective", "score", *parameter_names]
    fieldnames.extend(name for name in metric_names if name not in fieldnames)
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)

    with save_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def atomic_savez(path: Path, **arrays: Any) -> None:
    """Atomically write compressed NumPy data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def safe_write_json_once(path: Path, payload: Mapping[str, Any]) -> bool:
    """Write JSON only if the destination does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
        return True
    except FileExistsError:
        return False


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class TimingCallback:
    """Print DE iteration timing and snapped free-parameter values."""

    def __init__(self, param_specs: Mapping[str, GridSpec]) -> None:
        self.param_specs = param_specs
        self.param_names = list(param_specs)
        self.iteration_times: list[float] = []

    def __call__(self, xk: np.ndarray, convergence: float) -> None:
        self.iteration_times.append(time.time())
        values = {
            name: snap_to_grid(float(xk[index]), self.param_specs[name].min,
                              self.param_specs[name].max, self.param_specs[name].step)
            for index, name in enumerate(self.param_names)
        }
        param_str = ", ".join(f"{name}={value:.4g}" for name, value in values.items()) or "no free params"
        if len(self.iteration_times) > 1:
            elapsed = self.iteration_times[-1] - self.iteration_times[-2]
            print(f"  Iteration {len(self.iteration_times)}: {elapsed / 60:.3f}min | {param_str}, convergence={convergence:.4f}")
        else:
            print(f"  Iteration 1 (initial): {param_str}, convergence={convergence:.4f}")


@dataclass
class ObjectiveEvaluator:
    """Evaluate arbitrary model outputs with reusable DE/cache behavior.

    ``model_callback`` receives resolved parameter values. ``score_callback``
    receives the returned mapping and must return metric values. The score is
    the sum of the requested metric values and the minimization objective is
    its negative, matching the resting-state optimizer.
    """

    model_callback: ModelCallback
    score_callback: ScoreCallback
    metrics: Sequence[str]
    param_specs: Dict[str, GridSpec]
    fixed_params: Dict[str, Any]
    eval_dir: Path
    config: Dict[str, Any]
    cache: Optional[Dict[str, float]] = None
    save_model_outputs: bool = True

    def __post_init__(self) -> None:
        self.cache = {} if self.cache is None else self.cache
        self.param_specs = dict(self.param_specs)
        self.fixed_params = dict(self.fixed_params)
        self.metrics = list(self.metrics)

    def resolve_params(self, x: Sequence[float]) -> Dict[str, Any]:
        params = dict(self.fixed_params)
        for index, name in enumerate(self.param_specs):
            spec = self.param_specs[name]
            params[name] = snap_to_grid(float(x[index]), spec.min, spec.max, spec.step)
        return params

    def cache_key_and_path(self, params: Mapping[str, Any]) -> Tuple[str, Path]:
        metadata = dict(self.config)
        metadata.update(params)
        key = hash_payload(metadata)
        return key, self.eval_dir / f"{key}.json"

    def _identity(self, params: Mapping[str, Any], cache_key: str) -> Dict[str, Any]:
        return {
            "cache_key": cache_key,
            "params_hash": hash_payload(params),
            "run_hash": self.config.get("run_hash"),
            "objective_version": self.config.get("objective_version"),
        }

    def _load_json(self, path: Path) -> Dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["score"] = float(payload["score"])
        payload["objective"] = float(payload["objective"])
        for metric in self.metrics:
            if metric in payload:
                payload[metric] = float(payload[metric])
        return payload

    @staticmethod
    def _validate_identity(record: Mapping[str, Any], expected: Mapping[str, Any], source: Path) -> None:
        missing = [key for key in expected if key not in record]
        if missing:
            raise ValueError(f"{source} missing required objective identity keys: {missing}")
        mismatches = [f"{key}: expected {expected[key]!r}, got {record[key]!r}"
                      for key in expected if record[key] != expected[key]]
        if mismatches:
            raise ValueError(f"{source} objective identity mismatch:\n  - " + "\n  - ".join(mismatches))

    def _load_cached(self, params: Mapping[str, Any], key: str, json_path: Path) -> Dict[str, Any]:
        if not json_path.exists():
            raise FileNotFoundError(f"Objective cache artifact not found for {key}")
        expected = self._identity(params, key)
        json_record = self._load_json(json_path)
        self._validate_identity(json_record, expected, json_path)
        return {
            "cache_key": key,
            "objective": float(json_record["objective"]),
            "score": float(json_record["score"]),
            "metrics": {metric: float(json_record[metric]) for metric in self.metrics if metric in json_record},
        }

    def evaluate_params(self, params: Mapping[str, Any], *, return_model_outputs: bool = False) -> Dict[str, Any]:
        params = dict(params)
        key, json_path = self.cache_key_and_path(params)
        model_path = self.eval_dir / f"{key}_model_outputs.npz"
        if json_path.exists():
            result = self._load_cached(params, key, json_path)
            if return_model_outputs:
                if model_path.exists():
                    with np.load(model_path, allow_pickle=False) as cached:
                        result["model_outputs"] = {name: cached[name] for name in cached.files}
                else:
                    outputs = dict(self.model_callback(params))
                    arrays = {name: np.asarray(value) for name, value in outputs.items() if isinstance(value, np.ndarray)}
                    if arrays and self.save_model_outputs:
                        atomic_savez(model_path, **arrays)
                    result["model_outputs"] = outputs
            return result

        outputs = dict(self.model_callback(params))
        metric_values = {name: float(value) for name, value in self.score_callback(outputs).items()}
        score = float(sum(metric_values[name] for name in self.metrics if name in metric_values))
        objective = float(-score) if np.isfinite(score) else 1e6
        breadcrumb = {**self.config, **params, **self._identity(params, key), "objective": objective, "score": score, **metric_values}
        if not safe_write_json_once(json_path, breadcrumb):
            return self.evaluate_params(params, return_model_outputs=return_model_outputs)
        if return_model_outputs:
            arrays = {name: np.asarray(value) for name, value in outputs.items() if isinstance(value, np.ndarray)}
            if arrays and self.save_model_outputs:
                atomic_savez(model_path, **arrays)
        self._load_cached(params, key, json_path)
        result = {"cache_key": key, "objective": objective, "score": score, "metrics": metric_values}
        if return_model_outputs:
            result["model_outputs"] = outputs
        return result

    def __call__(self, x: np.ndarray) -> float:
        params = self.resolve_params(x)
        key, _ = self.cache_key_and_path(params)
        if key in self.cache:
            return self.cache[key]
        try:
            objective = float(self.evaluate_params(params)["objective"])
            self.cache[key] = objective
            return objective
        except Exception as exc:
            details = ", ".join(f"{name}={value}" for name, value in params.items())
            print(f"  ERROR at {details}: {type(exc).__name__}: {exc}")
            return 1e6


def run_differential_evolution(
    evaluator: ObjectiveEvaluator,
    *,
    maxiter: int = 50,
    popsize: int = 16,
    seed: Optional[int] = None,
    workers: int = 1,
    polish: bool = False,
    disp: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run DE or one evaluation for fixed bounds.

    Returns ``(best_params, result_dict)``. The result dictionary matches the
    fields written by the resting-state script's ``de_result.json`` artifact.
    """

    if not evaluator.param_specs:
        return dict(evaluator.fixed_params), {
            "x": [], "fun": float(evaluator.evaluate_params(evaluator.fixed_params)["objective"]),
            "nfev": 1, "nit": 0, "success": True,
            "message": "evaluated fixed default parameters",
        }

    bounds = [(spec.min, spec.max) for spec in evaluator.param_specs.values()]
    if all(spec.min == spec.max for spec in evaluator.param_specs.values()):
        params = dict(evaluator.fixed_params)
        for name, spec in evaluator.param_specs.items():
            params[name] = spec.min
        evaluation = evaluator.evaluate_params(params, return_model_outputs=True)
        return params, {
            "x": [float(params[name]) for name in evaluator.param_specs],
            "fun": float(evaluation["objective"]), "nfev": 1, "nit": 0,
            "success": True, "message": "Single-point evaluation (all bounds fixed).",
        }

    result = differential_evolution(
        evaluator,
        bounds=bounds,
        seed=seed,
        maxiter=int(maxiter),
        popsize=int(popsize),
        polish=bool(polish),
        workers=int(workers),
        updating="deferred" if int(workers) != 1 else "immediate",
        callback=TimingCallback(evaluator.param_specs),
        disp=disp,
    )
    params = dict(evaluator.fixed_params)
    for index, name in enumerate(evaluator.param_specs):
        spec = evaluator.param_specs[name]
        params[name] = snap_to_grid(float(result.x[index]), spec.min, spec.max, spec.step)
    return params, {
        "x": [float(value) for value in result.x], "fun": float(result.fun),
        "nfev": int(result.nfev), "nit": int(result.nit),
        "success": bool(result.success), "message": str(result.message),
    }
