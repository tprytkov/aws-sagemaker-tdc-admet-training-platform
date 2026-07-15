"""Evaluation, comparison, and model-card generation utilities."""

from admet_platform.evaluation.compare import compare_runs, evaluate_model_runs
from admet_platform.evaluation.loaders import discover_run_dirs, load_model_run, load_model_runs
from admet_platform.evaluation.schemas import ComparisonOptions, ComparisonResult, ModelRun

__all__ = [
    "ComparisonOptions",
    "ComparisonResult",
    "ModelRun",
    "compare_runs",
    "discover_run_dirs",
    "evaluate_model_runs",
    "load_model_run",
    "load_model_runs",
]
