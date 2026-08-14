"""Train a Chemprop endpoint from verified train/validation splits only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.applicability import applicability_domain  # noqa: E402
from admet_platform.chemprop.baselines import run_classification_baselines, run_regression_baselines  # noqa: E402
from admet_platform.chemprop.config import load_chemprop_config  # noqa: E402
from admet_platform.chemprop.data import load_verified_development_data  # noqa: E402
from admet_platform.chemprop.training import train_graph_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph-only Chemprop without opening locked test data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--accelerator", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    config = load_chemprop_config(args.config)
    verified = load_verified_development_data(config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    verification = {
        "protocol": "internal_leakage_controlled",
        "loaded_splits": ["train", "validation"],
        "locked_test_opened": False,
        "split_manifest_sha256": verified.split_manifest_sha256,
        "expected_split_hashes": verified.expected_hashes,
        "verified_split_hashes": verified.actual_hashes,
        "train_rows": len(verified.train),
        "validation_rows": len(verified.validation),
        "observed_label_counts": verified.label_counts,
    }
    (output / "data_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if config.tasks:
        applicability_root = output / "validation_applicability_domain"
        applicability_root.mkdir(exist_ok=True)
        for endpoint in config.tasks:
            train_observed = verified.train[endpoint].notna()
            validation_observed = verified.validation[endpoint].notna()
            applicability_domain(
                verified.train.loc[train_observed, "canonical_smiles"].astype(str).tolist(),
                verified.validation.loc[validation_observed, "canonical_smiles"].astype(str).tolist(),
                float(config.raw["applicability_domain"]["similarity_threshold"]),
            ).to_csv(applicability_root / f"{endpoint}.csv", index=False)
    else:
        applicability_domain(
            verified.train["canonical_smiles"].astype(str).tolist(),
            verified.validation["canonical_smiles"].astype(str).tolist(),
            float(config.raw["applicability_domain"]["similarity_threshold"]),
        ).to_csv(output / "validation_applicability_domain.csv", index=False)
    if not args.skip_baselines:
        if config.tasks:
            baselines = {}
            for endpoint in config.tasks:
                train = verified.train.loc[
                    verified.train[endpoint].notna(), ["canonical_smiles", endpoint]
                ].rename(columns={endpoint: "target"})
                validation = verified.validation.loc[
                    verified.validation[endpoint].notna(), ["canonical_smiles", endpoint]
                ].rename(columns={endpoint: "target"})
                baselines[endpoint] = run_regression_baselines(train, validation)
        else:
            baselines = run_classification_baselines(verified.train, verified.validation)
        (output / "baseline_validation_metrics.json").write_text(
            json.dumps(baselines, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    result = train_graph_model(
        config, verified.train, verified.validation, output, seed=args.seed,
        accelerator_override=args.accelerator,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
