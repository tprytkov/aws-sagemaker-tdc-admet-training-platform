"""Generate locked-test-safe real-data audit and validation baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.config import load_chemprop_config  # noqa: E402
from admet_platform.chemprop.preflight import (  # noqa: E402
    build_real_data_preflight_audit,
    run_real_validation_baselines,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    regression_config = load_chemprop_config("configs/chemprop/multitask_admet_regression.yaml")
    bbb_config = load_chemprop_config("configs/chemprop/bbb_martins.yaml")
    audit, regression, bbb = build_real_data_preflight_audit(regression_config, bbb_config)
    write_json(output / "real_data_contract_audit.json", audit)
    if not args.skip_baselines:
        baselines = run_real_validation_baselines(regression_config, regression, bbb)
        write_json(output / "validation_baselines.json", baselines)


if __name__ == "__main__":
    main()
