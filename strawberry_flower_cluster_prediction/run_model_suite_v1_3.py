#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from common_regression.feature_catalog import QualityRules, apply_kept_features, greedy_correlation_filter, quality_audit
from common_regression.runner import CommonRegressionRunner
from dataset import load_dataset


def build_runner(run_id: str) -> CommonRegressionRunner:
    config = yaml.safe_load((ROOT / "config/model_suite_v1_3.yaml").read_text())
    bundle, blocks, catalog = load_dataset()
    quality = quality_audit(bundle.train, catalog, QualityRules())
    catalog = catalog.merge(quality[["feature", "status", "missing_rate"]], on="feature", validate="one_to_one")
    catalog["coverage"] = 1.0; catalog["target_spearman"] = [bundle.train[f].corr(bundle.train[bundle.spec.target], method="spearman") for f in catalog.feature]
    kept, correlations = greedy_correlation_filter(bundle.train, catalog, {"individual_history": .90, "growth": .90})
    blocks = apply_kept_features(blocks, kept)
    out = ROOT / config["artifact_root"] / run_id; out.mkdir(parents=True, exist_ok=False)
    quality.to_csv(out / "feature_quality_audit.csv", index=False); correlations.to_csv(out / "correlation_clusters.csv", index=False)
    catalog.to_csv(out / "feature_catalog.csv", index=False)
    return CommonRegressionRunner(bundle, blocks, [], config, out)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True)
    parser.add_argument("--through", choices=["features", "tuning", "validation", "final-fit", "test"], default="validation")
    args = parser.parse_args(); runner = build_runner(args.run_id); runner.select_features()
    if args.through == "features": return
    runner.tune()
    if args.through == "tuning": return
    runner.validate()
    if args.through == "validation": return
    runner.freeze(); runner.fit_final()
    if args.through == "final-fit": return
    runner.test_once()


if __name__ == "__main__": main()
