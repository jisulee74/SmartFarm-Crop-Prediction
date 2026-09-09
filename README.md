# SmartFarm-Crop-Prediction

Reproducible regression pipelines and experiment artifacts for predicting crop growth indicators from Smart Farm Korea data.

## Projects

| Directory | Task |
|---|---|
| `common_regression/` | Shared data contracts, preprocessing, model adapters, metrics, temporal validation, and selection utilities. |
| `strawberry_flower_cluster_prediction/` | Predicts the next strawberry flower-cluster count from growth history and crop context. |
| `tomato_flower_count_prediction/` | Predicts the next total tomato flower count across observed trusses using flower history, truss structure, crop context, and recent indoor environment data. |
| `strawberry_fruit_set_prediction/` | Predicts the next fruit-set count independently for the first, second, and third strawberry trusses. |

## Candidate models

The experiments compare Poisson Regression, Random Forest, CatBoost, MLP, and TabM. Persistence is evaluated as a separate baseline. The tomato flower-count and strawberry fruit-set experiments additionally include Temporal Fusion Transformer (TFT).

## Evaluation protocol

- Input features only use information available at or before the prediction date.
- Feature selection and hyperparameter tuning use training and validation data.
- Test data are reserved for evaluation after the configuration is fixed.
- Reported regression metrics are RMSE, MAE, R-squared, and concordance correlation coefficient (CCC).

## Repository layout

Each prediction project contains its pipeline code, configuration, required processed data, and retained validation/test artifacts. Historical plans and data-quality audits are stored under the relevant project's `docs/` or `data_audits/` directory.

## Environment

Python 3.11 is recommended. Install the shared requirements with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r common_regression/requirements-v1.3.txt
```

Database credentials are not stored in this repository. Database extraction commands read connection settings from environment variables such as `FARMSTOM_DB_HOST`, `FARMSTOM_DB_USER`, and `FARMSTOM_DB_PASSWORD`.

## Rebuilding the tomato environment cache

The following rebuildable raw cache is intentionally excluded because it exceeds GitHub's 100 MB single-file limit:

```text
tomato_flower_count_prediction/data/raw/environment.parquet
```

With valid database environment variables, regenerate it through the tomato pipeline:

```bash
python tomato_flower_count_prediction/run_pipeline.py \
  --run-id total_flower_count_v1_sample_key_retry3 \
  --rebuild-cache
```

The retained processed train/validation/test datasets already contain the aggregated recent-environment features, so existing experiment results can be inspected without rebuilding this raw cache.

## Retained experiment results

- Strawberry flower-cluster count: `strawberry_flower_cluster_prediction/artifacts/final_v1_3/`
- Tomato total flower count: `tomato_flower_count_prediction/artifacts/total_flower_count_v1_sample_key_retry3/`
- Strawberry fruit set: `strawberry_fruit_set_prediction/facility_holdout_fixed13_with_tft_v1/artifacts/`

## Data notice

This repository contains derived datasets and experiment artifacts originating from Smart Farm Korea data. Access and redistribution must follow the applicable data-use agreement and organizational policy.


## Observation dashboard and experiment catalog

- [Web dashboard](flower_fruit_prediction/README.md): eight tomato flower / strawberry fruit-set observation charts with facility, crop-cycle and individual filters.
- [Experiment catalog](ax_catalog_experiments/README.md): category definitions, staged experiments, CSV/Excel exports and reproducible run records.
- [Excel report](ax_catalog_experiments/outputs/AX_Experiment_Report.xlsx)
- [Individual observation CSVs](ax_catalog_experiments/outputs/timeseries/)
- [Power BI design](ax_catalog_experiments/POWER_BI_DESIGN.md)

Run the dashboard with `python flower_fruit_prediction/server.py`, then open the printed localhost URL. This serves the CSV files bundled in the dashboard directory. It does not publish a hosted site automatically.

The repository keeps the historical AXData contents at its root. The `AXData` compatibility namespace supports existing commands such as `python -m AXData.ax_catalog_experiments report` from the repository root.

### Experiment interpretation

Archived model runs completed, but some predictions exhibit numerical instability (near-zero training variance amplified by scaling). These results require correction and re-evaluation before final model conclusions; execution success is not a data-quality or model-validity guarantee. Observation CSVs contain actual observations, not these predictions. Full database inventory verification remains separate from the completed local-cohort experiments.

Run JSONs, predictions, prepared datasets and reports are retained. Rebuildable model weights, verbose logs, Python caches, credentials and the oversized raw environment cache are excluded. Existing database refresh commands require environment configuration; no credentials are bundled.
