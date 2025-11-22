# BayesiPy Regression Benchmarks

This folder contains a unified benchmark for **large-scale regression** on tabular / time–series datasets using the uncertainty methods implemented in **BayesiPy**.

Datasets:

- `Airline`  
- `Year`  
- `Taxi`

Methods:

- `map` – deterministic MLP + homoscedastic noise (estimated from residuals)
- `fmgp` – Fixed-Mean Gaussian Process head on top of a pretrained MLP
- `lla` – Last-Layer Laplace approximation
- `ella` – Accelerated Laplace (ELLA)
- `valla` – Variational Laplace (VaLLA)
- `mfvi` – Mean-field variational inference
- `sngp` – Spectral-normalized Gaussian process

The main entry point is:

```text
benchmarks/regression/regression_unified.py
```

---

## 1. Environment & installation

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate

# Install BayesiPy and dependencies
pip install -e ".[dev]"
pip install mlflow
```

If you use an MLflow database backend (recommended):

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

This will create/use `mlflow.db` in the current directory.

---

## 2. Script overview

`regression_unified.py` does the following:

1. Parses command line arguments and/or a JSON config file.
2. Loads the selected dataset and pretrained MLP:
   - `Airline_Dataset` / `Airline_MLP`
   - `Year_Dataset`    / `Year_MLP`
   - `Taxi_Dataset`    / `Taxi_MLP`
3. Builds the chosen uncertainty method (`map`, `fmgp`, `lla`, `ella`, `valla`, `mfvi`, `sngp`).
4. Trains/fits the estimator (if applicable).
5. Evaluates on the test set using `bayesipy.utils.metrics.Regression` and `score`.
6. Saves metrics as CSV and logs everything to **MLflow**:
   - parameters (filtered per method),
   - metrics (RMSE, NLL, CRPS, timings, …),
   - artifacts (CSV results + config JSON).

All models are assumed to expose a `.predict(x)` method returning `(Fmean, Fvar)` in **original data space** (the wrapper for `map` takes care of undoing target normalization).

---

## 3. CLI arguments

Minimal core arguments:

- `--dataset {airline,year,taxi}`
- `--method {map,fmgp,lla,ella,valla,mfvi,sngp}`
- `--seed SEED` 
- `--batch_size N` (default: `100`)
- `--output PATH` (default: `benchmarks/regression/results`)
- `--verbose` (flag)

Each method has its own hyperparameters, all exposed via CLI:

- FMGP:
  - `--fmgp_iterations`, `--fmgp_lr`, `--fmgp_noise_variance`
  - `--fmgp_kernel`
  - `--fmgp_inducing_locations`, `--fmgp_num_inducing`
- LLA:
  - `--lla_subset`, `--lla_hessian`
- ELLA:
  - `--ella_subsample`, `--ella_n_eigenvalues`, `--ella_prior`
- VaLLA:
  - `--valla_inducing_locations`, `--valla_num_inducing`
  - `--valla_noise_variance`, `--valla_iterations`, `--valla_lr`
- MFVI:
  - `--mfvi_iterations`, `--mfvi_prior`, `--mfvi_n_samples`
  - `--mfvi_noise_variance`
- SNGP:
  - `--sngp_kernel_scale`, `--sngp_n_random_features`
  - `--sngp_gp_output_bias`, `--sngp_layer_norm_eps`
  - `--sngp_n_power_iterations`
  - `--sngp_scale_random_features` (flag)
  - `--sngp_normalize_input` (flag)
  - `--sngp_gp_cov_momentum`, `--sngp_gp_cov_ridge_penalty`
  - `--sngp_iterations`, `--sngp_lr`, `--sngp_weight_decay`
  - `--sngp_noise_variance`

Run `python regression_unified.py -h` for the full list.

---

## 4. Running single experiments

From the repo root:

```bash
python benchmarks/regression/regression_unified_mlflow.py   --dataset year   --method lla   --seed 0   --batch_size 128   --output benchmarks/regression/results   --verbose
```

This will:

- train/evaluate `lla` on the Year dataset with seed 0,
- print all metrics as a one-row table,
- write:

  ```text
  benchmarks/regression/results/year/lla_0.csv
  benchmarks/regression/results/year/config_year_lla_seed0.json
  ```

- log a single run to the MLflow experiment `bayesipy_regression`  
  with run name `year_lla_seed0`.

---

## 5. Using JSON config files

Instead of specifying all arguments on the command line, you can create a JSON config and pass it with `--config`.

Example: `benchmarks/regression/configs/year_lla_seed0.json`

```json
{
  "verbose": true,
  "output": "benchmarks/regression/results",
  "dataset": "year",
  "method": "lla",
  "batch_size": 128,
  "seed": 0,

  "lla_subset": "last_layer",
  "lla_hessian": "kron",
}
```

Run:

```bash
python benchmarks/regression/regression_unified_mlflow.py   --config benchmarks/regression/configs/year_lla_seed0.json
```

Any CLI flags you pass alongside `--config` can override fields in the JSON if needed.

---

## 6. MLflow logging details

- **Experiment name**: `bayesipy_regression`
- **Runs**:
  - Parent runs: one per `(dataset, method)` when using multiple seeds.
  - Child runs: one per seed (nested under the parent run).
- **Parameters**:
  - Always: core parameters (`dataset`, `method`, `batch_size`, `seed`, `output`, `verbose`).
  - Method-specific: only the parameters matching the method prefix are logged:
    - for `lla`: `lla_*`
    - for `sngp`: `sngp_*`
    - etc.
- **Metrics**:
  - Regression metrics from `score` (RMSE, NLL, calibration metrics, CRPS, …).
  - `train_time` and `test_time`.
  - For parent run (multi-seed): mean and std of each numeric metric across seeds.
- **Artifacts**:
  - CSV metrics file per run: `<output>/<dataset>/<method>_<seed>.csv`
  - JSON config used for the run: `config_<dataset>_<method>_seed<seed>.json`

### Viewing MLflow UI

If you set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

In the **Experiments** list, select `bayesipy_regression`.

To make the table more informative, enable columns such as:

- `params.dataset`
- `params.method`
- `params.seed`
- `metrics.RMSE`, `metrics.NLL`, `metrics.CRPS`, etc.
- `metrics.RMSE_mean`, `metrics.RMSE_std` (for parent runs).

---

## 7. Notes & tips

- GPU vs CPU is selected automatically via `torch.cuda.is_available()`.  
- The script uses `torch.float64` by default; you can change that inside the script if you want `float32` for speed.
- The MAP baseline uses a `MapWrapper` that:
  - fits the deterministic MLP normally,
  - estimates a single homoscedastic noise variance from training residuals,
  - returns `(Fmean, Fvar)` in original target space.
- If you update hyperparameters or add new methods, prefer:
  - adding CLI flags in `get_parser()`, and
  - extending `build_estimator()` and `fit()` accordingly,
  so everything is still reproducible and tracked.
