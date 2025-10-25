#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoML con FLAML (regresión) — versión script
- Ordena por timestamp antes de seleccionar features (evita el KeyError de ts_local).
- Usa solo columnas numéricas (coherente con baseline y PyCaret-script).
- Guarda modelo en models/06_flaml/, métricas en reports/, y benchmark en reports/automl_bench.csv
- Loggea en MLflow (tracking local por defecto: ./mlruns)
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from flaml import AutoML
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow


# ----------------------------- utils -----------------------------
def make_X_y_numeric(df: pd.DataFrame, y_col: str, id_col: str, ts_col: str):
    """Prepara X,y:
    - Ordena por ts_col si existe (coerce a datetime)
    - Deja solo numéricas (excluye y, id, ts) y rellena NaN=0.0
    """
    df = df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    y = pd.to_numeric(df[y_col], errors="coerce").astype(float).values
    X = df.drop(columns=[c for c in [y_col, id_col, ts_col] if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0.0)
    return X, y


def append_metrics_csv(path_csv: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df = pd.read_csv(path_csv) if os.path.exists(path_csv) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(path_csv, index=False)
    return df


# ----------------------------- main -----------------------------
def main(args):
    # --- Paths & dirs
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)

    # --- MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # --- Load splits
    assert os.path.exists(args.train), f"No existe {args.train}"
    assert os.path.exists(args.val),   f"No existe {args.val}"
    df_tr = pd.read_parquet(args.train)
    df_va = pd.read_parquet(args.val)
    df_te = pd.read_parquet(args.test) if (args.test and os.path.exists(args.test)) else pd.DataFrame()

    # --- Prepare X,y (num-only, orden temporal previo)
    Y_COL, ID_COL, TS_COL = args.y, args.id, args.ts
    Xtr, ytr = make_X_y_numeric(df_tr, Y_COL, ID_COL, TS_COL)
    Xva, yva = make_X_y_numeric(df_va, Y_COL, ID_COL, TS_COL)
    Xte = yte = None
    if not df_te.empty:
        Xte, yte = make_X_y_numeric(df_te, Y_COL, ID_COL, TS_COL)

    # --- Logging FLAML a archivo (evita saturar consola / barras)
    log_path = os.path.join(args.outdir, "flaml.log")

    automl = AutoML()
    with mlflow.start_run(run_name="flaml_automl"):
        automl.fit(
            X_train=Xtr, y_train=ytr,
            task="regression",
            time_budget=args.time_budget,   # segundos
            metric=args.metric.lower(),     # flaml usa lowercase
            eval_method="holdout",
            X_val=Xva, y_val=yva,
            seed=args.seed,
            log_file_name=log_path,
            n_jobs=-1,
        )

        # --- Validación
        yhat_va = np.clip(automl.predict(Xva), 0, None)
        rmse_v = mean_squared_error(yva, yhat_va, squared=False)
        mae_v  = mean_absolute_error(yva, yhat_va)
        r2_v   = r2_score(yva, yhat_va)

        mlflow.log_param("time_budget_sec", args.time_budget)
        mlflow.log_param("metric", args.metric)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_metric("val_rmse", rmse_v)
        mlflow.log_metric("val_mae",  mae_v)
        mlflow.log_metric("val_r2",   r2_v)

        # --- Guardar modelo
        model_path = os.path.join(args.outdir, "flaml_automl.pkl")
        joblib.dump(automl, model_path)
        mlflow.log_artifact(model_path)
        if os.path.exists(log_path):
            mlflow.log_artifact(log_path)

    # --- Test (opcional)
    rows_metrics = [{"framework": "flaml", "split": "val", "rmse": rmse_v, "mae": mae_v, "r2": r2_v}]
    bench_row = {
        "framework": "flaml",
        "model_path": model_path,
        "val_rmse": rmse_v, "val_mae": mae_v, "val_r2": r2_v
    }

    if Xte is not None:
        yhat_te = np.clip(automl.predict(Xte), 0, None)
        rmse_t = mean_squared_error(yte, yhat_te, squared=False)
        mae_t  = mean_absolute_error(yte, yhat_te)
        r2_t   = r2_score(yte, yhat_te)
        rows_metrics.append({"framework":"flaml","split":"test","rmse":rmse_t,"mae":mae_t,"r2":r2_t})
        bench_row.update({"test_rmse": rmse_t, "test_mae": mae_t, "test_r2": r2_t})

    # --- Guardar métricas
    metrics_path = os.path.join(args.reports, "automl_metrics_by_split.csv")
    append_metrics_csv(metrics_path, rows_metrics)

    bench_path = os.path.join(args.reports, "automl_bench.csv")
    bench = pd.DataFrame([bench_row])
    if os.path.exists(bench_path):
        prev = pd.read_csv(bench_path)
        bench = pd.concat([prev, bench], ignore_index=True)
    bench.to_csv(bench_path, index=False)

    print("\n[OK] FLAML AutoML")
    print("  Modelo:", model_path)
    print(f"  VAL   → RMSE={rmse_v:.4f} | MAE={mae_v:.4f} | R2={r2_v:.4f}")
    if "test_rmse" in bench_row:
        print(f"  TEST  → RMSE={bench_row['test_rmse']:.4f} | MAE={bench_row['test_mae']:.4f} | R2={bench_row['test_r2']:.4f}")
    print("  Métricas:", metrics_path)
    print("  Benchmark:", bench_path)
    print("  Log FLAML:", log_path if os.path.exists(log_path) else "(no generado)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AutoML FLAML (regresión) — Ecobici")
    ap.add_argument("--train", default="data/splits/train.parquet")
    ap.add_argument("--val",   default="data/splits/val.parquet")
    ap.add_argument("--test",  default="data/splits/test.parquet")
    ap.add_argument("--outdir",  default="models/06_flaml")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--y",   dest="y",  default="num_bikes_available")
    ap.add_argument("--id",  dest="id", default="station_id")
    ap.add_argument("--ts",  dest="ts", default="ts_local")
    ap.add_argument("--time-budget", dest="time_budget", type=int, default=600)  # segundos
    ap.add_argument("--metric", default="RMSE")   # RMSE/MAE/R2 (se pasa lowercase a flaml)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", dest="mlflow_uri", default="mlruns")
    ap.add_argument("--experiment", default="ecobici_automl_flaml")
    args = ap.parse_args()
    main(args)