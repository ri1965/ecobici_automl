#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoML con PyCaret (regresión) — versión script
- Preprocesamiento alineado con predict_batch.py:
  * expand_json_like_columns + onehot_low_card + seleccionar numéricas.
- Validación temporal (fold_strategy="timeseries", sin shuffle).
- Guarda modelo en models/03A_pycaret/, y métricas en reports/automl_metrics_by_split.csv
- Loggea en MLflow (tracking local por defecto: ./mlruns)
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pycaret.regression import setup, compare_models, finalize_model, save_model, predict_model, pull
import mlflow

# ----------------------------- utils (mismas que predict_batch.py) -----------------------------
def _looks_like_json_dict(s: str) -> bool:
    s = str(s).strip()
    return s.startswith("{") and s.endswith("}")

def expand_json_like_columns(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """Detecta columnas object que parecen JSON (dict) y las expande a columnas numéricas."""
    df2 = df.copy()
    obj_cols = [c for c in df2.columns if c not in exclude and df2[c].dtype == "object"]
    for col in obj_cols:
        sample = df2[col].dropna().astype(str).head(50)
        if len(sample) == 0:
            continue
        if sample.map(_looks_like_json_dict).mean() >= 0.6:
            def _parse(x):
                try:
                    d = json.loads(x) if isinstance(x, str) else x
                    return d if isinstance(d, dict) else {}
                except Exception:
                    return {}
            exp = df2[col].apply(_parse).apply(pd.Series)
            if exp is not None and exp.shape[1] > 0:
                exp = exp.add_prefix(f"{col}_")
                for c in exp.columns:
                    exp[c] = pd.to_numeric(exp[c], errors="coerce").fillna(0.0)
                df2 = pd.concat([df2.drop(columns=[col]), exp], axis=1)
    return df2

def onehot_low_card(dfX: pd.DataFrame, max_card: int = 20) -> pd.DataFrame:
    low = [c for c in dfX.columns if dfX[c].dtype == "object" and dfX[c].nunique(dropna=True) <= max_card]
    if low:
        dfX = pd.get_dummies(dfX, columns=low, drop_first=True)
    return dfX

def make_numeric_features(dfX: pd.DataFrame) -> pd.DataFrame:
    dfX = expand_json_like_columns(dfX, exclude=[])
    dfX = onehot_low_card(dfX)
    return dfX.select_dtypes(include=["number"]).fillna(0.0)

def make_numeric_dataset(df: pd.DataFrame, y_col: str, id_col: str, ts_col: str) -> pd.DataFrame:
    """Devuelve target + features numéricas tras la misma expansión que en inferencia."""
    df = df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # Quitamos id/ts/target para crear X, expandimos/one-hot, y volvemos a unir y.
    drop_cols = [c for c in [y_col, id_col, ts_col] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = make_numeric_features(X)
    data_num = df[[y_col]].join(X)
    return data_num

def eval_on(df: pd.DataFrame, model, y_col: str):
    pred = predict_model(model, data=df.copy())
    y_true = pd.to_numeric(df[y_col], errors="coerce").astype(float).values
    y_hat  = pd.to_numeric(pred["prediction_label"], errors="coerce").astype(float).values
    rmse = mean_squared_error(y_true, y_hat, squared=False)
    mae  = mean_absolute_error(y_true, y_hat)
    r2   = r2_score(y_true, y_hat)
    return rmse, mae, r2

def append_metrics_csv(path_csv: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df = pd.read_csv(path_csv) if os.path.exists(path_csv) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(path_csv, index=False)
    return df

# ----------------------------- main -----------------------------
def main(args):
    # --- Paths & config
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)

    # --- MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # --- Cargar splits
    df_tr = pd.read_parquet(args.train)
    df_va = pd.read_parquet(args.val)
    df_te = pd.read_parquet(args.test) if (args.test and os.path.exists(args.test)) else pd.DataFrame()

    # --- Datasets numéricos (alineados a predict_batch)
    Y_COL, ID_COL, TS_COL = args.y, args.id, args.ts
    data_tr = make_numeric_dataset(df_tr, Y_COL, ID_COL, TS_COL)
    data_va = make_numeric_dataset(df_va, Y_COL, ID_COL, TS_COL)
    data_te = make_numeric_dataset(df_te, Y_COL, ID_COL, TS_COL) if not df_te.empty else pd.DataFrame()

    # --- Setup (validación temporal)
    s = setup(
        data=data_tr,
        target=Y_COL,
        session_id=args.seed,
        fold_strategy="timeseries",
        fold=args.folds,
        data_split_shuffle=False,
        fold_shuffle=False,
        ignore_features=None,   # ya pasamos features listas
        normalize=False,
        log_experiment=True,
        experiment_name=args.experiment,
        html=False,
        verbose=False,
    )

    # --- AutoML PyCaret
    best = compare_models(sort=args.metric)
    try:
        compare_results = pull().copy()
        compare_results.to_csv(os.path.join(args.reports, "pycaret_compare_results.csv"), index=False)
    except Exception:
        compare_results = None

    best_final = finalize_model(best)

    # --- Guardar modelo
    save_base = os.path.join(args.outdir, "pycaret_best_model")
    _ = save_model(best_final, save_base)  # crea ..._pycaret_best_model.pkl
    model_path = save_base + ".pkl"

    # --- Evaluación externa (VAL y TEST si existe) usando los data_* ya preprocesados
    rmse_v, mae_v, r2_v = eval_on(data_va, best_final, Y_COL)
    rows = [{"framework": "pycaret", "split": "val", "rmse": rmse_v, "mae": mae_v, "r2": r2_v}]
    if not data_te.empty:
        rmse_t, mae_t, r2_t = eval_on(data_te, best_final, Y_COL)
        rows.append({"framework": "pycaret", "split": "test", "rmse": rmse_t, "mae": mae_t, "r2": r2_t})

    metrics_path = os.path.join(args.reports, "automl_metrics_by_split.csv")
    append_metrics_csv(metrics_path, rows)

    # --- MLflow logging
    
    with mlflow.start_run(run_name="pycaret_automl_final", nested=True):
        mlflow.log_param("folds", args.folds)
        mlflow.log_param("metric_sort", args.metric)
        mlflow.log_param("seed", args.seed)
        mlflow.log_metric("val_rmse", rmse_v)
        mlflow.log_metric("val_mae", mae_v)
        mlflow.log_metric("val_r2", r2_v)
        if not data_te.empty:
            mlflow.log_metric("test_rmse", rmse_t)
            mlflow.log_metric("test_mae", mae_t)
            mlflow.log_metric("test_r2", r2_t)
        mlflow.log_artifact(model_path)
        if compare_results is not None:
            mlflow.log_artifact(os.path.join(args.reports, "pycaret_compare_results.csv"))
        mlflow.log_artifact(metrics_path)

    print("\n[OK] PyCaret AutoML")
    print("  Modelo:", model_path)
    print(f"  VAL   → RMSE={rmse_v:.4f} | MAE={mae_v:.4f} | R2={r2_v:.4f}")
    if not data_te.empty:
        print(f"  TEST  → RMSE={rmse_t:.4f} | MAE={mae_t:.4f} | R2={r2_t:.4f}")
    print("  Métricas acumuladas:", metrics_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AutoML PyCaret (regresión) — Ecobici")
    ap.add_argument("--train", default="data/splits/train.parquet")
    ap.add_argument("--val",   default="data/splits/val.parquet")
    ap.add_argument("--test",  default="data/splits/test.parquet")
    ap.add_argument("--outdir",  default="models/03A_pycaret")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--y",   dest="y",  default="num_bikes_available")
    ap.add_argument("--id",  dest="id", default="station_id")
    ap.add_argument("--ts",  dest="ts", default="ts_local")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--metric", default="RMSE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", dest="mlflow_uri", default="mlruns")
    ap.add_argument("--experiment", default="ecobici_pycaret_automl")
    args = ap.parse_args()
    main(args)