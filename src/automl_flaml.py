#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoML con FLAML (regresión) — versión script (refit manual)
- Ordena por timestamp antes de seleccionar features.
- Usa solo columnas numéricas (coherente con baseline/PyCaret).
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

# ===================== utils =====================

def make_X_y_numeric(df: pd.DataFrame, y_col: str, id_col: str, ts_col: str):
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

# ---------- instanciador del mejor estimador (refit manual) ----------
def _build_estimator(name: str, cfg: dict, seed: int):
    name = (name or "").lower()
    # Normalizamos hiperparámetros comunes
    params = dict(cfg or {})
    # FLAML a veces entrega floats donde el estimador espera int
    for k in ["n_estimators", "max_depth", "min_child_samples", "num_leaves", "max_bin"]:
        if k in params and params[k] is not None:
            try:
                params[k] = int(round(params[k]))
            except Exception:
                pass

    if name in ("lgbm", "lightgbm", "lgbmregressor"):
        from lightgbm import LGBMRegressor
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        return LGBMRegressor(**params)

    if name in ("xgboost", "xgb", "xgbregressor", "xgb_limitdepth"):
        from xgboost import XGBRegressor
        # Mapeos típicos que pueden venir de FLAML
        alias = {
            "reg_alpha": "alpha",
            "reg_lambda": "lambda",
        }
        for a, b in alias.items():
            if a in params and b not in params:
                params[b] = params.pop(a)
        params.setdefault("random_state", seed)
        params.setdefault("n_estimators", 200)
        params.setdefault("n_jobs", -1)
        params.setdefault("tree_method", "hist")
        return XGBRegressor(**params)

    if name in ("rf", "random_forest", "randomforest", "randomforestregressor"):
        from sklearn.ensemble import RandomForestRegressor
        params.setdefault("random_state", seed)
        params.setdefault("n_estimators", 300)
        params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**params)

    if name in ("extra_tree", "extratrees", "extratreesregressor"):
        from sklearn.ensemble import ExtraTreesRegressor
        params.setdefault("random_state", seed)
        params.setdefault("n_estimators", 300)
        params.setdefault("n_jobs", -1)
        return ExtraTreesRegressor(**params)

    # Fallback: usar directamente el modelo ya entrenado por FLAML (sin refit full)
    return None

# ===================== main =====================

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

    # --- Prepare X,y
    Y_COL, ID_COL, TS_COL = args.y, args.id, args.ts
    Xtr, ytr = make_X_y_numeric(df_tr, Y_COL, ID_COL, TS_COL)
    Xva, yva = make_X_y_numeric(df_va, Y_COL, ID_COL, TS_COL)
    Xte = yte = None
    if not df_te.empty:
        Xte, yte = make_X_y_numeric(df_te, Y_COL, ID_COL, TS_COL)

    log_path = os.path.join(args.outdir, "flaml.log")
    automl = AutoML()

    # --------- BÚSQUEDA / SELECCIÓN DEL MEJOR -----------
    with mlflow.start_run(run_name="flaml_automl_select"):
        automl.fit(
            X_train=Xtr, y_train=ytr,
            task="regression",
            time_budget=args.time_budget,
            metric=args.metric.lower(),
            eval_method="holdout",
            X_val=Xva, y_val=yva,
            seed=args.seed,
            log_file_name=log_path,
            n_jobs=-1,
        )

        # Evaluación en VAL del modelo seleccionado (tal cual lo devolvió FLAML)
        yhat_va = np.clip(automl.predict(Xva), 0, None)
        rmse_v = mean_squared_error(yva, yhat_va, squared=False)
        mae_v  = mean_absolute_error(yva, yhat_va)
        r2_v   = r2_score(yva, yhat_va)

        mlflow.log_param("time_budget_sec", args.time_budget)
        mlflow.log_param("metric", args.metric)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_metric("val_rmse_select", rmse_v)
        mlflow.log_metric("val_mae_select",  mae_v)
        mlflow.log_metric("val_r2_select",   r2_v)
        if os.path.exists(log_path):
            mlflow.log_artifact(log_path)

    # --------- REFIT MANUAL EN TODO EL SET DISPONIBLE (TRAIN+VAL) ----------
    X_full = np.vstack([Xtr, Xva])
    y_full = np.concatenate([ytr, yva])

    best_name = automl.best_estimator
    best_cfg  = automl.best_config
    seed      = int(args.seed)

    refit_model = _build_estimator(best_name, best_cfg, seed)
    if refit_model is None:
        # fallback: usamos el modelo que dejó FLAML (ya entrenado) y lo persistimos
        refit_model = automl.model

    with mlflow.start_run(run_name="flaml_refit_full", nested=True):
        try:
            refit_model.fit(X_full, y_full)
        except Exception:
            # si el modelo ya venía entrenado (fallback), seguimos
            pass

        # Eval externamente en VAL/TEST con el modelo refiteado
        yhat_va_refit = np.clip(refit_model.predict(Xva), 0, None)
        rmse_v2 = mean_squared_error(yva, yhat_va_refit, squared=False)
        mae_v2  = mean_absolute_error(yva, yhat_va_refit)
        r2_v2   = r2_score(yva, yhat_va_refit)

        mlflow.log_metric("val_rmse", rmse_v2)
        mlflow.log_metric("val_mae",  mae_v2)
        mlflow.log_metric("val_r2",   r2_v2)

        model_path = os.path.join(args.outdir, "flaml_automl.pkl")
        joblib.dump(refit_model, model_path)
        mlflow.log_artifact(model_path)

    # --------- TEST (opcional) ----------
    rows_metrics = [{"framework": "flaml", "split": "val", "rmse": rmse_v2, "mae": mae_v2, "r2": r2_v2}]
    bench_row = {
        "framework": "flaml",
        "model_path": model_path,
        "val_rmse": rmse_v2, "val_mae": mae_v2, "val_r2": r2_v2
    }
    if Xte is not None:
        yhat_te = np.clip(refit_model.predict(Xte), 0, None)
        rmse_t = mean_squared_error(yte, yhat_te, squared=False)
        mae_t  = mean_absolute_error(yte, yhat_te)
        r2_t   = r2_score(yte, yhat_te)
        rows_metrics.append({"framework":"flaml","split":"test","rmse":rmse_t,"mae":mae_t,"r2":r2_t})
        bench_row.update({"test_rmse": rmse_t, "test_mae": mae_t, "test_r2": r2_t})

    # --------- Persistir métricas / bench ----------
    metrics_path = os.path.join(args.reports, "automl_metrics_by_split.csv")
    append_metrics_csv(metrics_path, rows_metrics)

    bench_path = os.path.join(args.reports, "automl_bench.csv")
    bench = pd.DataFrame([bench_row])
    if os.path.exists(bench_path):
        prev = pd.read_csv(bench_path)
        bench = pd.concat([prev, bench], ignore_index=True)
    bench.to_csv(bench_path, index=False)

    print("\n[OK] FLAML AutoML (con refit manual)")
    print("  Modelo:", model_path)
    print(f"  VAL   → RMSE={rmse_v2:.4f} | MAE={mae_v2:.4f} | R2={r2_v2:.4f}")
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
    ap.add_argument("--time-budget", dest="time_budget", type=int, default=600)
    ap.add_argument("--metric", default="RMSE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", dest="mlflow_uri", default="mlruns")
    ap.add_argument("--experiment", default="ecobici_automl_flaml")
    # flag “refine-time” se ignora en esta versión (refit manual):
    ap.add_argument("--refine-time", dest="refine_time", type=int, default=0)
    args = ap.parse_args()
    main(args)