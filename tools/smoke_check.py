#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smoke check — verificación rápida del repo (sin re-entrenar)
- Chequea estructura mínima de carpetas
- Verifica archivos base curados/splits
- Lee benchmark y localiza el mejor modelo
- Hace una predicción mínima con 1 fila de train (solo numéricas)
- Revisa symlinks o archivos regulares de predictions/latest y tiles/latest
"""

import os
import sys
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd


# ----------------------------- Config -----------------------------
# Rutas relativas al root del repo
BENCH_CSV   = "reports/automl_bench.csv"
CURATED     = "data/curated/ecobici_model_ready.parquet"
TRAIN_SPLIT = "data/splits/train.parquet"
STATIONS    = "data/curated/station_information.parquet"

REQUIRED_DIRS = [
    "data/raw", "data/curated", "data/splits",
    "models", "reports", "predictions", "tiles", "mlruns"
]

Y_COL  = "num_bikes_available"
ID_COL = "station_id"
TS_COL = "ts_local"


# ----------------------------- Utils -----------------------------
def fail(msg, code=1):
    print(f"❌ {msg}")
    sys.exit(code)

def warn(msg):
    print(f"⚠️  {msg}")

def ok(msg):
    print(f"✅ {msg}")

def _looks_like_json_dict(s: str) -> bool:
    s = str(s).strip()
    return s.startswith("{") and s.endswith("}")

def expand_json_like_columns(df: pd.DataFrame, exclude=None) -> pd.DataFrame:
    """Expande columnas object que parezcan JSON (dict) a numéricas."""
    exclude = exclude or []
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

def choose_best_from_bench(bench_csv: str) -> dict:
    if not os.path.exists(bench_csv):
        fail(f"No existe {bench_csv}. Corré los notebooks/py de AutoML primero.")
    bench = pd.read_csv(bench_csv)
    if bench.empty:
        fail(f"{bench_csv} está vacío.")
    if "val_rmse" not in bench.columns or "model_path" not in bench.columns:
        fail(f"{bench_csv} debe contener columnas val_rmse y model_path.")
    # Ganador = menor val_rmse
    row = bench.sort_values("val_rmse", ascending=True).iloc[0]
    return row.to_dict()


# ----------------------------- Main -----------------------------
def main():
    print("🔎 Smoke check — verificación rápida del repo (sin re-entrenar)\n")

    # 1) Carpetas mínimas
    missing = [d for d in REQUIRED_DIRS if not os.path.isdir(d)]
    if missing:
        fail(f"Faltan directorios: {missing}")
    ok("Estructura de carpetas OK")

    # 2) Archivos clave
    if not os.path.exists(CURATED):
        fail(f"Falta {CURATED} (ejecutá los PASO 1/2 para construirlo).")
    if not os.path.exists(TRAIN_SPLIT):
        fail(f"Falta {TRAIN_SPLIT} (corré el PASO 2 de splits).")
    if not os.path.exists(STATIONS):
        warn(f"No se encontró {STATIONS}. Solo afectará tiles; el resto puede pasar.")
    ok("Archivos de datos base OK")

    # 3) Benchmark y pickle ganador
    best = choose_best_from_bench(BENCH_CSV)
    model_path_rel = str(best["model_path"])
    model_path = os.path.abspath(model_path_rel) if not os.path.isabs(model_path_rel) else model_path_rel

    if not os.path.exists(model_path):
        fail(f"El pickle del mejor modelo no existe: {model_path}\n"
             f"Revisá rutas en {os.path.abspath(BENCH_CSV)} (se recomienda que sean relativas al root).")
    ok(f"Benchmark OK — mejor: {best.get('framework')} → {model_path_rel}")

    # 4) Predicción mínima con 1 fila de train (sin correr predict_batch)
    df_train = pd.read_parquet(TRAIN_SPLIT)
    if TS_COL in df_train.columns:
        df_train[TS_COL] = pd.to_datetime(df_train[TS_COL], errors="coerce")
        df_train = df_train.dropna(subset=[TS_COL]).sort_values(TS_COL)

    # Elegimos una fila que tenga lags si existen (y_lag1, y_lag2, y_ma3)
    cols_needed = [c for c in ["y_lag1", "y_lag2", "y_ma3"] if c in df_train.columns]
    sample = df_train.dropna(subset=cols_needed).tail(1) if cols_needed else df_train.tail(1)
    if sample.empty:
        fail("No se pudo seleccionar una fila de muestra desde train para testear predicción.")

    # Esquema de entrenamiento (solo numéricas, sin id/ts/target)
    drop_cols = [c for c in [Y_COL, ID_COL, TS_COL] if c in df_train.columns]
    X_schema = make_numeric_features(df_train.drop(columns=drop_cols, errors="ignore"))
    train_cols = list(X_schema.columns)

    # Construir X de test desde el sample, reindex al esquema
    X_const = make_numeric_features(sample.drop(columns=drop_cols, errors="ignore"))
    X_test = X_const.reindex(columns=train_cols, fill_value=0.0)

    # Cargar modelo y predecir
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        fail("El mejor modelo no expone .predict().")
    try:
        yhat = float(model.predict(X_test)[0])
        ok(f"Predicción de prueba OK (ŷ={yhat:.4f})")
    except Exception as e:
        fail(f"No se pudo predecir con el mejor modelo: {e}")

    # 5) latest.parquet (predicciones) — aceptar symlink o archivo regular
    pred_latest = Path("predictions/latest.parquet")
    if pred_latest.exists():
        if pred_latest.is_symlink():
            tgt = os.readlink(pred_latest)
            if os.path.exists(tgt):
                ok(f"Predictions OK → {pred_latest} -> {tgt}")
            else:
                warn(f"Symlink roto: {pred_latest} -> {tgt}")
        else:
            ok(f"Predictions OK → {pred_latest} (archivo regular)")
    else:
        warn("No existe predictions/latest.parquet (se crea con `make predict`).")

    # 6) tiles/latest.parquet — aceptar symlink o archivo regular
    tiles_latest = Path("tiles/latest.parquet")
    if tiles_latest.exists():
        if tiles_latest.is_symlink():
            tgt = os.readlink(tiles_latest)
            if os.path.exists(tgt):
                ok(f"Tiles OK → {tiles_latest} -> {tgt}")
            else:
                warn(f"Symlink roto: {tiles_latest} -> {tgt}")
        else:
            ok(f"Tiles OK → {tiles_latest} (archivo regular)")
    else:
        warn("No existe tiles/latest.parquet (se crea con `make tiles`).")

    # 7) mlruns
    if os.path.isdir("mlruns"):
        ok("MLflow store (./mlruns) presente")
    else:
        warn("No se encontró ./mlruns (se crea al loggear runs).")

    print("\n🎉 Smoke check completado: repo listo para usar.")


if __name__ == "__main__":
    main()