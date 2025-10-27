#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paso 7 — Comparación y registro del mejor modelo
------------------------------------------------
Lee los resultados de reports/automl_bench.csv o automl_metrics_by_split.csv.
Soporta:
- Formato "ancho" con columnas: framework, model_path, val_rmse, test_rmse, ...
- Formato "largo" con columnas: framework, split (val/test), rmse, mae, r2

Convierte a formato "ancho" si hace falta, elige el mejor modelo por la métrica
principal (default: val_rmse) y registra el modelo ganador en MLflow Model Registry.

Salidas:
- reports/model_selection.csv (resumen comparativo)
- modelo ganador en MLflow (stage Staging/Production)
"""

import os
import argparse
import pandas as pd
import mlflow


def choose_best_model(df: pd.DataFrame, metric: str = "val_rmse") -> pd.Series:
    """Devuelve la fila del modelo ganador según la métrica."""
    df = df.copy()
    if metric not in df.columns:
        raise ValueError(f"La métrica '{metric}' no existe en las columnas: {list(df.columns)}")
    df = df.dropna(subset=[metric])
    if df.empty:
        raise ValueError(f"No hay datos válidos en '{metric}'")

    # menor es mejor para RMSE/MAE, mayor para R²
    if "r2" in metric.lower():
        best = df.loc[df[metric].idxmax()]
    else:
        best = df.loc[df[metric].idxmin()]
    return best


def normalize_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acepta DF en formato "ancho" (val_rmse/test_rmse) o "largo" (split, rmse) y
    devuelve siempre formato ancho con: framework, val_rmse, test_rmse (y si están,
    val_mae/test_mae, val_r2/test_r2). Mantiene 'model_path' si ya existía.
    """
    orig_cols = list(df.columns)
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Si ya viene "ancho", lo devolvemos (ordenando columnas clave si existen)
    if "val_rmse" in df.columns or "test_rmse" in df.columns:
        cols = [c for c in [
            "framework", "model_path",
            "val_rmse", "val_mae", "val_r2",
            "test_rmse", "test_mae", "test_r2"
        ] if c in df.columns]
        return df[cols] if cols else df

    # Intento de conversión desde formato "largo" (split, rmse, mae, r2)
    needed = {"framework", "split", "rmse"}
    if not needed.issubset(set(df.columns)):
        raise KeyError(
            "No encuentro columnas esperadas. "
            f"Tengo {orig_cols}. Se necesita 'val_rmse' o las columnas {needed} para derivarlas."
        )

    # Normalizar split y pivotear
    tmp = df.loc[df["split"].notna()].copy()
    tmp["split"] = tmp["split"].str.lower().map({"val": "val", "validation": "val", "test": "test"})

    # RMSE
    rmse_wide = tmp.pivot_table(index="framework", columns="split", values="rmse", aggfunc="min")
    rmse_wide = rmse_wide.rename(columns={"val": "val_rmse", "test": "test_rmse"})

    # MAE (si existe)
    if "mae" in tmp.columns:
        mae_wide = tmp.pivot_table(index="framework", columns="split", values="mae", aggfunc="min")
        mae_wide = mae_wide.rename(columns={"val": "val_mae", "test": "test_mae"})
        rmse_wide = rmse_wide.join(mae_wide, how="left")

    # R2 (si existe)
    if "r2" in tmp.columns:
        r2_wide = tmp.pivot_table(index="framework", columns="split", values="r2", aggfunc="max")
        r2_wide = r2_wide.rename(columns={"val": "val_r2", "test": "test_r2"})
        rmse_wide = rmse_wide.join(r2_wide, how="left")

    rmse_wide = rmse_wide.reset_index()

    # Si existía model_path en el DF original (formato ancho), lo preservamos; si no, se resolverá luego por args
    if "model_path" in orig_cols:
        # Intento de merge por framework
        mp = df.loc[df["model_path"].notna(), ["framework", "model_path"]].drop_duplicates()
        rmse_wide = rmse_wide.merge(mp, on="framework", how="left")

    return rmse_wide


def fill_model_paths(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Si no hay 'model_path', intenta completar en base a los argumentos:
    --pycaret-model-path y --flaml-model-path (y opcionales).
    """
    if "model_path" in df.columns and df["model_path"].notna().any():
        return df  # ya hay rutas

    mapping = {}
    if args.pycaret_model_path:
        mapping["pycaret"] = args.pycaret_model_path
    if args.flaml_model_path:
        mapping["flaml"] = args.flaml_model_path
    if args.baseline_model_path:
        mapping["baseline"] = args.baseline_model_path
    if args.h2o_model_path:
        mapping["h2o"] = args.h2o_model_path
    if args.xgb_model_path:
        mapping["xgboost"] = args.xgb_model_path
        mapping["xgb"] = args.xgb_model_path

    if not mapping:
        # No hay forma de inferir model_path
        raise ValueError(
            "No se encontró columna 'model_path' en el CSV y no se pasaron rutas por argumentos.\n"
            "Usá, por ejemplo: --pycaret-model-path models/03A_pycaret/pycaret_best_model.pkl"
        )

    df = df.copy()
    df["framework"] = df["framework"].str.lower()
    df["model_path"] = df["framework"].map(mapping)
    return df


def main(args):
    print("⚖️  Comparando modelos en:", args.bench)
    assert os.path.exists(args.bench), f"No existe {args.bench}"
    df = pd.read_csv(args.bench)

    # 1) Normalizar tabla de métricas (ancho)
    df = normalize_metrics_table(df)

    # 2) Completar model_path si falta (vía args)
    df = fill_model_paths(df, args)

    # 3) Ordenar y guardar comparativa
    metric = args.metric
    if metric not in df.columns:
        raise ValueError(f"La métrica '{metric}' no existe en el benchmark. Columnas: {list(df.columns)}")
    df_sorted = df.sort_values(metric, ascending=("r2" not in metric.lower())).reset_index(drop=True)

    os.makedirs(args.reports, exist_ok=True)
    out_path = os.path.join(args.reports, "model_selection.csv")
    df_sorted.to_csv(out_path, index=False)
    print(f"📄 Tabla comparativa guardada en: {out_path}")

    # 4) Elegir mejor modelo
    best = choose_best_model(df_sorted, metric)
    print("\n🏆 Mejor modelo (por", metric, "):")
    print(best)

    # 5) Registrar en MLflow Model Registry
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    model_path = best.get("model_path", None)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No existe el archivo de modelo para registrar: {model_path}\n"
            "Asegurate de pasar la ruta correcta con --pycaret-model-path / --flaml-model-path, etc."
        )

    abs_model = os.path.abspath(model_path)
    model_uri = f"file://{abs_model}"
    model_name = args.model_name

    with mlflow.start_run(run_name="model_selection"):
        print(f"\n📦 Registrando modelo '{model_name}' desde {model_uri}")
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=args.stage,
            archive_existing_versions=True,
        )

        # Log params/metrics de selección
        mlflow.log_param("winner_framework", best.get("framework"))
        mlflow.log_param("stage", args.stage)
        mlflow.log_param("metric", metric)
        for k in ["val_rmse", "val_mae", "val_r2", "test_rmse", "test_mae", "test_r2"]:
            if k in best.index and pd.notna(best[k]):
                mlflow.log_metric(k, float(best[k]))

    print(f"✅ Modelo registrado en MLflow como '{model_name}' → Stage: {args.stage}")
    print("   Ver en UI con: mlflow ui --backend-store-uri", args.mlflow_uri)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paso 7 — Comparación y registro de modelos AutoML")
    ap.add_argument("--bench", default="reports/automl_bench.csv", help="CSV de benchmark (ancho o largo).")
    ap.add_argument("--reports", default="reports", help="Carpeta de salida para comparativas.")
    ap.add_argument("--metric", default="val_rmse", help="Métrica para ordenar/seleccionar (val_rmse|test_rmse|val_r2...).")
    ap.add_argument("--mlflow-uri", dest="mlflow_uri", default="mlruns", help="Tracking URI de MLflow.")
    ap.add_argument("--experiment", default="ecobici_model_selection", help="Nombre del experimento en MLflow.")
    ap.add_argument("--model-name", default="ecobici_best", help="Nombre del modelo en el Model Registry.")
    ap.add_argument("--stage", default="Staging", choices=["None", "Staging", "Production"], help="Stage destino del modelo.")

    # Rutas opcionales para completar 'model_path' según framework si no viene en el CSV
    ap.add_argument("--pycaret-model-path", default=None, help="Ruta al .pkl de PyCaret (ej.: models/03A_pycaret/pycaret_best_model.pkl)")
    ap.add_argument("--flaml-model-path", default=None, help="Ruta al .pkl de FLAML (si aplica).")
    ap.add_argument("--baseline-model-path", default=None, help="Ruta al .pkl de Baseline (si aplica).")
    ap.add_argument("--h2o-model-path", default=None, help="Ruta al .zip/.mojo de H2O (si aplica).")
    ap.add_argument("--xgb-model-path", default=None, help="Ruta al .json/.pkl de XGBoost (si aplica).")

    args = ap.parse_args()
    main(args)