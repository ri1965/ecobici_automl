#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paso 7 — Comparación y registro del mejor modelo
------------------------------------------------
Lee los resultados de reports/automl_bench.csv o automl_metrics_by_split.csv,
compara frameworks (baseline RF, PyCaret, FLAML, etc.) por métrica principal
y registra el mejor modelo en MLflow Model Registry.

Salida:
- reports/model_selection.csv (resumen comparativo)
- modelo ganador marcado como 'Staging' o 'Production' en MLflow
"""

import os
import argparse
import pandas as pd
import mlflow


# ------------------------------------------------------------
def choose_best_model(df: pd.DataFrame, metric: str = "val_rmse") -> pd.Series:
    """Devuelve la fila del modelo ganador según la métrica."""
    df = df.copy()
    df = df.dropna(subset=[metric])
    if df.empty:
        raise ValueError(f"No hay datos válidos en '{metric}'")
    # menor es mejor para RMSE/MAE, mayor para R²
    if "r2" in metric.lower():
        best = df.loc[df[metric].idxmax()]
    else:
        best = df.loc[df[metric].idxmin()]
    return best


def main(args):
    print("⚖️  Comparando modelos en:", args.bench)
    assert os.path.exists(args.bench), f"No existe {args.bench}"
    df = pd.read_csv(args.bench)

    # ordenar columnas
    cols = [c for c in ["framework", "model_path", "val_rmse", "val_mae", "val_r2",
                        "test_rmse", "test_mae", "test_r2"] if c in df.columns]
    df = df[cols].sort_values("val_rmse").reset_index(drop=True)

    # elegir mejor modelo
    best = choose_best_model(df, args.metric)
    print("\n🏆 Mejor modelo:")
    print(best)

    # guardar comparativa
    os.makedirs(args.reports, exist_ok=True)
    out_path = os.path.join(args.reports, "model_selection.csv")
    df.to_csv(out_path, index=False)
    print(f"\n📄 Tabla comparativa guardada en: {out_path}")

    # registrar en MLflow Model Registry
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="model_selection") as run:
        model_uri = f"file://{os.path.abspath(best['model_path'])}"
        model_name = args.model_name
        print(f"\n📦 Registrando modelo '{model_name}' desde {model_uri}")

        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        # setear stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=args.stage,
            archive_existing_versions=True,
        )

        mlflow.log_param("winner_framework", best["framework"])
        mlflow.log_param("stage", args.stage)
        mlflow.log_param("metric", args.metric)
        mlflow.log_metric("val_rmse", best.get("val_rmse", None))
        mlflow.log_metric("val_mae",  best.get("val_mae", None))
        mlflow.log_metric("val_r2",   best.get("val_r2", None))

    print(f"✅ Modelo registrado en MLflow como '{model_name}' → Stage: {args.stage}")
    print("   Ver en UI con: make mlflow-ui")


# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paso 7 — Comparación y registro de modelos AutoML")
    ap.add_argument("--bench", default="reports/automl_bench.csv")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--metric", default="val_rmse")
    ap.add_argument("--mlflow-uri", dest="mlflow_uri", default="mlruns")
    ap.add_argument("--experiment", default="ecobici_model_selection")
    ap.add_argument("--model-name", default="ecobici_best")
    ap.add_argument("--stage", default="Staging", choices=["None", "Staging", "Production"])
    args = ap.parse_args()
    main(args)