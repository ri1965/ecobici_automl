#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paso 7 — Comparación y registro del mejor modelo
------------------------------------------------
Soporta CSV "ancho" (val_rmse/test_rmse/...) o "largo" (framework, split, rmse, ...).
Elige el mejor por métrica, lo registra en MLflow y (opcional) escribe el alias
Champion en filesystem para inferencia batch.

Salidas:
- reports/model_selection.csv
- reports/champion_selection.json
- MLflow Model Registry (stage)
- models/Champion/best_model.pkl (si --champion-path)
"""

import os, argparse, json, time, shutil
from pathlib import Path
import pandas as pd
import mlflow


# ------------------------------------------------------------
def choose_best_model(df: pd.DataFrame, metric: str = "val_rmse") -> pd.Series:
    """Devuelve la fila del modelo ganador según la métrica."""
    df = df.copy()
    if metric not in df.columns:
        raise ValueError(f"La métrica '{metric}' no existe: {list(df.columns)}")
    df = df.dropna(subset=[metric])
    if df.empty:
        raise ValueError(f"No hay datos válidos en '{metric}'")
    if "r2" in metric.lower():
        return df.loc[df[metric].idxmax()]
    else:
        return df.loc[df[metric].idxmin()]


def normalize_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tabla de métricas a formato ancho (val/test)."""
    orig_cols = list(df.columns)
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "val_rmse" in df.columns or "test_rmse" in df.columns:
        cols = [c for c in ["framework", "model_path", "val_rmse", "val_mae", "val_r2",
                            "test_rmse", "test_mae", "test_r2"] if c in df.columns]
        return df[cols] if cols else df

    needed = {"framework", "split", "rmse"}
    if not needed.issubset(set(df.columns)):
        raise KeyError(f"Faltan columnas. Tengo {orig_cols}. Necesito 'val_rmse' o {needed}.")

    tmp = df.loc[df["split"].notna()].copy()
    tmp["split"] = tmp["split"].str.lower().map({"val": "val", "validation": "val", "test": "test"})

    rmse = tmp.pivot_table(index="framework", columns="split", values="rmse", aggfunc="min") \
              .rename(columns={"val": "val_rmse", "test": "test_rmse"})
    if "mae" in tmp.columns:
        mae = tmp.pivot_table(index="framework", columns="split", values="mae", aggfunc="min") \
                 .rename(columns={"val": "val_mae", "test": "test_mae"})
        rmse = rmse.join(mae, how="left")
    if "r2" in tmp.columns:
        r2 = tmp.pivot_table(index="framework", columns="split", values="r2", aggfunc="max") \
                .rename(columns={"val": "val_r2", "test": "test_r2"})
        rmse = rmse.join(r2, how="left")

    out = rmse.reset_index()

    # Mantener model_path si existía
    if "model_path" in orig_cols:
        mp = df.loc[df["model_path"].notna(), ["framework", "model_path"]].drop_duplicates()
        out = out.merge(mp, on="framework", how="left")
    return out


def fill_model_paths(df: pd.DataFrame, args) -> pd.DataFrame:
    """Completa rutas a modelos si no estaban en el CSV."""
    if "model_path" in df.columns and df["model_path"].notna().any():
        return df

    mapping = {}
    if args.pycaret_model_path:  mapping["pycaret"]  = args.pycaret_model_path
    if args.flaml_model_path:    mapping["flaml"]    = args.flaml_model_path
    if args.baseline_model_path: mapping["baseline"] = args.baseline_model_path
    if args.h2o_model_path:      mapping["h2o"]      = args.h2o_model_path
    if args.xgb_model_path:
        mapping["xgboost"] = args.xgb_model_path
        mapping["xgb"]     = args.xgb_model_path

    if not mapping:
        raise ValueError("No hay 'model_path' en CSV ni rutas por argumentos. Ej.: --pycaret-model-path models/03A_pycaret/pycaret_best_model.pkl")

    df = df.copy()
    df["framework"] = df["framework"].str.lower()
    df["model_path"] = df["framework"].map(mapping)
    return df


# ------------------------------------------------------------
def main(args):
    print("⚖️  Comparando modelos en:", args.bench)
    assert os.path.exists(args.bench), f"No existe {args.bench}"

    # 1️⃣ Leer y normalizar tabla
    df = pd.read_csv(args.bench)
    df = normalize_metrics_table(df)
    df = fill_model_paths(df, args)

    # 2️⃣ Elegir métrica
    metric_alias = {
        "rmse": "val_rmse", "mae": "val_mae", "r2": "val_r2",
        "val_rmse": "val_rmse", "val_mae": "val_mae", "val_r2": "val_r2",
        "test_rmse": "test_rmse", "test_mae": "test_mae", "test_r2": "test_r2",
    }
    metric = metric_alias.get(args.metric.strip().lower(), args.metric)
    if metric not in df.columns:
        raise ValueError(f"Métrica '{metric}' no existe. Columnas: {list(df.columns)}")

    ascending = ("r2" not in metric.lower())
    df_sorted = df.sort_values(metric, ascending=ascending).reset_index(drop=True)

    os.makedirs(args.reports, exist_ok=True)
    out_sel = os.path.join(args.reports, "model_selection.csv")
    df_sorted.to_csv(out_sel, index=False)
    print(f"📄 Tabla comparativa guardada en: {out_sel}")

    # 3️⃣ Seleccionar mejor modelo
    best = choose_best_model(df_sorted, metric)
    print("\n🏆 Mejor modelo (por", metric, "):")
    print(best)

    # 4️⃣ Registrar en MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    model_path = best.get("model_path", None)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el archivo del modelo ganador: {model_path}")

    abs_model = os.path.abspath(model_path)
    model_uri = f"file://{abs_model}"
    model_name = args.model_name

    with mlflow.start_run(run_name="model_selection"):
        print(f"\n📦 Registrando modelo '{model_name}' desde {model_uri}")
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=result.version, stage=args.stage, archive_existing_versions=True,
        )

        mlflow.log_param("winner_framework", best.get("framework"))
        mlflow.log_param("stage", args.stage)
        mlflow.log_param("metric", metric)
        for k in ["val_rmse", "val_mae", "val_r2", "test_rmse", "test_mae", "test_r2"]:
            if k in best.index and pd.notna(best[k]):
                try:
                    mlflow.log_metric(k, float(best[k]))
                except:
                    pass

    print(f"✅ Modelo registrado en MLflow como '{model_name}' → Stage: {args.stage}")
    print("   Ver en UI con: mlflow ui --backend-store-uri", args.mlflow_uri)

    # 5️⃣ Crear alias Champion local
    if args.champion_path:
        dst = Path(args.champion_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_model, dst)
        print(f"💾 Alias Champion actualizado en filesystem → {dst}")

    # 6️⃣ Guardar manifiesto de selección
    manifest = {
        "winner_framework": str(best.get("framework")),
        "metric_used": metric,
        "winner_model_path": abs_model,
        "mlflow_model_name": model_name,
        "mlflow_stage": args.stage,
        "champion_path": str(args.champion_path) if args.champion_path else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_manifest = os.path.join(args.reports, "champion_selection.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print("📄 Manifiesto de selección →", out_manifest)


# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paso 7 — Comparación y registro de modelos AutoML")
    ap.add_argument("--bench", default="reports/automl_bench.csv", help="CSV de benchmark (ancho o largo).")
    ap.add_argument("--reports", default="reports", help="Carpeta de salida para comparativas.")
    ap.add_argument("--metric", default="val_rmse", help="Métrica (RMSE/MAE/R2 o val_*/test_*).")
    ap.add_argument("--mlflow-uri", dest="mlflow_uri", default="mlruns", help="Tracking URI de MLflow.")
    ap.add_argument("--experiment", default="ecobici_model_selection", help="Nombre del experimento en MLflow.")
    ap.add_argument("--model-name", default="ecobici_champion", help="Nombre del modelo en el Model Registry.")
    ap.add_argument("--stage", default="Production", choices=["None", "Staging", "Production"], help="Stage destino.")
    ap.add_argument("--pycaret-model-path", default=None, help="Ruta al .pkl de PyCaret (models/03A_pycaret/pycaret_best_model.pkl)")
    ap.add_argument("--flaml-model-path",   default=None, help="Ruta al .pkl de FLAML (models/06_flaml/flaml_automl.pkl)")
    ap.add_argument("--baseline-model-path",default=None)
    ap.add_argument("--h2o-model-path",     default=None)
    ap.add_argument("--xgb-model-path",     default=None)
    ap.add_argument("--champion-path",      default=None, help="Alias filesystem: models/Champion/best_model.pkl")
    args = ap.parse_args()
    main(args)