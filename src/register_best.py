#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
register_best.py
----------------
Selecciona el mejor modelo desde un benchmark CSV y lo registra en MLflow Model Registry.

Entrada esperada (reports/automl_bench.csv):
- Debe tener al menos: framework, model_path, val_rmse (y opcionalmente val_mae, val_r2)
- Las rutas en model_path pueden venir relativas (p. ej. '../models/...'); este script las normaliza.

Ejemplo de uso:
python src/register_best.py \
  --bench_csv reports/automl_bench.csv \
  --mlflow-uri mlruns \
  --registry_name ecobici_best \
  --out_selection reports/model_selection.csv \
  --metric val_rmse --greater_is_better false \
  --stage Staging
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient


# ------------------------ utilidades ------------------------ #
def _resolve_model_path(p: str) -> str:
    """Intenta varias normalizaciones para encontrar un path existente."""
    if not isinstance(p, str):
        return ""
    p = p.strip()
    cands = [
        p,
        os.path.normpath(p),
        os.path.abspath(os.path.normpath(p)),
        os.path.normpath(p.lstrip("./").lstrip("../")),
        os.path.abspath(os.path.normpath(p.lstrip("./").lstrip("../"))),
        os.path.join(os.getcwd(), p),
        os.path.join(os.getcwd(), p.lstrip("./").lstrip("../")),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return ""


def _choose_winner(df: pd.DataFrame, metric: str, greater_is_better: bool,
                   tiebreakers: list[str]) -> pd.Series:
    """Elige la mejor fila por métrica y desempates."""
    if metric not in df.columns:
        raise ValueError(f"El CSV no contiene la métrica '{metric}'. Columnas: {list(df.columns)}")

    # Filtra métricas válidas
    dfm = df.copy()
    dfm = dfm[~dfm[metric].isna()].copy()
    if dfm.empty:
        raise ValueError(f"No hay filas con '{metric}' válido en el benchmark.")

    # Orden principal
    ascending = not greater_is_better
    sort_cols = [metric]
    sort_asc  = [ascending]

    # Tiebreakers (si existen en el CSV)
    for tb in tiebreakers:
        if tb in dfm.columns:
            # Por convención: si tb es RMSE/MAE → menor es mejor; si tb es R2 → mayor es mejor
            if tb.lower() in ("rmse", "mae", "mape", "mse", "val_rmse", "val_mae"):
                sort_cols.append(tb); sort_asc.append(True)
            else:
                sort_cols.append(tb); sort_asc.append(False)

    dfm = dfm.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
    return dfm.iloc[0]


def _log_model_to_mlflow(model, flavor_hint: str = "auto") -> str:
    """
    Registra el modelo en el run actual como artifact 'model' y retorna un model_uri tipo runs:/<run_id>/model.
    Intenta sklearn primero; si no, usa pyfunc con un wrapper genérico.
    """
    import mlflow.pyfunc
    model_uri = None

    def _is_sklearn_estimator(m):
        try:
            import sklearn  # noqa: F401
            # heurística: que tenga predict y get_params
            return hasattr(m, "predict") and hasattr(m, "get_params")
        except Exception:
            return False

    if flavor_hint == "sklearn" or (flavor_hint == "auto" and _is_sklearn_estimator(model)):
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_path="model")
    else:
        # Wrapper pyfunc genérico
        class _PredictWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, inner):
                self.inner = inner
            def predict(self, context, model_input):
                # model_input: pandas.DataFrame
                if hasattr(self.inner, "predict"):
                    y = self.inner.predict(model_input)
                else:
                    raise RuntimeError("El objeto no tiene .predict().")
                # asegurar vector np.float64
                y = np.asarray(y).astype(float)
                return y

        artifacts = {}
        conda_env = None  # opcional: podrías generar uno
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_PredictWrapper(model),
            artifacts=artifacts,
            conda_env=conda_env,
        )

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    return model_uri


# ------------------------ main ------------------------ #
def main(args):
    assert os.path.exists(args.bench_csv), f"No existe el benchmark CSV: {args.bench_csv}"

    # MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # Leer benchmark
    bench = pd.read_csv(args.bench_csv)
    if "model_path" not in bench.columns:
        raise ValueError("El CSV no tiene columna 'model_path'.")

    # Elegir ganador
    winner = _choose_winner(
        bench,
        metric=args.metric,
        greater_is_better=(str(args.greater_is_better).lower() == "true"),
        tiebreakers=args.tiebreakers.split(",") if args.tiebreakers else []
    )

    raw_model_path = str(winner["model_path"]).strip()
    model_path_resolved = _resolve_model_path(raw_model_path)
    if not model_path_resolved:
        raise AssertionError(f"No existe el pickle ganador: {raw_model_path} (revisar rutas en el CSV)")

    print(f"🏆 Ganador: {winner.get('framework','?')} → {raw_model_path}")
    print(f"   Ruta resuelta: {model_path_resolved}")

    # Cargar modelo
    model = joblib.load(model_path_resolved)

    # Log + registrar en Registry
    with mlflow.start_run(run_name="model_selection_register"):
        # Log params/metrics del ganador
        for col in winner.index:
            val = winner[col]
            # evitar objetos no serializables
            if isinstance(val, (str, float, int, bool)) or val is None or np.isscalar(val):
                mlflow.log_param(f"winner_{col}", val)
        if args.metric in winner:
            try:
                mlflow.log_metric("selection_metric", float(winner[args.metric]))
            except Exception:
                pass

        # Intentar loggear como sklearn; si no, pyfunc wrapper
        flavor_hint = "sklearn"  # la mayoría de casos (PyCaret/FLAML) son estimadores sklearn
        model_uri = _log_model_to_mlflow(model, flavor_hint=flavor_hint)

        # Registrar en Registry
        result = mlflow.register_model(model_uri=model_uri, name=args.registry_name)
        version = result.version
        registered_name = result.name
        print(f"📌 Registrado en Registry: models:/{registered_name}/{version}")

        # Transicionar de etapa si corresponde
        if args.stage:
            client = MlflowClient()
            client.transition_model_version_stage(
                name=registered_name,
                version=version,
                stage=args.stage,
                archive_existing_versions=False,
            )
            print(f"🔖 Stage → {args.stage}")

    # Guardar selección
    os.makedirs(os.path.dirname(args.out_selection), exist_ok=True)
    out = winner.to_dict()
    out.update({
        "model_path_resolved": model_path_resolved,
        "registered_name": registered_name,
        "registered_version": version,
        "stage": args.stage or "",
    })
    pd.DataFrame([out]).to_csv(args.out_selection, index=False)
    print(f"🧾 Detalle guardado en: {args.out_selection}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Selecciona y registra el mejor modelo en MLflow Registry.")
    ap.add_argument("--bench_csv", required=True, help="CSV con benchmark (reports/automl_bench.csv)")
    ap.add_argument("--mlflow-uri", dest="mlflow_uri", default="mlruns")
    ap.add_argument("--experiment", default="ecobici_model_selection")
    ap.add_argument("--registry_name", required=True, help="Nombre en Model Registry (p. ej. ecobici_best)")
    ap.add_argument("--out_selection", default="reports/model_selection.csv")
    ap.add_argument("--metric", default="val_rmse", help="Columna de métrica para ordenar (default: val_rmse)")
    ap.add_argument("--greater_is_better", default="false", help="true/false (default: false)")
    ap.add_argument("--tiebreakers", default="val_mae,val_r2", help="Lista separada por comas para desempate")
    ap.add_argument("--stage", default="", help="Opcional: Staging o Production")
    args = ap.parse_args()
    main(args)