# src/predict_batch.py
import os, sys, json, warnings, argparse
from functools import partial
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

# ------------------------------- #
# Warnings + tqdm (una sola barra)
# ------------------------------- #
warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, file=sys.stdout, dynamic_ncols=True, mininterval=0.2, leave=True)
except Exception:  # fallback si no hay tqdm
    def tqdm(x, **k):
        return x

# ------------------------------- #
# Utilidades de features
# ------------------------------- #
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

def time_features_from_ts(ts: pd.Timestamp) -> dict:
    hour  = ts.hour
    dow   = ts.dayofweek
    month = ts.month
    is_weekend = int(dow in (5, 6))
    hour_sin = np.sin(2*np.pi*hour/24)
    hour_cos = np.cos(2*np.pi*hour/24)
    return {
        "hour": hour, "dow": dow, "month": month, "is_weekend": is_weekend,
        "hour_sin": hour_sin, "hour_cos": hour_cos
    }

def make_X_y_train_schema(df: pd.DataFrame, y_col: str, id_col: str, ts_col: str):
    drop_cols = [c for c in [y_col, id_col, ts_col] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    X = expand_json_like_columns(X, exclude=[])
    X = onehot_low_card(X)
    X = X.select_dtypes(include=["number"]).fillna(0.0)
    y = df[y_col].astype(float).copy()
    return X, y

# ------------------------------- #
# Core
# ------------------------------- #
def main(
    model_path: str,
    data_path: str,
    train_split_path: str,
    out_dir: str,
    outfile: str,
    asof_str: str,
    horizons: list[int],
    freq_minutes: int,
    y_col: str,
    id_col: str,
    ts_col: str,
):
    # Checks
    assert os.path.exists(model_path), f"Modelo no encontrado: {model_path}"
    assert os.path.exists(data_path),  f"Datos no encontrados: {data_path}"
    assert os.path.exists(train_split_path), f"Train split no encontrado: {train_split_path}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"ASOF: {asof_str} | HORIZONS: {horizons} | FREQ_MIN: {freq_minutes}")

    # Modelo
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise ValueError("El modelo cargado no expone .predict().")

    # Datos
    df = pd.read_parquet(data_path).copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    # Timezone armonizado
    tz = df[ts_col].dt.tz  # tzinfo o None
    asof_ts = pd.to_datetime(asof_str)
    if tz is not None:
        if asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize(tz)
        else:
            asof_ts = asof_ts.tz_convert(tz)
    else:
        if asof_ts.tzinfo is not None:
            asof_ts = asof_ts.tz_convert(None)

    # Última observación por estación ≤ asof
    df = df.sort_values([id_col, ts_col])
    latest = (
        df[df[ts_col] <= asof_ts]
        .groupby(id_col, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    if latest.empty:
        raise RuntimeError("No hay observaciones ≤ asof para ninguna estación.")

    print(f"Estaciones con dato ≤ asof: {latest[id_col].nunique()}")

    # Esquema de entrenamiento (columnas y orden)
    df_train_for_schema = pd.read_parquet(train_split_path)
    X_schema, _ = make_X_y_train_schema(df_train_for_schema, y_col, id_col, ts_col)
    TRAIN_COLS = list(X_schema.columns)

    # Constantes numéricas por estación
    latest_num = latest.copy()
    latest_num = latest_num.drop(columns=[c for c in [y_col, id_col, ts_col] if c in latest_num.columns], errors="ignore")
    latest_num = make_numeric_features(latest_num)
    latest_num.index = latest[id_col].values

    # Predicción recursiva (una barra exterior)
    preds = []
    total_est = len(latest)

    for i in tqdm(range(total_est), desc="🔄 Estaciones"):
        row = latest.iloc[i]
        sid = row[id_col]

        y_lag1 = float(row.get("y_lag1", float("nan")))
        y_lag2 = float(row.get("y_lag2", float("nan")))
        y_ma3  = float(row.get("y_ma3", float("nan")))
        if np.isnan(y_lag1) or np.isnan(y_lag2) or np.isnan(y_ma3):
            continue

        const_feats = latest_num.loc[sid].astype(float).to_dict() if sid in latest_num.index else {}
        cur_ts = asof_ts

        for h in horizons:
            ts_pred = cur_ts + timedelta(minutes=freq_minutes * h)

            feats = {
                **time_features_from_ts(ts_pred),
                "y_lag1": y_lag1, "y_lag2": y_lag2, "y_ma3": y_ma3,
                **const_feats
            }

            X_raw = pd.DataFrame([feats]).select_dtypes(include=["number"]).fillna(0.0)
            X = X_raw.reindex(columns=TRAIN_COLS, fill_value=0.0)

            yhat = float(model.predict(X)[0])

            yhat_lo = yhat_hi = None
            if hasattr(model, "estimators_") and isinstance(model.estimators_, list) and len(model.estimators_) > 1:
                trees = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
                std = float(np.std(trees))
                yhat_lo = yhat - 1.96 * std
                yhat_hi = yhat + 1.96 * std

            preds.append({
                id_col: sid,
                "timestamp_pred": ts_pred,
                "h": int(h),
                "yhat": yhat,
                "yhat_lo": yhat_lo,
                "yhat_hi": yhat_hi
            })

            # actualización recursiva de lags
            y_lag2 = y_lag1
            y_lag1 = yhat
            y_ma3  = float(np.mean([yhat, y_lag2, y_ma3]))

    # Celda 6 — Resultado y guardado
    if not preds:
        raise RuntimeError("No se generaron predicciones (posible falta de lags en todas las estaciones).")

    df_pred = (
        pd.DataFrame(preds)
        .sort_values([id_col, "timestamp_pred", "h"])
        .reset_index(drop=True)
    )

    out_path = os.path.join(out_dir, outfile)
    df_pred.to_parquet(out_path, index=False)
    print(f"[OK] Predicciones guardadas en: {out_path}")
    print(df_pred.head(10))

# ------------------------------- #
# CLI
# ------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predicción batch multi-horizonte (recursiva) por estación.")
    ap.add_argument("--model",        required=True, help="Ruta al modelo entrenado (pkl).")
    ap.add_argument("--input",        required=True, help="Parquet listo para modelar (ecobici_model_ready.parquet).")
    ap.add_argument("--train_split",  required=True, help="Parquet de train para derivar el esquema de columnas.")
    ap.add_argument("--out_pred",     default="predictions", help="Carpeta de salida (default: predictions).")
    ap.add_argument("--outfile",      default="predictions.parquet", help="Archivo de salida (default: predictions.parquet).")
    ap.add_argument("--asof",         required=True, help='Timestamp de referencia, ej. "2025-10-24 10:00:00".')
    ap.add_argument("--horizons",     nargs="+", type=int, default=[1,3,6,12], help="Horizontes (ej. 1 3 6 12).")
    ap.add_argument("--freq_minutes", type=int, default=60, help="Minutos por paso (default: 60).")
    ap.add_argument("--y",            dest="y_col",  default="num_bikes_available")
    ap.add_argument("--id",           dest="id_col", default="station_id")
    ap.add_argument("--ts",           dest="ts_col", default="ts_local")
    args = ap.parse_args()

    main(
        model_path=args.model,
        data_path=args.input,
        train_split_path=args.train_split,
        out_dir=args.out_pred,
        outfile=args.outfile,
        asof_str=args.asof,
        horizons=args.horizons,
        freq_minutes=args.freq_minutes,
        y_col=args.y_col,
        id_col=args.id_col,
        ts_col=args.ts_col,
    )