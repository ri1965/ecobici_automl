# src/predict_batch.py
import os, sys, json, warnings, argparse
from functools import partial
from datetime import timedelta
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, file=sys.stdout, dynamic_ncols=True, mininterval=0.2, leave=True)
except Exception:
    def tqdm(x, **k): return x

# ===============================
# Utils
# ===============================
def _looks_like_json_dict(s: str) -> bool:
    s = str(s).strip()
    return s.startswith("{") and s.endswith("}")

def expand_json_like_columns(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
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
    """Deriva el esquema de columns numéricas a partir del split de train (como en training)."""
    df = df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    drop_cols = [c for c in [y_col, id_col, ts_col] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    X = make_numeric_features(X)
    y = df[y_col].astype(float).copy()
    return X, y

def _ensure_naive_datetime(series: pd.Series) -> pd.Series:
    """Si viene con tz, lo vuelve naive preservando hora local."""
    try:
        if getattr(series.dt, "tz", None) is not None:
            return series.dt.tz_convert(None) if hasattr(series.dt, "tz_convert") else series.dt.tz_localize(None)
    except Exception:
        pass
    return series

def _coerce_station_id_dtype(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    if s.str.fullmatch(r"\d+").all():
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    return s

TIME_KEYS = {"hour","dow","month","is_weekend","hour_sin","hour_cos"}
LAG_KEYS  = {"y_lag1","y_lag2","y_ma3"}
BLOCK_KEYS = TIME_KEYS | LAG_KEYS

# -------------------------------
# Descubrir columnas esperadas
# -------------------------------
def discover_expected_columns(model, fallback_cols: list[str]) -> list[str]:
    """
    Devuelve las columnas EXACTAS que el modelo espera.
    - Busca feature_names_in_ en el modelo o en sus steps (Pipeline).
    - Si no encuentra, usa fallback_cols (esquema del split de train).
    """
    # 1) Directo
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # 2) Pipelines sklearn/pycaret
    tried = set()
    def _walk(obj):
        if id(obj) in tried:
            return None
        tried.add(id(obj))
        # atributo directo
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
        # sklearn Pipeline: obj.steps
        if hasattr(obj, "steps") and isinstance(obj.steps, list):
            for name, step in obj.steps:
                cols = _walk(step)
                if cols is not None:
                    return cols
        # ColumnTransformer: transformers_
        if hasattr(obj, "transformers_"):
            for _, trans, _ in getattr(obj, "transformers_", []):
                cols = _walk(trans)
                if cols is not None:
                    return cols
        # pycaret internal
        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                cols = _walk(step)
                if cols is not None:
                    return cols
        return None

    cols = _walk(model)
    return list(cols) if cols is not None else list(fallback_cols)

# -------------------------------
# Predicción agnóstica
# -------------------------------
def predict_any(model, X: pd.DataFrame) -> np.ndarray:
    """
    Si es un modelo de PyCaret, intentamos predict_model para que respete su pipeline.
    Si falla o no está PyCaret, usamos .predict.
    """
    try:
        from pycaret.regression import predict_model as _pc_predict_model
        try:
            out = _pc_predict_model(model, data=X.copy())
            return pd.to_numeric(out["prediction_label"], errors="coerce").astype(float).values
        except Exception:
            pass
    except Exception:
        pass
    # fallback genérico
    return np.asarray(model.predict(X), dtype=float)

# ===============================
# Core
# ===============================
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
    debug_sid: int | None = None,
):
    # --- Validaciones iniciales
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Datos no encontrados: {data_path}")
    if not os.path.exists(train_split_path):
        raise FileNotFoundError(f"Train split no encontrado: {train_split_path}")
    if freq_minutes <= 0:
        raise ValueError("freq_minutes debe ser > 0")
    if not horizons:
        raise ValueError("Debes indicar al menos un horizonte")
    horizons = sorted(set(horizons))

    os.makedirs(out_dir, exist_ok=True)

    print(f"ASOF: {asof_str} | HORIZONS: {horizons} | FREQ_MIN: {freq_minutes}")

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        # aunque intentamos predict_model arriba, exigimos .predict para fallback
        raise ValueError("El modelo cargado no expone .predict().")

    # --- Datos base
    df = pd.read_parquet(data_path).copy()
    if ts_col not in df.columns or id_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Faltan columnas requeridas. Tengo: {df.columns.tolist()}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df[ts_col] = _ensure_naive_datetime(df[ts_col])
    df[id_col] = _coerce_station_id_dtype(df[id_col])

    # --- Resolver asof
    if str(asof_str).lower() == "now":
        asof_ts = df[ts_col].max()
    else:
        asof_ts = pd.to_datetime(asof_str, errors="coerce")
        if pd.isna(asof_ts):
            raise ValueError(f"asof inválido: {asof_str}")

    # --- Última observación válida por estación (<= asof)
    df = df.sort_values([id_col, ts_col])
    latest = (
        df[df[ts_col] <= asof_ts]
        .groupby(id_col, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    if latest.empty:
        raise RuntimeError("No hay observaciones ≤ asof para ninguna estación.")

    n_st = latest[id_col].nunique()
    print(f"Estaciones con dato ≤ asof: {n_st}")

    # --- Esquema base desde TRAIN (fallback)
    df_train_for_schema = pd.read_parquet(train_split_path)
    X_schema, _ = make_X_y_train_schema(df_train_for_schema, y_col, id_col, ts_col)
    TRAIN_COLS = list(X_schema.columns)

    # --- Lista FINAL esperada por el modelo (prioriza feature_names_in_)
    EXPECTED_COLS = discover_expected_columns(model, TRAIN_COLS)

    # Debug
    print("[DEBUG] cols ∩ tiempo:", sorted(set(EXPECTED_COLS) & TIME_KEYS))
    print("[DEBUG] cols ∩ lags:  ", sorted(set(EXPECTED_COLS) & LAG_KEYS))

    # --- Features estáticas (numéricas) de la última fila por estación
    latest_num = latest.copy()
    latest_num = latest_num.drop(columns=[c for c in [y_col, id_col, ts_col] if c in latest_num.columns], errors="ignore")
    latest_num = make_numeric_features(latest_num)
    latest_num = latest_num.drop(columns=[c for c in latest_num.columns if c in BLOCK_KEYS], errors="ignore")
    latest_num.index = latest[id_col].values

    preds = []
    total_est = len(latest)

    for i in tqdm(range(total_est), desc="🔄 Estaciones"):
        row = latest.iloc[i]
        sid = row[id_col]

        # Lags de partida
        y_lag1 = float(row.get("y_lag1", float("nan")))
        y_lag2 = float(row.get("y_lag2", float("nan")))
        y_ma3  = float(row.get("y_ma3", float("nan")))
        if np.isnan(y_lag1) or np.isnan(y_lag2) or np.isnan(y_ma3):
            continue

        const_feats = latest_num.loc[sid].astype(float).to_dict() if sid in latest_num.index else {}
        capacity = float(const_feats.get("capacity", np.nan)) if "capacity" in const_feats else np.nan
        cur_ts = asof_ts

        for h in horizons:
            ts_pred = cur_ts + timedelta(minutes=freq_minutes * h)
            f_time = time_features_from_ts(ts_pred)

            feats = {
                **{k: v for k, v in const_feats.items() if k not in BLOCK_KEYS},
                "y_lag1": y_lag1, "y_lag2": y_lag2, "y_ma3": y_ma3,
                **f_time,
            }

            X_raw = pd.DataFrame([feats]).select_dtypes(include=["number"]).fillna(0.0)

            # --- Alineación estricta a EXPECTED_COLS (sin extras)
            missing = [c for c in EXPECTED_COLS if c not in X_raw.columns]
            for c in missing:
                X_raw[c] = 0.0
            extras = [c for c in X_raw.columns if c not in EXPECTED_COLS]
            if extras:
                X_raw = X_raw.drop(columns=extras)
            X = X_raw[EXPECTED_COLS]

            if i == 0 and h == horizons[0]:
                zero_cols = [c for c in X.columns if X[c].sum() == 0]
                if zero_cols:
                    print(f"[DEBUG] Columnas en 0 tras alinear: {zero_cols[:10]} ...")

            # Predicción (PyCaret-friendly si aplica)
            yhat = float(predict_any(model, X)[0])

            # IC rudimentario para ensambles (si aplica)
            yhat_lo = yhat_hi = None
            if hasattr(model, "estimators_") and isinstance(model.estimators_, list) and len(model.estimators_) > 1:
                trees = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
                std = float(np.std(trees))
                yhat_lo = yhat - 1.96 * std
                yhat_hi = yhat + 1.96 * std

            # Clip por capacidad si disponible
            if not np.isnan(capacity):
                yhat = float(np.clip(yhat, 0.0, capacity))
                if yhat_lo is not None:
                    yhat_lo = float(np.clip(yhat_lo, 0.0, capacity))
                    yhat_hi = float(np.clip(yhat_hi, 0.0, capacity))

            preds.append({
                id_col: sid,
                "timestamp_pred": ts_pred,
                "h": int(h),
                "yhat": yhat,
                "yhat_lo": yhat_lo,
                "yhat_hi": yhat_hi
            })

            # Recursión de lags
            prev_lag1, prev_lag2 = y_lag1, y_lag2
            y_lag2 = prev_lag1
            y_lag1 = yhat
            y_ma3  = float(np.mean([yhat, prev_lag1, prev_lag2]))

    if not preds:
        raise RuntimeError("No se generaron predicciones (posible falta de lags en todas las estaciones).")

    df_pred = (pd.DataFrame(preds)
               .sort_values([id_col, "timestamp_pred", "h"])
               .reset_index(drop=True))

    # --- Nombre de salida idempotente (incluye asof y freq) + latest.parquet
    base = os.path.splitext(os.path.basename(outfile))[0]
    ext  = os.path.splitext(outfile)[1] or ".parquet"
    asof_tag = pd.to_datetime(asof_ts).strftime("%Y%m%d_%H%M")
    out_name = f"{base}__asof{asof_tag}__f{freq_minutes}m.parquet"
    out_path = os.path.join(out_dir, out_name)

    os.makedirs(out_dir, exist_ok=True)
    df_pred.to_parquet(out_path, index=False)

    latest_link = os.path.join(out_dir, "latest.parquet")
    tmp = latest_link + ".tmp"
    df_pred.to_parquet(tmp, index=False)
    os.replace(tmp, latest_link)

    print(f"[OK] Predicciones guardadas en: {out_path}")
    print(f"[OK] Enlace actualizado: {latest_link}")
    print(f"[INFO] filas={len(df_pred)}  estaciones={df_pred[id_col].nunique()}  "
          f"horizontes={sorted(df_pred['h'].unique().tolist())}")
    print(df_pred.head(10))

# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predicción batch multi-horizonte (recursiva) por estación.")
    ap.add_argument("--model",        default="models/03D/best_model.pkl",
                    help="Ruta al modelo entrenado (pkl o pipeline).")
    ap.add_argument("--input",        default="data/curated/ecobici_model_ready.parquet",
                    help="Parquet listo para modelar (ecobici_model_ready.parquet).")
    ap.add_argument("--train_split",  default="data/splits/train.parquet",
                    help="Parquet de train para derivar el esquema de columnas.")
    ap.add_argument("--out_pred",     default="predictions",
                    help="Carpeta de salida.")
    ap.add_argument("--outfile",      default="latest.parquet",
                    help="Nombre base de archivo (se enriquece con asof/freq).")
    ap.add_argument("--asof",         default="now",
                    help='Timestamp de referencia, ej. "2025-10-28 10:00:00" o "now".')
    ap.add_argument("--horizons",     nargs="+", type=int, default=[1, 3, 6, 12],
                    help="Horizontes (1 3 6 12).")
    ap.add_argument("--freq_minutes", type=int, default=60,
                    help="Minutos por paso (default:60).")
    ap.add_argument("--y",            dest="y_col",  default="num_bikes_available")
    ap.add_argument("--id",           dest="id_col", default="station_id")
    ap.add_argument("--ts",           dest="ts_col", default="ts_local")
    ap.add_argument("--debug_sid",    type=int, default=None,
                    help="Station ID para volcar features (debug).")
    args = ap.parse_args()

    try:
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
            debug_sid=args.debug_sid,
        )
        print("[DONE] predict_batch OK")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] predict_batch: {e}")
        sys.exit(1)