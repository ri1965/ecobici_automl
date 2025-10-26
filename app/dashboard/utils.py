import os
import pandas as pd

# Paths
DATA_DIR = os.path.join("data", "curated")
REPORTS_DIR = os.path.join("reports")

def load_predictions() -> pd.DataFrame:
    """Carga el parquet de predicciones más reciente desde /predictions."""
    preds_dir = "predictions"
    if not os.path.exists(preds_dir):
        return pd.DataFrame()

    files = [f for f in os.listdir(preds_dir) if f.endswith(".parquet")]
    if not files:
        return pd.DataFrame()

    # el más nuevo por mtime
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(preds_dir, f)))
    path = os.path.join(preds_dir, latest)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[WARN] No se pudo leer {path}: {e}")
        return pd.DataFrame()

    # Normalizaciones
    # timestamp
    if "ts" not in df.columns:
        for cand in ["timestamp_pred", "timestamp", "time", "ts_local", "last_reported", "fecha"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "ts"})
                break
    # yhat
    if "yhat" not in df.columns:
        for cand in ["y_pred", "pred", "predicted_bikes", "prediction", "forecast"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "yhat"})
                break
    # id
    if "station_id" not in df.columns:
        for cand in ["id_station", "id", "station"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "station_id"})
                break
    # h
    if "h" not in df.columns:
        for cand in ["horizon", "horas", "h_pred"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "h"})
                break

    # tipos
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").astype("Int64")
    if "h" in df.columns:
        df["h"] = pd.to_numeric(df["h"], errors="coerce").astype("Int64")
    if "yhat" in df.columns:
        df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")

    # guarda la ruta para mostrar en el dashboard
    df.attrs["__source_path__"] = path

    print(f"[INFO] Predicciones: {path} ({len(df)} filas)")
    return df

def load_stations() -> pd.DataFrame:
    for cand in ["station_information.parquet", "status_information.parquet"]:
        p = os.path.join(DATA_DIR, cand)
        if os.path.exists(p):
            try:
                return pd.read_parquet(p)
            except Exception as e:
                print(f"[WARN] No se pudo leer {p}: {e}")
    return pd.DataFrame()

def load_model_selection() -> pd.DataFrame:
    for cand in ["model_selection.csv", "automl_bench.csv"]:
        p = os.path.join(REPORTS_DIR, cand)
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception as e:
                print(f"[WARN] No se pudo leer {p}: {e}")
    return pd.DataFrame()

def load_metrics_by_split() -> pd.DataFrame:
    for cand in ["metrics_by_split.csv", "automl_metrics_by_split.csv"]:
        p = os.path.join(REPORTS_DIR, cand)
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception as e:
                print(f"[WARN] No se pudo leer {p}: {e}")
    return pd.DataFrame()

def load_backtest_summary() -> pd.DataFrame:
    p = os.path.join(REPORTS_DIR, "backtest_summary.csv")
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] No se pudo leer {p}: {e}")
    return pd.DataFrame()

def load_backtest_samples() -> pd.DataFrame:
    files = [f for f in os.listdir(REPORTS_DIR)
             if "backtest_split" in f and f.endswith("_sample_preds.csv")]
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(os.path.join(REPORTS_DIR, f))
            df["split"] = f.split("_sample_preds.csv")[0].replace("backtest_", "")
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] No se pudo leer {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_feature_importance(top_n: int = 20) -> pd.DataFrame:
    for cand in ["feature_importance_rf.csv", "feature_importance.csv"]:
        p = os.path.join(REPORTS_DIR, cand)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if df.shape[1] == 2:
                    df.columns = ["feature", "importance"]
                if "importance" in df.columns:
                    df = df.sort_values("importance", ascending=False).head(top_n)
                return df
            except Exception as e:
                print(f"[WARN] No se pudo leer {p}: {e}")
    return pd.DataFrame()