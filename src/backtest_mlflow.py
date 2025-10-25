# src/backtest_mlflow.py
import os, sys, json, warnings, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# tqdm por stdout (una sola barra en consola)
try:
    from functools import partial
    from tqdm import tqdm as _tqdm
    tqdm = partial(_tqdm, file=sys.stdout, dynamic_ncols=True, mininterval=0.2, leave=True)
except Exception:
    def tqdm(x, **k): return x

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# SciPy es opcional (para KS). Si no está, seguimos sin drift test.
try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

import mlflow


# -----------------------------
# Utils de features / limpieza
# -----------------------------
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

def make_X_y(df: pd.DataFrame, y_col: str, id_col: str, ts_col: str):
    drop_cols = [c for c in [y_col, id_col, ts_col] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    X = expand_json_like_columns(X, exclude=[])
    low = [c for c in X.columns if X[c].dtype == "object" and X[c].nunique(dropna=True) <= 20]
    if low:
        X = pd.get_dummies(X, columns=low, drop_first=True)
    X = X.select_dtypes(include=["number"]).fillna(0.0)
    y = df[y_col].astype(float).copy()
    return X, y

def preprocess(df: pd.DataFrame, ts_col: str, id_col: str) -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values([id_col, ts_col]).reset_index(drop=True)
    return df

# -----------------------------
# Backtesting (expanding window)
# -----------------------------
def make_time_splits(
    df: pd.DataFrame,
    ts_col: str,
    n_splits: int = 3,
    p_train_start: float = 0.50,
    p_train_end: float = 0.75,
    p_val: float = 0.15,
):
    """
    Genera n_splits con train creciente (expanding) y validación fija (p_val) sobre la línea de tiempo global.
    """
    tsu = df[ts_col].sort_values().unique()
    n = len(tsu)
    trains = np.linspace(p_train_start, p_train_end, n_splits)
    for i, p_tr in enumerate(trains, start=1):
        cut_tr_end = tsu[int(n * p_tr)]
        cut_val_end = tsu[min(n - 1, int(n * (p_tr + p_val)))]
        tr = df[df[ts_col] <= cut_tr_end].copy()
        va = df[(df[ts_col] > cut_tr_end) & (df[ts_col] <= cut_val_end)].copy()
        if len(va) == 0:
            continue
        yield i, tr, va, cut_tr_end, cut_val_end

# -----------------------------
# Chequeos de frecuencia / cobertura
# -----------------------------
def quick_time_profile(df: pd.DataFrame, ts_col: str) -> dict:
    s = df[ts_col].sort_values().dropna()
    if s.empty:
        return {"n_rows": 0, "n_days": 0, "freq_minutes_mode": None, "hours_min": None, "hours_max": None}
    diffs = s.diff().dropna().dt.total_seconds() / 60.0
    freq_mode = None
    if not diffs.empty:
        # modo aproximado (al minuto)
        freq_mode = float(diffs.round().mode().iloc[0])
    # cobertura por día (horario mínimo y máximo observado)
    byday = s.groupby(s.dt.date).agg(["min", "max"])
    hours_min = float(byday["min"].dt.hour.median()) if not byday.empty else None
    hours_max = float(byday["max"].dt.hour.median()) if not byday.empty else None
    return {
        "n_rows": int(len(df)),
        "n_days": int(s.dt.date.nunique()),
        "freq_minutes_mode": freq_mode,
        "hours_min": hours_min,
        "hours_max": hours_max,
    }

# -----------------------------
# Main
# -----------------------------
def main(
    curated_path: str,
    reports_dir: str,
    experiment_name: str,
    y_col: str,
    id_col: str,
    ts_col: str,
    n_splits: int,
    p_train_start: float,
    p_train_end: float,
    p_val: float,
    n_estimators: int,
    random_state: int,
):
    assert os.path.exists(curated_path), f"No existe {curated_path}"
    os.makedirs(reports_dir, exist_ok=True)

    # MLflow local
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)

    # Cargar y perfilar tiempo
    df_all = preprocess(pd.read_parquet(curated_path), ts_col, id_col)
    prof = quick_time_profile(df_all, ts_col)
    print(
        f"[Perfil] filas={prof['n_rows']} | días={prof['n_days']} | "
        f"freq_mode≈{prof['freq_minutes_mode']} min | franja típica≈{prof['hours_min']}:00–{prof['hours_max']}:00"
    )

    # Validación de tu hipótesis (minuto + 8–21 + ~20 días)
    warn = []
    if prof["freq_minutes_mode"] is not None and abs(prof["freq_minutes_mode"] - 1.0) > 0.1:
        warn.append(f"- Frecuencia modal no es ~1 minuto (modo={prof['freq_minutes_mode']}).")
    if prof["hours_min"] is not None and prof["hours_min"] > 8:
        warn.append(f"- Hora mínima típica > 8 (min={prof['hours_min']}).")
    if prof["hours_max"] is not None and prof["hours_max"] < 21:
        warn.append(f"- Hora máxima típica < 21 (max={prof['hours_max']}).")
    if prof["n_days"] and prof["n_days"] < 18:
        warn.append(f"- Días distintos < 20 (días={prof['n_days']}).")

    if warn:
        print("[AVISO] El perfil temporal no calza exactamente con 'cada minuto, 8–21h, ~20 días':")
        for w in warn: print(" ", w)
    else:
        print("[OK] El perfil temporal es consistente con 'cada minuto, 8–21h, ~20 días'.")

    # Backtesting
    SEED = random_state
    RF_PARAMS = dict(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        random_state=SEED,
        n_jobs=-1,
    )

    summary = []
    for i, tr, va, cut_tr_end, cut_val_end in tqdm(
        make_time_splits(df_all, ts_col, n_splits=n_splits, p_train_start=p_train_start, p_train_end=p_train_end, p_val=p_val),
        total=n_splits, desc="Backtesting"
    ):
        Xtr, ytr = make_X_y(tr, y_col, id_col, ts_col)
        Xva, yva = make_X_y(va, y_col, id_col, ts_col)

        # Baseline persistente si existe y_lag1
        base_rmse = base_mae = base_r2 = None
        if "y_lag1" in Xva.columns:
            yhat_b = Xva["y_lag1"].values
            base_rmse = mean_squared_error(yva, yhat_b, squared=False)
            base_mae  = mean_absolute_error(yva, yhat_b)
            base_r2   = r2_score(yva, yhat_b)

        # Modelo
        rf = RandomForestRegressor(**RF_PARAMS).fit(Xtr, ytr)
        pred = rf.predict(Xva)
        rmse = mean_squared_error(yva, pred, squared=False)
        mae  = mean_absolute_error(yva, pred)
        r2   = r2_score(yva, pred)

        # Muestra de predicciones por split (artefacto)
        preds_df = pd.DataFrame({
            "ts": va[ts_col].values,
            "station_id": va[id_col].values,
            "y_true": yva.values,
            "y_pred": pred
        }).sort_values(["station_id", "ts"]).head(50)
        art_path = os.path.join(reports_dir, f"backtest_split{i}_sample_preds.csv")
        preds_df.to_csv(art_path, index=False)

        # MLflow: un run por split
        with mlflow.start_run(run_name=f"bt_split_{i}"):
            mlflow.log_param("split_id", i)
            mlflow.log_param("cut_train_end", str(cut_tr_end))
            mlflow.log_param("cut_val_end", str(cut_val_end))
            mlflow.log_param("y_col", y_col)
            mlflow.log_param("id_col", id_col)
            mlflow.log_param("ts_col", ts_col)
            for k, v in RF_PARAMS.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_mae",  mae)
            mlflow.log_metric("val_r2",   r2)
            if base_rmse is not None:
                mlflow.log_metric("baseline_rmse", base_rmse)
                mlflow.log_metric("baseline_mae",  base_mae)
                mlflow.log_metric("baseline_r2",   base_r2)

            mlflow.log_artifact(art_path)

        summary.append({
            "split": i,
            "train_end": cut_tr_end,
            "val_end": cut_val_end,
            "val_rmse": rmse,
            "val_mae": mae,
            "val_r2": r2,
            "baseline_rmse": base_rmse,
            "baseline_mae": base_mae,
            "baseline_r2": base_r2
        })

    bt = pd.DataFrame(summary)
    bt_path = os.path.join(reports_dir, "backtest_summary.csv")
    bt.to_csv(bt_path, index=False)
    print(f"[OK] Backtest summary guardado en: {bt_path}")
    if not bt.empty:
        try:
            # resumen simple por pantalla
            print(bt.describe().round(4))
        except Exception:
            pass

    # Monitoreo mínimo (drift + % de ceros)
    mon_path = os.path.join(reports_dir, "monitoring_quickcheck.csv")
    rows_written = 0

    if len(summary) >= 2:
        first_cut = summary[0]["val_end"]
        last_cut  = summary[-1]["val_end"]

        early = df_all[df_all[ts_col] <= first_cut]
        late  = df_all[(df_all[ts_col] > last_cut - (last_cut - first_cut)) & (df_all[ts_col] <= last_cut)]

        with open(mon_path, "w") as f:
            f.write("# Quick monitoring checks\n")
        rows_written += 1

        # % de ceros por estación
        zeros = (
            df_all.groupby(id_col)[y_col]
            .apply(lambda s: (s == 0).mean())
            .reset_index(name="p_zero")
            .sort_values("p_zero", ascending=False)
        )
        zeros.to_csv(mon_path, mode="a", index=False)

        # KS drift si hay SciPy y y_lag1
        if HAS_SCIPY and "y_lag1" in df_all.columns:
            e = early["y_lag1"].dropna().astype(float)
            l = late["y_lag1"].dropna().astype(float)
            if len(e) > 100 and len(l) > 100:
                ks = ks_2samp(e, l)
                with open(mon_path, "a") as f:
                    f.write("\n# KS drift (early vs late) sobre y_lag1\n")
                    f.write("feature,ks_stat,pvalue\n")
                    f.write(f"y_lag1,{ks.statistic},{ks.pvalue}\n")
                rows_written += 1

    if rows_written:
        print(f"[OK] Monitoreo guardado en: {mon_path}")
    else:
        print("[INFO] Monitoreo omitido (splits insuficientes o SciPy no disponible).")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Etapa 5: Backtesting + MLflow + Monitoreo mínimo.")
    ap.add_argument("--curated",     default="data/curated/ecobici_model_ready.parquet")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--experiment",  default="ecobici_backtest_rf")
    ap.add_argument("--y",           dest="y_col", default="num_bikes_available")
    ap.add_argument("--id",          dest="id_col", default="station_id")
    ap.add_argument("--ts",          dest="ts_col", default="ts_local")
    ap.add_argument("--n_splits",    type=int, default=3)
    ap.add_argument("--p_train_start", type=float, default=0.50)
    ap.add_argument("--p_train_end",   type=float, default=0.75)
    ap.add_argument("--p_val",         type=float, default=0.15)
    ap.add_argument("--n_estimators",  type=int, default=300)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    main(
        curated_path=args.curated,
        reports_dir=args.reports_dir,
        experiment_name=args.experiment,
        y_col=args.y_col,
        id_col=args.id_col,
        ts_col=args.ts_col,
        n_splits=args.n_splits,
        p_train_start=args.p_train_start,
        p_train_end=args.p_train_end,
        p_val=args.p_val,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
       