# src/train.py
import os
import argparse
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    low_card = [c for c in X.columns if X[c].dtype == "object" and X[c].nunique(dropna=True) <= 20]
    if low_card:
        X = pd.get_dummies(X, columns=low_card, drop_first=True)
    X = X.select_dtypes(include=["number"]).copy().fillna(0.0)
    y = df[y_col].astype(float).copy()
    return X, y

def eval_split(model, X, y):
    pred = model.predict(X)
    rmse = mean_squared_error(y, pred, squared=False)
    mae  = mean_absolute_error(y, pred)
    r2   = r2_score(y, pred)
    return rmse, mae, r2

def main(train_pq, val_pq, test_pq, out_dir, y_col, id_col, ts_col, n_estimators, seed, reports_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # cargar
    df_tr = pd.read_parquet(train_pq)
    df_va = pd.read_parquet(val_pq)

    Xtr, ytr = make_X_y(df_tr, y_col, id_col, ts_col)
    Xva, yva = make_X_y(df_va, y_col, id_col, ts_col)

    # baseline (lag1) si existe en Xva
    rows = []
    if "y_lag1" in Xva.columns:
        yhat_base = Xva["y_lag1"].values
        base_rmse = mean_squared_error(yva, yhat_base, squared=False)
        base_mae  = mean_absolute_error(yva, yhat_base)
        base_r2   = r2_score(yva, yhat_base)
        rows.append({"split":"baseline_lag1","rmse":base_rmse,"mae":base_mae,"r2":base_r2})

    # modelo
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(Xtr, ytr)

    val_rmse, val_mae, val_r2 = eval_split(rf, Xva, yva)
    rows.insert(0, {"split":"val","rmse":val_rmse,"mae":val_mae,"r2":val_r2})

    # test opcional
    if test_pq and os.path.exists(test_pq):
        df_te = pd.read_parquet(test_pq)
        Xt, yt = make_X_y(df_te, y_col, id_col, ts_col)
        test_rmse, test_mae, test_r2 = eval_split(rf, Xt, yt)
        rows.append({"split":"test","rmse":test_rmse,"mae":test_mae,"r2":test_r2})

    # guardar reporte
    rep = pd.DataFrame(rows)
    rep_path = os.path.join(reports_dir, "metrics_by_split.csv")
    rep.to_csv(rep_path, index=False)

    # guardar modelo
    model_path = os.path.join(out_dir, "best_model.pkl")
    joblib.dump(rf, model_path)

    # feature importance (si existe)
    fi_path = os.path.join(reports_dir, "feature_importance_rf.csv")
    if hasattr(rf, "feature_importances_"):
        fi = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
        fi.to_csv(fi_path, header=["importance"])

    # logs consola
    print("[OK] Entrenamiento finalizado")
    print(rep)
    print(f"Modelo guardado en: {model_path}")
    if os.path.exists(fi_path):
        print(f"Feature importance → {fi_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrenamiento simple (RandomForest) para disponibilidad de bicis.")
    ap.add_argument("--train", required=True, help="data/splits/train.parquet")
    ap.add_argument("--val",   required=True, help="data/splits/val.parquet")
    ap.add_argument("--test",  default="",    help="(opcional) data/splits/test.parquet")
    ap.add_argument("--out",   required=True, help="carpeta destino, p.ej. models/03D")
    ap.add_argument("--y",     default="num_bikes_available")
    ap.add_argument("--id",    default="station_id")
    ap.add_argument("--ts",    default="ts_local")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--reports", default="reports", help="carpeta de reportes (default: reports)")
    args = ap.parse_args()

    main(args.train, args.val, args.test, args.out,
         args.y, args.id, args.ts, args.n_estimators, args.seed, args.reports)