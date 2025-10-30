#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run.py — Pipeline simple con competencia PyCaret + FLAML

import sys, subprocess, time, yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
CFG  = yaml.safe_load(open(ROOT/"config/config.yaml"))
S = {
  "prepare_features": ROOT/"src/prepare_features.py",
  "automl_pycaret":   ROOT/"src/automl_pycaret.py",
  "automl_flaml":     ROOT/"src/automl_flaml.py",
  "compare_register": ROOT/"src/compare_and_register.py",
  "predict_batch":    ROOT/"src/predict_batch.py",
  "make_tiles":       ROOT/"src/make_tiles.py",
}

def sh(cmd):
    print("CMD>", " ".join(map(str, cmd)))
    return subprocess.call(list(map(str, cmd)), cwd=ROOT)

# ----------------------------
# 1) PREPARE FEATURES
# ----------------------------
def step_prepare():
    inp = ROOT/CFG["paths"]["features_in"]
    out = ROOT/CFG["paths"]["curated_file"]
    ts  = CFG["data"]["ts_col"]
    sid = CFG["data"]["id_col"]
    y   = CFG["data"]["target"]
    return sh([sys.executable, S["prepare_features"], "--in", inp, "--out", out, "--ts", ts, "--id", sid, "--y", y])

# ----------------------------
# 2A) TRAIN PYCARET
# ----------------------------
def step_train_pycaret():
    if not S["automl_pycaret"].exists():
        print("[INFO] automl_pycaret.py no existe; salto este paso.")
        return 0

    train_p   = ROOT/CFG["paths"]["train_split"]
    val_p     = ROOT/CFG["paths"]["val_split"]
    test_p    = ROOT/CFG["paths"]["test_split"]
    reports   = ROOT/CFG["paths"]["reports_dir"]
    models_dir= ROOT/CFG["paths"]["models_dir"]
    y         = CFG["data"]["target"]
    sid       = CFG["data"]["id_col"]
    ts        = CFG["data"]["ts_col"]

    auto_cfg  = CFG.get("automl", {})
    metric    = auto_cfg.get("metric", CFG["training"]["metric"])
    folds     = str(auto_cfg.get("folds", 3))
    tune      = bool(auto_cfg.get("tune", False))
    tune_it   = int(auto_cfg.get("tune_iters", 50))

    seed      = str(CFG["training"]["random_state"])
    mlflowuri = CFG["runtime"]["mlflow_tracking_uri"]
    expname   = CFG.get("pycaret", {}).get("experiment_name", "ecobici_pycaret")

    outdir = ROOT / models_dir / "03A_pycaret"

    cmd = [
        sys.executable, S["automl_pycaret"],
        "--train", train_p, "--val", val_p, "--test", test_p,
        "--outdir", outdir, "--reports", reports,
        "--y", y, "--id", sid, "--ts", ts,
        "--folds", folds,
        "--metric", metric, "--seed", seed,
        "--mlflow-uri", mlflowuri, "--experiment", expname,
    ]
    if tune:
        cmd += ["--tune", "--tune-iters", str(tune_it)]

    return sh(cmd)

# ----------------------------
# 2B) TRAIN FLAML
# ----------------------------
def step_train_flaml():
    if not S["automl_flaml"].exists():
        print("[INFO] automl_flaml.py no existe; salto este paso.")
        return 0

    train_p   = ROOT/CFG["paths"]["train_split"]
    val_p     = ROOT/CFG["paths"]["val_split"]
    test_p    = ROOT/CFG["paths"]["test_split"]
    reports   = ROOT/CFG["paths"]["reports_dir"]
    models_dir= ROOT/CFG["paths"]["models_dir"]
    y         = CFG["data"]["target"]
    sid       = CFG["data"]["id_col"]
    ts        = CFG["data"]["ts_col"]

    flaml_cfg  = CFG.get("flaml", {})
    time_budget = int(flaml_cfg.get("time_budget_sec", 600))
    refine_time = int(flaml_cfg.get("refine_time_sec", 0))
    metric      = CFG["automl"].get("metric", "RMSE")
    seed        = int(CFG["training"]["random_state"])
    mlflowuri   = CFG["runtime"]["mlflow_tracking_uri"]
    expname     = flaml_cfg.get("experiment", "ecobici_automl_flaml")

    outdir = ROOT / models_dir / "06_flaml"

    cmd = [
        sys.executable, S["automl_flaml"],
        "--train", train_p, "--val", val_p, "--test", test_p,
        "--outdir", outdir, "--reports", reports,
        "--y", y, "--id", sid, "--ts", ts,
        "--time-budget", str(time_budget),
        "--refine-time", str(refine_time),
        "--metric", metric, "--seed", str(seed),
        "--mlflow-uri", mlflowuri, "--experiment", expname,
    ]
    return sh(cmd)

# ----------------------------
# Selección automática del framework (PyCaret / FLAML)
# ----------------------------

def step_train():
    fw = str(CFG["automl"].get("framework", "pycaret")).lower()
    if fw == "pycaret":
        return step_train_pycaret()
    elif fw == "flaml":
        return step_train_flaml()
    elif fw == "both":
        rc = step_train_pycaret()
        if rc != 0:
            return rc
        rc = step_train_flaml()
        if rc != 0:
            return rc
        return 0
    else:
        print(f"[ERROR] framework no soportado: {fw}")
        return 1

# ----------------------------
# 3) COMPARE & REGISTER (Champion + alias filesystem)
# ----------------------------
def step_promote():
    reports     = ROOT / CFG["paths"]["reports_dir"]
    bench_csv   = reports / "automl_metrics_by_split.csv"
    mlflow_uri  = CFG["runtime"]["mlflow_tracking_uri"]
    experiment  = "ecobici_model_selection"
    model_name  = "ecobici_champion"
    stage       = "Production"

    pycaret_pkl = ROOT / "models" / "03A_pycaret" / "pycaret_best_model.pkl"
    flaml_pkl   = ROOT / "models" / "06_flaml"     / "flaml_automl.pkl"
    champion    = ROOT / CFG["registry"]["champion_path"]  # ej. models/Champion/best_model.pkl

    # 👇 Esta línea crea la carpeta 'models/Champion' si no existe
    champion.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, S["compare_register"],
        "--bench", str(bench_csv),
        "--reports", str(reports),
        "--metric", "val_rmse",
        "--mlflow-uri", mlflow_uri,
        "--experiment", experiment,
        "--model-name", model_name,
        "--stage", stage,
        "--pycaret-model-path", str(pycaret_pkl),
    ]
    if flaml_pkl.exists():
        cmd += ["--flaml-model-path", str(flaml_pkl)]
    cmd += ["--champion-path", str(champion)]
    return sh(cmd)

# ----------------------------
# 4) PREDICT + TILES
# ----------------------------
def step_predict_tiles():
    # 4.1 Predicción batch
    input_file   = ROOT / CFG["paths"]["curated_file"]
    model_path   = ROOT / CFG["registry"]["champion_path"]
    train_split  = ROOT / CFG["paths"]["train_split"]
    out_pred_dir = ROOT / CFG["paths"]["predictions_dir"]
    asof         = CFG["inference"]["asof"]
    horizons     = [str(h) for h in CFG["inference"]["horizons"]]
    y_col  = CFG["data"]["target"]; id_col = CFG["data"]["id_col"]; ts_col = CFG["data"]["ts_col"]

    cmd_pred = [
        sys.executable, S["predict_batch"],
        "--model", str(model_path),
        "--input", str(input_file),
        "--train_split", str(train_split),
        "--horizons", *horizons,
        "--asof", asof,
        "--out_pred", str(out_pred_dir),
        "--y", y_col, "--id", id_col, "--ts", ts_col,
    ]
    rc1 = sh(cmd_pred)
    if rc1 != 0: return rc1

    # 4.2 Elegir predicción para tiles
    pred_path = out_pred_dir / "latest.parquet"
    if not pred_path.exists():
        parqs = sorted(out_pred_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not parqs:
            print("[ERROR] No encontré archivos de predicción en", out_pred_dir)
            return 1
        pred_path = parqs[0]

    # 4.3 Generar tiles
    tiles_dir = ROOT / CFG["paths"]["tiles_dir"]
    stations  = ROOT / CFG["paths"]["stations_file"]
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out_tiles = tiles_dir / f"tiles_{datetime.now():%Y%m%d-%H%M%S}.parquet"

    cmd_tiles = [
        sys.executable, S["make_tiles"],
        "--pred", str(pred_path),
        "--stations", str(stations),
        "--out", str(out_tiles),
    ]
    return sh(cmd_tiles)

# ----------------------------
# 4) RE-FIT DEL CHAMPION EN FULL DATA (train+val o curated)
# ----------------------------
def step_refit_champion():
    """
    - Detecta el framework ganador leyendo reports/automl_metrics_by_split.csv (split=val).
    - Arma el dataset "full" según retrain.mode (trainval|curated).
    - Reentrena con el framework correspondiente y SOBREESCRIBE:
      models/Champion/best_model.pkl
    """
    import json, joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path

    reports = ROOT / CFG["paths"]["reports_dir"]
    metrics_csv = reports / "automl_metrics_by_split.csv"
    assert metrics_csv.exists(), f"No existe {metrics_csv}; corré train/promote antes."

    # --- 0) Elegir framework ganador por val_rmse (menor es mejor)
    m = pd.read_csv(metrics_csv)
    m = m[m["split"] == "val"]
    m = m.sort_values("rmse", ascending=True)
    assert not m.empty, "No hay filas VAL en automl_metrics_by_split.csv"
    winner_framework = str(m.iloc[0]["framework"]).lower()
    print(f"[INFO] Champion framework (por val_rmse): {winner_framework}")

    # --- 1) Datos FULL según config
    mode   = str(CFG.get("retrain", {}).get("mode", "trainval")).lower()
    y_col  = CFG["data"]["target"]
    id_col = CFG["data"]["id_col"]
    ts_col = CFG["data"]["ts_col"]

    def _load_full_df():
        if mode == "curated":
            return pd.read_parquet(ROOT / CFG["paths"]["curated_file"])
        else:  # trainval
            dtr = pd.read_parquet(ROOT / CFG["paths"]["train_split"])
            dva = pd.read_parquet(ROOT / CFG["paths"]["val_split"])
            return pd.concat([dtr, dva], ignore_index=True)

    df_full = _load_full_df()
    assert not df_full.empty, "df_full vacío en refit"

    # --- 2) Helpers de preprocesado NUM-only + orden temporal
    def _to_numeric_dataset(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

        drop_cols = [c for c in [y_col, id_col, ts_col] if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore")
        X = X.select_dtypes(include=["number"]).fillna(0.0)
        return df[[y_col]].join(X)

    def _to_Xy_numeric(df: pd.DataFrame):
        df = df.copy()
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
        y = pd.to_numeric(df[y_col], errors="coerce").astype(float).values
        X = df.drop(columns=[c for c in [y_col, id_col, ts_col] if c in df.columns], errors="ignore")
        X = X.select_dtypes(include=["number"]).fillna(0.0)
        return X, y

    out_dir = ROOT / "models" / "Champion"
    out_dir.mkdir(parents=True, exist_ok=True)
    alias_path = out_dir / "best_model.pkl"  # <- alias estable que usa predict

    # --- 3) Reentrenar según framework
    if winner_framework == "pycaret":
        from pycaret.regression import setup, load_model, finalize_model, save_model

        data_full = _to_numeric_dataset(df_full)

        # Setup consistente (timeseries, sin shuffle)
        _ = setup(
            data=data_full,
            target=y_col,
            session_id=CFG["training"]["random_state"],
            fold_strategy="timeseries",
            fold=int(CFG["automl"].get("folds", 3)),
            data_split_shuffle=False,
            fold_shuffle=False,
            html=False,
            verbose=False,
            log_experiment=False,
        )

        # Cargar el modelo PyCaret ganador original:
        # Usamos el modelo guardado por PyCaret (con save_model) si existe;
        # si el Champion actual es una copia, igual es legible por load_model(strip .pkl).
        champ_file = ROOT / CFG["registry"]["champion_path"]
        base = str(champ_file).replace(".pkl", "")
        try:
            model = load_model(base)
        except Exception:
            # Fallback: si no fue guardado con save_model(), probamos joblib.load para estimador compatible
            model = joblib.load(champ_file)

        final = finalize_model(model)  # entrena en TODO el data_full del setup
        base_out = out_dir / "best_model_refit"
        save_model(final, str(base_out))  # PyCaret agrega ".pkl"
        # mover/renombrar como alias estable
        dst = out_dir / "best_model.pkl"
        if dst.exists():
            dst.unlink()
        os.replace(str(base_out) + ".pkl", dst)
        print(f"[OK] Refit Champion (PyCaret) → {dst}")
        return 0

    elif winner_framework == "flaml":
        from sklearn.base import clone
        from flaml import AutoML

        Xf, yf = _to_Xy_numeric(df_full)

        # Cargamos el objeto AutoML campeón, clonamos su estimador y lo reentrenamos
        champ_file = ROOT / CFG["registry"]["champion_path"]
        automl = joblib.load(champ_file)

        # Intento 1: clonar estimador y setear best_config
        est = automl.model.estimator
        try:
            est2 = clone(est)
        except Exception:
            est2 = est.__class__()  # fallback

        # Filtramos hyperparams compatibles
        try:
            valid_keys = set(est2.get_params().keys())
            best_cfg = {k: v for k, v in automl.best_config.items() if k in valid_keys}
            est2.set_params(**best_cfg)
        except Exception:
            pass

        est2.fit(Xf, yf)
        joblib.dump(est2, alias_path)
        print(f"[OK] Refit Champion (FLAML) → {alias_path}")
        return 0

    else:
        print(f"[WARN] Framework no reconocido para refit: {winner_framework}")
        return 0

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Runner Ecobici-AutoML")
    ap.add_argument("--mode", choices=["all", "train", "promote", "predict"],
                    default="all", help="Qué ejecutar")
    ap.add_argument("--retrain", action="store_true",
                    help="Forzar reentrenamiento (sólo útil en mode=train/all)")
    args = ap.parse_args()

    t0 = time.time()
    for k, v in S.items():
        assert v.exists(), f"Falta script: {v}"
    assert (ROOT / "config/config.yaml").exists(), "Falta config/config.yaml"

    def _ensure_curated():
        curated = ROOT / CFG["paths"]["curated_file"]
        if not curated.exists():
            print("[INFO] No existe curated_file; corriendo prepare…")
            rc = step_prepare(); assert rc == 0
        return curated

    def _ensure_champion():
        champ = ROOT / CFG["registry"]["champion_path"]
        if not champ.exists():
            print("[INFO] No existe Champion; ejecutando promote (sin reentrenar)…")
            rc = step_promote(); assert rc == 0
        return champ

    if args.mode == "all":
        print("▶ 1/5 prepare");   rc=step_prepare();         assert rc==0, "prepare_features falló"
        print("▶ 2/5 train");     rc=step_train();           assert rc==0, "train falló"
        print("▶ 3/5 promote");   rc=step_promote();         assert rc==0, "compare_register falló"
        print("▶ 4/5 refit");     rc=step_refit_champion();  assert rc==0, "refit falló"
        print("▶ 5/5 predict");   rc=step_predict_tiles();   assert rc==0, "predict/tiles falló"

    elif args.mode == "train":
        print("▶ prepare");       rc=step_prepare();      assert rc==0
        print("▶ train");         rc=step_train();        assert rc==0
        print("▶ promote");       rc=step_promote();      assert rc==0

    elif args.mode == "promote":
        print("▶ promote");       rc=step_promote();      assert rc==0

    elif args.mode == "predict":
        _ensure_curated()
        _ensure_champion()
        print("▶ predict");       rc=step_predict_tiles();assert rc==0

    print(f"✅ Listo en {round(time.time()-t0,2)}s")