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
    metric    = CFG["training"]["metric"]
    seed      = str(CFG["training"]["random_state"])
    mlflowuri = CFG["runtime"]["mlflow_tracking_uri"]
    expname   = CFG.get("pycaret", {}).get("experiment_name", "ecobici_pycaret")

    folds = str(CFG["training"].get("folds", 3))
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
    metric    = CFG["training"]["metric"]
    seed      = str(CFG["training"]["random_state"])
    mlflowuri = CFG["runtime"]["mlflow_tracking_uri"]
    expname   = CFG.get("flaml", {}).get("experiment", "ecobici_flaml")
    time_budget = str(CFG.get("flaml", {}).get("time_budget_sec", 600))

    outdir = ROOT / models_dir / "06_flaml"

    cmd = [
        sys.executable, S["automl_flaml"],
        "--train", train_p, "--val", val_p, "--test", test_p,
        "--outdir", outdir, "--reports", reports,
        "--y", y, "--id", sid, "--ts", ts,
        "--time-budget", time_budget,
        "--metric", metric, "--seed", seed,
        "--mlflow-uri", mlflowuri, "--experiment", expname,
    ]
    return sh(cmd)

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
# MAIN
# ----------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path as _P

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
        print("▶ 1/4 prepare");   rc=step_prepare();      assert rc==0, "prepare_features falló"
        print("▶ 2/4 train");     rc=step_train();        assert rc==0, "train falló"
        print("▶ 3/4 promote");   rc=step_promote();      assert rc==0, "compare_register falló"
        print("▶ 4/4 predict");   rc=step_predict_tiles();assert rc==0, "predict/tiles falló"

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