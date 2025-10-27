#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run.py — MVP simple con 4 pasos

import sys, subprocess, time, yaml
from pathlib import Path

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

def sh(cmd): print("CMD>", " ".join(map(str, cmd))); return subprocess.call(list(map(str,cmd)), cwd=ROOT)

def step_prepare():
    inp=ROOT/CFG["paths"]["features_in"]; out=ROOT/CFG["paths"]["curated_file"]
    ts=CFG["data"]["ts_col"]; sid=CFG["data"]["id_col"]; y=CFG["data"]["target"]
    return sh([sys.executable, S["prepare_features"], "--in", inp, "--out", out, "--ts", ts, "--id", sid, "--y", y])

def step_train():
    # Configs
    curated   = ROOT/CFG["paths"]["curated_file"]   # (no lo usan estos scripts, van por splits)
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
    fw        = CFG["training"]["framework"].lower()

    # Elegir script y outdir según framework
    if fw == "flaml":
        script  = S["automl_flaml"]
        outdir  = ROOT / models_dir / "06_flaml"
        expname = CFG.get("flaml", {}).get("experiment", "ecobici_automl_flaml")
        time_budget = str(CFG.get("flaml", {}).get("time_budget_sec", 600))
        cmd = [
            sys.executable, script,
            "--train", train_p, "--val", val_p, "--test", test_p,
            "--outdir", outdir, "--reports", reports,
            "--y", y, "--id", sid, "--ts", ts,
            "--time-budget", time_budget,
            "--metric", metric, "--seed", seed,
            "--mlflow-uri", mlflowuri, "--experiment", expname,
        ]
    else:
        # default: pycaret
        script  = S["automl_pycaret"]
        outdir  = ROOT / models_dir / "03A_pycaret"
        expname = CFG.get("pycaret", {}).get("experiment_name", "ecobici_pycaret_automl")
        folds   = str(CFG["training"].get("folds", 3))
        cmd = [
            sys.executable, script,
            "--train", train_p, "--val", val_p, "--test", test_p,
            "--outdir", outdir, "--reports", reports,
            "--y", y, "--id", sid, "--ts", ts,
            "--folds", folds,
            "--metric", metric, "--seed", seed,
            "--mlflow-uri", mlflowuri, "--experiment", expname,
        ]

    print("CMD>", " ".join(map(str, cmd)))
    return sh(cmd)

def step_promote():
    reports   = ROOT/CFG["paths"]["reports_dir"]
    mlflowuri = CFG["runtime"]["mlflow_tracking_uri"]
    experiment= CFG["pycaret"].get("experiment_name", "ecobici_pycaret")
    model_name = "ecobici_champion"
    stage      = "Production"

    bench_csv   = ROOT / reports / "automl_metrics_by_split.csv"
    pycaret_pkl = ROOT / CFG["paths"]["models_dir"] / "03A_pycaret" / "pycaret_best_model.pkl"

    # 👇 usar la columna real del CSV (val_rmse o test_rmse)
    metric_to_use = "val_rmse"

    cmd = [
        sys.executable, S["compare_register"],
        "--bench", bench_csv,
        "--reports", reports,
        "--metric", metric_to_use,
        "--mlflow-uri", mlflowuri,
        "--experiment", experiment,
        "--model-name", model_name,
        "--stage", stage,
        "--pycaret-model-path", pycaret_pkl,
    ]
    print("CMD>", " ".join(map(str, cmd)))
    return sh(cmd)

def step_predict_tiles():
    # --- 1) Predicción batch ---
    input_file   = ROOT / CFG["paths"]["curated_file"]
    model_path   = ROOT / CFG["registry"]["champion_path"]
    train_split  = ROOT / CFG["paths"]["train_split"]
    out_pred_dir = ROOT / CFG["paths"]["predictions_dir"]
    asof         = CFG["inference"]["asof"]
    horizons     = [str(h) for h in CFG["inference"]["horizons"]]
    y_col  = CFG["data"]["target"]
    id_col = CFG["data"]["id_col"]
    ts_col = CFG["data"]["ts_col"]

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
    print("CMD>", " ".join(map(str, cmd_pred)))
    rc1 = sh(cmd_pred)
    if rc1 != 0:
        return rc1

    # --- 2) Elegir el archivo de predicciones a usar para tiles ---
    pred_path = out_pred_dir / "latest.parquet"
    if not pred_path.exists():
        # fallback: tomar el parquet más nuevo de la carpeta
        parqs = sorted(out_pred_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not parqs:
            print("[ERROR] No encontré archivos de predicción en", out_pred_dir)
            return 1
        pred_path = parqs[0]

    from datetime import datetime

    # --- 3) Generar tiles ---
    tiles_dir = ROOT / CFG["paths"]["tiles_dir"]
    stations  = ROOT / CFG["paths"]["stations_file"]
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # 👇 OUT debe ser un ARCHIVO, no una carpeta
    out_tiles = tiles_dir / f"tiles_{datetime.now():%Y%m%d-%H%M%S}.parquet"

    cmd_tiles = [
        sys.executable, S["make_tiles"],
        "--pred", str(pred_path),
        "--stations", str(stations),
        "--out", str(out_tiles),   # <— archivo .parquet destino
    ]
    print("CMD>", " ".join(map(str, cmd_tiles)))
    return sh(cmd_tiles)

if __name__=="__main__":
    t0=time.time()
    for k,v in S.items(): assert v.exists(), f"Falta script: {v}"
    assert (ROOT/"config/config.yaml").exists(), "Falta config"

    print("▶ 1/4 prepare");   rc=step_prepare();      assert rc==0, "prepare_features falló"
    print("▶ 2/4 train");     rc=step_train();        assert rc==0, "train falló"
    print("▶ 3/4 promote");   rc=step_promote();      assert rc==0, "compare_register falló"
    print("▶ 4/4 predict");   rc=step_predict_tiles();assert rc==0, "predict/tiles falló"

    print(f"✅ Listo en {round(time.time()-t0,2)}s")