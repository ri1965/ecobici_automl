# src/train_slots.py
# -*- coding: utf-8 -*-
import argparse, os
import pandas as pd
import numpy as np
from packaging import version
import pycaret
from pycaret.regression import setup, compare_models, finalize_model, save_model, pull

INCLUDE_MODELS = ["lr","lasso","ridge","en","knn","dt","rf","et","gbr","huber"]

def _select_numeric_features(df: pd.DataFrame, target: str) -> list:
    num = df.select_dtypes(include=["number"]).copy()
    if target in num.columns:
        num = num.drop(columns=[target])
    # descartar columnas con >80% NaN
    nn_rate = 1 - (num.isna().mean())
    keep = nn_rate[nn_rate >= 0.20].index.tolist()
    num = num[keep]
    # descartar varianza cero (ignora NaN)
    var = num.var(numeric_only=True, ddof=0)
    keep2 = var[var > 0].index.tolist()
    return keep2

def train_per_slot(df: pd.DataFrame, slot_label: str, target="y",
                   out_dir="models/slots", test_size=0.2, min_rows=100):
    # filtrar por franja y y no nula
    subset = df[(df["slot_label"] == slot_label) & (df[target].notna())].copy()
    total_raw = (df["slot_label"] == slot_label).sum()
    kept = len(subset); dropped = total_raw - kept
    print(f"\n🕒 {slot_label}: {total_raw:,} filas totales | descartadas y NaN: {dropped:,} | a entrenar: {kept:,}")
    if kept < min_rows:
        print(f"⚠️ Pocas filas para {slot_label} (<{min_rows}). Se omite.")
        return None

    # columnas no modelables
    drop_cols = ["station_id","t_slot","slot_label","generated_from"]
    subset = subset.drop(columns=[c for c in drop_cols if c in subset.columns], errors="ignore")

    # seleccionar features numéricas útiles
    feat_cols = _select_numeric_features(subset, target)
    if not feat_cols:
        print(f"⚠️ Sin features numéricas útiles para {slot_label}. Se omite.")
        return None

    subset = subset[[target] + feat_cols].copy()

    # preparar setup (compat PyCaret 2/3)
    exp_name = f"ecobici_slot_{slot_label.replace(':','')}"
    pc_version = version.parse(pycaret.__version__)
    kwargs = dict(
        data=subset, target=target, session_id=42,
        train_size=1 - test_size, fold=5, experiment_name=exp_name, verbose=False,
    )
    if pc_version < version.parse("3.0"):
        kwargs.update(dict(silent=True, log_experiment=False))

    setup(**kwargs)

    # forzar modelos clásicos; PyCaret evaluará los disponibles
    try:
        best = compare_models(include=INCLUDE_MODELS, sort="MAE")
    except Exception as e:
        print(f"⚠️ compare_models falló con include={INCLUDE_MODELS}. Intento libre. Motivo: {e}")
        best = compare_models(sort="MAE")

    final = finalize_model(best)

    slot_dir = os.path.join(out_dir, slot_label.replace(":", ""))
    os.makedirs(slot_dir, exist_ok=True)
    save_model(final, os.path.join(slot_dir, "best_model"))
    print(f"✅ Modelo de {slot_label} guardado en {slot_dir}/best_model.pkl")

    res = pull()
    if isinstance(res, pd.DataFrame):
        res.to_csv(os.path.join(slot_dir, "training_summary.csv"), index=False)
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", default="y")
    ap.add_argument("--out_dir", default="models/slots")
    ap.add_argument("--slots", nargs="+", default=["09:00","13:00","16:00","20:00"])
    ap.add_argument("--min_rows", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    for sl in args.slots:
        train_per_slot(df, sl, target=args.target, out_dir=args.out_dir, min_rows=args.min_rows)

if __name__ == "__main__":
    main()