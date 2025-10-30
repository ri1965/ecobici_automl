# src/predict_slots.py
# -*- coding: utf-8 -*-
"""
Genera predicciones por franja horaria usando los modelos entrenados por slot.
Compatible con PyCaret 2.x y 3.x.
"""

import argparse
import os
import pandas as pd
from pycaret.regression import load_model, predict_model

DEFAULT_SLOTS = ["09:00", "13:00", "16:00", "20:00"]
DEFAULT_TZ = "America/Argentina/Buenos_Aires"

def _select_feature_cols(df: pd.DataFrame, target: str = "y") -> list:
    num = df.select_dtypes(include=["number"]).copy()
    if target in num.columns:
        num = num.drop(columns=[target])
    return num.columns.tolist()

def predict_for_slot(df_feats: pd.DataFrame, slot_label: str, models_dir: str, out_dir: str) -> str:
    sub = df_feats[df_feats["slot_label"] == slot_label].copy()
    if sub.empty:
        print(f"⚠️ Sin filas para {slot_label}; se omite.")
        return ""

    model_path = os.path.join(models_dir, slot_label.replace(":", ""), "best_model")
    if not os.path.exists(model_path + ".pkl"):
        print(f"⚠️ No se encontró modelo para {slot_label}: {model_path}.pkl")
        return ""

    print(f"🔹 Generando predicciones para franja {slot_label} ({len(sub):,} filas)...")
    model = load_model(model_path)

    feat_cols = _select_feature_cols(sub, target="y")
    preds = predict_model(model, data=sub[feat_cols])

    # Detectar nombre de columna de salida (Label o prediction_label)
    if "Label" in preds.columns:
        pred_col = "Label"
    elif "prediction_label" in preds.columns:
        pred_col = "prediction_label"
    else:
        raise KeyError(f"No se encontró columna de predicción en output de PyCaret. Columns: {preds.columns.tolist()}")

    sub["yhat"] = preds[pred_col].values
    sub["generated_at"] = pd.Timestamp.now(tz=DEFAULT_TZ)

    slot_dir = os.path.join(out_dir, slot_label.replace(":", ""))
    os.makedirs(slot_dir, exist_ok=True)
    out_path = os.path.join(slot_dir, "predictions.parquet")

    sub_out = sub[["station_id", "t_slot", "slot_label", "yhat", "generated_at"]]
    sub_out.to_parquet(out_path, index=False)

    print(f"✅ Predicciones {slot_label}: {len(sub_out):,} filas → {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet con features por franja.")
    parser.add_argument("--models_dir", default="models/slots")
    parser.add_argument("--out_dir", default="predictions/slots")
    parser.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    for sl in args.slots:
        predict_for_slot(df, sl, args.models_dir, args.out_dir)

if __name__ == "__main__":
    main()