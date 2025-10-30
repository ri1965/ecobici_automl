# src/features_slots.py
# -*- coding: utf-8 -*-
"""
Adjunta features sin fuga a la tabla de slots (targets por franja).
Entrada:
  - data/curated/ecobici_slots_train.parquet  (station_id, t_slot, slot_label, y, lead_minutes, ts_matched?)
  - data/curated/ecobici_model_ready.parquet  (station_id, ts_local, num_bikes_available, etc.)
Salida:
  - data/curated/ecobici_slots_train_features.parquet
"""

import argparse
import pandas as pd
import numpy as np

DEFAULT_TZ = "America/Argentina/Buenos_Aires"

def _ensure_tz(series: pd.Series, tz: str) -> pd.Series:
    if series.dt.tz is None:
        return series.dt.tz_localize(tz)
    return series.dt.tz_convert(tz)

def _normalize_station_id(s: pd.Series) -> pd.Series:
    s_str = s.astype(str)
    if s_str.str.fullmatch(r"\d+").all():
        return pd.to_numeric(s_str, errors="coerce").astype("Int64")
    return s_str

def build_features_for_slots(
    df_slots: pd.DataFrame,
    df_raw: pd.DataFrame,
    tz: str,
    ts_col_raw: str,
    y_col_raw: str,
    max_lag: int = 12,
    ma_windows: tuple = (3, 6, 12),
) -> pd.DataFrame:
    # --- Normalizar entradas
    df_slots = df_slots.copy()
    df_slots["station_id"] = _normalize_station_id(df_slots["station_id"])
    df_slots["t_slot"] = pd.to_datetime(df_slots["t_slot"], errors="coerce")
    df_slots["t_slot"] = _ensure_tz(df_slots["t_slot"], tz)
    if "lead_minutes" not in df_slots.columns:
        df_slots["lead_minutes"] = 10  # fallback
    if "generated_from" not in df_slots.columns:
        df_slots["generated_from"] = "preprocess/slots_v1"

    df_raw = df_raw.copy()
    df_raw["station_id"] = _normalize_station_id(df_raw["station_id"])
    df_raw["ts"] = pd.to_datetime(df_raw[ts_col_raw], errors="coerce")
    df_raw["ts"] = _ensure_tz(df_raw["ts"], tz)
    df_raw["y_raw"] = pd.to_numeric(df_raw[y_col_raw], errors="coerce")
    df_raw = df_raw.dropna(subset=["station_id", "ts"]).sort_values(["station_id", "ts"])

    # --- Precalcular lags y MAs por estación en la serie completa
    feats_list = []
    for sid, g in df_raw.groupby("station_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        # Lags
        for k in range(1, max_lag + 1):
            g[f"y_lag{k}"] = g["y_raw"].shift(k)
        # Medias móviles (pasadas)
        for w in ma_windows:
            g[f"y_ma{w}"] = g["y_raw"].rolling(window=w, min_periods=1).mean()
        # Tendencia corta
        g["y_diff1"] = g["y_raw"] - g["y_lag1"]
        g["station_id"] = sid
        feats_list.append(g)

    df_feats_full = pd.concat(feats_list, ignore_index=True)

    # --- Para cada fila de slot, tomar features hasta (t_slot - lead_minutes)
    merged_rows = []
    for sid, sl in df_slots.groupby("station_id", sort=False):
        base = df_feats_full[df_feats_full["station_id"] == sid].sort_values("ts")
        slc = sl.sort_values("t_slot").copy()

        # Construir el límite: t_slot - lead_minutes
        slc["t_cut"] = slc["t_slot"] - pd.to_timedelta(slc["lead_minutes"], unit="m")

        # Conservar lead_minutes para que esté disponible luego del merge
        mg = pd.merge_asof(
            slc[["t_slot", "slot_label", "t_cut", "lead_minutes"]],
            base,
            left_on="t_cut",
            right_on="ts",
            direction="backward",
            allow_exact_matches=True,
        )
        mg["station_id"] = sid
        merged_rows.append(mg)

    out = pd.concat(merged_rows, ignore_index=True)

    # --- Selección/renombrado y features calendarias del propio slot
    out = out.rename(columns={"y_raw": "y_hist_at_cut"})
    out["hour"] = out["t_slot"].dt.hour
    out["dow"] = out["t_slot"].dt.dayofweek
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    # --- Unir el y del slot (y trazabilidad) desde df_slots
    out = out.merge(
        df_slots[["station_id", "t_slot", "slot_label", "y", "lead_minutes", "generated_from"]],
        on=["station_id", "t_slot", "slot_label", "lead_minutes"],
        how="left",
        suffixes=("", "_slotdup"),
    )

    # --- Orden final
    keep_cols = [
        "station_id", "t_slot", "slot_label",
        "y", "lead_minutes", "generated_from",
        "y_hist_at_cut",
    ]
    keep_cols += [c for c in out.columns if c.startswith("y_lag")]
    keep_cols += [c for c in out.columns if c.startswith("y_ma")]
    keep_cols += ["y_diff1", "hour", "dow", "is_weekend", "hour_sin", "hour_cos"]

    out = out[keep_cols].sort_values(["station_id", "t_slot"]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots_input", required=True, help="Parquet con targets por franja (ecobici_slots_train.parquet).")
    ap.add_argument("--raw_input", required=True, help="Parquet base consolidado (ecobici_model_ready.parquet).")
    ap.add_argument("--output", required=True, help="Parquet de salida con features.")
    ap.add_argument("--tz", default=DEFAULT_TZ)
    ap.add_argument("--tscol_raw", default="ts_local")
    ap.add_argument("--ycol_raw", default="num_bikes_available")
    ap.add_argument("--max_lag", type=int, default=12)
    ap.add_argument("--ma_windows", nargs="+", type=int, default=[3,6,12])
    args = ap.parse_args()

    df_slots = pd.read_parquet(args.slots_input)
    df_raw   = pd.read_parquet(args.raw_input)

    out = build_features_for_slots(
        df_slots=df_slots,
        df_raw=df_raw,
        tz=args.tz,
        ts_col_raw=args.tscol_raw,
        y_col_raw=args.ycol_raw,
        max_lag=args.max_lag,
        ma_windows=tuple(args.ma_windows),
    )
    out.to_parquet(args.output, index=False)
    print(f"✅ Features por franjas generadas: {len(out):,} filas → {args.output}")

if __name__ == "__main__":
    main()