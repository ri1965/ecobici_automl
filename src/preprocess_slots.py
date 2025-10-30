# src/preprocess_slots.py
# -*- coding: utf-8 -*-
"""
Genera targets por franjas fijas (09:00, 13:00, 16:00, 20:00) para cada (station_id, día).
Salida: data/curated/ecobici_slots_train.parquet
NO crea features (eso es el Paso 2).
"""

import argparse
from typing import Optional
import pandas as pd
import numpy as np

DEFAULT_SLOTS = ["09:00", "13:00", "16:00", "20:00"]
DEFAULT_TZ = "America/Argentina/Buenos_Aires"

# ----------------------- Helpers -----------------------

def _find_ts_col(df: pd.DataFrame) -> str:
    candidates = ["ts", "ts_local", "timestamp", "datetime", "time"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No se encontró columna temporal. Probé {candidates}.")

def _find_y_col(df: pd.DataFrame) -> str:
    candidates = [
        "y",
        "bikes", "bikes_available", "available_bikes",
        "num_bikes_available",  # GBFS típico
        "target", "value",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No se encontró la columna objetivo. Probé {candidates}.")

def _ensure_tz(series: pd.Series, tz: str) -> pd.Series:
    # Si es naive → localiza. Si ya tiene tz → convierte a la tz pedida.
    if series.dt.tz is None:
        return series.dt.tz_localize(tz)
    return series.dt.tz_convert(tz)

def _normalize_station_id(s: pd.Series) -> pd.Series:
    # Si todos son dígitos, pasa a int64; si no, mantené string
    s_str = s.astype(str)
    if s_str.str.fullmatch(r"\d+").all():
        return pd.to_numeric(s_str, errors="coerce").astype("Int64")
    return s_str

# ----------------------- Core -----------------------

def build_slot_targets(
    df_raw: pd.DataFrame,
    slots=DEFAULT_SLOTS,
    tz: str = DEFAULT_TZ,
    how: str = "asof",
    asof_tolerance: str = "5min",
    lead_minutes: int = 10,
    y_col_name: Optional[str] = None,
    ts_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Crea una fila por (station_id, fecha, slot).
    t_slot es la marca exacta del slot (ej. 2025-10-28 13:00:00-03:00).
    'how':
      - 'asof'  → toma el último valor <= t_slot con tolerancia (backward, sin fuga).
      - 'exact' → exige observación exacta en t_slot (si no, y = NaN).
    """
    # ---- 1) Normalizar columnas básicas
    ts_col = ts_col_name if ts_col_name else _find_ts_col(df_raw)
    y_col  = y_col_name  if y_col_name  else _find_y_col(df_raw)

    if "station_id" not in df_raw.columns:
        raise ValueError("Falta la columna 'station_id' en el input.")

    df = df_raw.copy()
    df["station_id"] = _normalize_station_id(df["station_id"])
    df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    if df["ts"].isna().any():
        n_bad = int(df["ts"].isna().sum())
        raise ValueError(f"Hay {n_bad} timestamps no parseables en el input.")
    df["ts"] = _ensure_tz(df["ts"], tz)
    df["y"] = pd.to_numeric(df[y_col], errors="coerce")

    df = df[["station_id", "ts", "y"]].dropna(subset=["station_id", "ts"])

    # ---- 2) Armar grilla de (station_id, fecha, slots)
    start_day = df["ts"].dt.floor("D").min()
    end_day   = df["ts"].dt.floor("D").max()
    all_days  = pd.date_range(start_day, end_day, freq="D", tz=tz)

    stations = df["station_id"].dropna().unique()
    grid = []
    for sid in stations:
        for d in all_days:
            for sl in slots:
                hh, mm = sl.split(":")
                t_slot = d.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
                grid.append((sid, t_slot, sl))
    slots_df = pd.DataFrame(grid, columns=["station_id", "t_slot", "slot_label"])

    # Limitar al rango observado
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    slots_df = slots_df[(slots_df["t_slot"] >= min_ts) & (slots_df["t_slot"] <= max_ts)]

    # ---- 3) Obtener y en t_slot
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)
    slots_df = slots_df.sort_values(["station_id", "t_slot"]).reset_index(drop=True)

    if how == "exact":
        merged = slots_df.merge(
            df.rename(columns={"ts": "t_slot"})[["station_id", "t_slot", "y"]],
            on=["station_id", "t_slot"],
            how="left",
        )
        merged["ts_matched"] = merged["t_slot"]
    else:
        tol = pd.Timedelta(asof_tolerance)
        merged_list = []
        for sid, dfg in df.groupby("station_id", sort=False):
            dfg = dfg[["ts", "y"]].sort_values("ts")
            slg = slots_df[slots_df["station_id"] == sid][["t_slot", "slot_label"]].sort_values("t_slot")
            mg = pd.merge_asof(
                slg,
                dfg,
                left_on="t_slot",
                right_on="ts",
                direction="backward",
                tolerance=tol,
            )
            mg = mg.rename(columns={"y": "y_obs", "ts": "ts_matched"})
            mg["station_id"] = sid
            merged_list.append(mg)

        merged = pd.concat(merged_list, ignore_index=True)
        merged = merged[["station_id", "t_slot", "slot_label", "y_obs", "ts_matched"]].rename(columns={"y_obs": "y"})

    # ---- 4) Trazabilidad
    merged["generated_from"] = "preprocess/slots_v1"
    merged["lead_minutes"] = int(lead_minutes)

    # Orden amistoso
    base_cols = ["station_id", "t_slot", "slot_label", "y", "generated_from", "lead_minutes", "ts_matched"]
    extra = [c for c in merged.columns if c not in base_cols]
    merged = merged[base_cols + extra]

    return merged

# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet de entrada con la serie consolidada (model_ready).")
    parser.add_argument("--output", required=True, help="Parquet de salida con targets por franja.")
    parser.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS, help='Franjas "HH:MM". Ej: 09:00 13:00 16:00 20:00')
    parser.add_argument("--tz", default=DEFAULT_TZ, help="Zona horaria (default America/Argentina/Buenos_Aires).")
    parser.add_argument("--how", choices=["asof", "exact"], default="asof", help="Matching del target en t_slot.")
    parser.add_argument("--asof_tolerance", default="5min", help="Tolerancia para asof backward (ej. 5min).")
    parser.add_argument("--lead_minutes", type=int, default=10, help="Δ a usar en features (documental en este paso).")
    parser.add_argument("--ycol", default=None, help="Nombre exacto de la columna objetivo (p.ej. num_bikes_available).")
    parser.add_argument("--tscol", default=None, help="Nombre exacto de la columna temporal (p.ej. ts_local).")
    args = parser.parse_args()

    df_raw = pd.read_parquet(args.input)
    out = build_slot_targets(
        df_raw=df_raw,
        slots=args.slots,
        tz=args.tz,
        how=args.how,
        asof_tolerance=args.asof_tolerance,
        lead_minutes=args.lead_minutes,
        y_col_name=args.ycol,
        ts_col_name=args.tscol,
    )
    out.to_parquet(args.output, index=False)
    print(f"✅ Slots generados: {len(out):,} filas → {args.output}")

if __name__ == "__main__":
    main()