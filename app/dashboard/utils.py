# utils.py (reemplazar la función load_predictions por esta)

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

def _concat_parquets(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            df["__srcfile__"] = str(p)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out.attrs["__source_path__"] = ", ".join([str(p) for p in paths])
    return out

def load_predictions() -> pd.DataFrame:
    """Carga predicciones en modo slots si existen; sino, modo horizonte (legacy)."""
    # --- 1) Intentar modo SLOTS ---
    slots_dir = Path("predictions/slots")
    slot_files = sorted(slots_dir.glob("*/predictions.parquet"))
    if slot_files:
        df = _concat_parquets(slot_files)
        # columnas mínimas esperadas en slots
        # station_id | t_slot | slot_label | yhat | generated_at
        # Normalización
        if "station_id" in df.columns:
            s = df["station_id"].astype(str)
            df["station_id"] = pd.to_numeric(s, errors="coerce").astype("Int64") if s.str.fullmatch(r"\d+").all() else s
        if "t_slot" in df.columns:
            df["t_slot"] = pd.to_datetime(df["t_slot"], errors="coerce", utc=False)
        if "yhat" in df.columns:
            df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
        # asegurar columna 'slot_label'
        if "slot_label" not in df.columns:
            # si no vino, tratar de inferirla del path (0900, 1300, etc.)
            try:
                df["slot_label"] = df["__srcfile__"].str.extract(r"/slots/(\d{4})/")[0].str.replace(
                    r"(\d{2})(\d{2})", r"\1:\2", regex=True
                )
            except Exception:
                pass
        return df

    # --- 2) Fallback: modo HORIZONTE (legacy) ---
    legacy_files = sorted(Path("predictions").glob("*.parquet"))
    if legacy_files:
        df = _concat_parquets(legacy_files)
        # columnas típicas legacy: station_id | ts | h | yhat ...
        if "station_id" in df.columns:
            s = df["station_id"].astype(str)
            df["station_id"] = pd.to_numeric(s, errors="coerce").astype("Int64") if s.str.fullmatch(r"\d+").all() else s
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
        if "yhat" in df.columns:
            df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
        return df

    # --- 3) Si no hay nada, devolver DF vacío con columnas estándar ---
    empty = pd.DataFrame(columns=["station_id", "t_slot", "slot_label", "yhat"])
    empty.attrs["__source_path__"] = "(sin archivos de predicción encontrados)"
    return empty

def load_stations() -> pd.DataFrame:
    # tu versión anterior; dejo un fallback robusto por si acaso
    for p in [
        Path("data/curated/station_information.parquet"),
        Path("data/raw/station_information.parquet"),
    ]:
        if p.exists():
            df = pd.read_parquet(p)
            return df
    return pd.DataFrame(columns=["station_id","name","lat","lon","capacity"])