# src/prepare_features.py
import os
import argparse
import numpy as np
import pandas as pd

def coerce_station_id_dtype(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    if s.str.fullmatch(r"\d+").all():
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    return s

def main(in_path: str, out_path: str, ts_col: str, id_col: str, y_col: str, export_csv: str | None):
    assert os.path.exists(in_path), f"No existe {in_path}"
    df = pd.read_parquet(in_path)

    # --- Validaciones mínimas
    for c in (ts_col, id_col, y_col):
        if c not in df.columns:
            raise ValueError(f"Falta la columna requerida: '{c}'. "
                             f"Columnas disponibles: {df.columns.tolist()}")

    # --- Tipos y limpieza básica
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().any():
        n_bad = int(df[ts_col].isna().sum())
        print(f"[WARN] {n_bad} filas con {ts_col} inválido → se descartarán.")
        df = df[~df[ts_col].isna()].copy()

    df[id_col] = coerce_station_id_dtype(df[id_col])

    # Orden temporal por estación y deduplicación
    df = (df.sort_values([id_col, ts_col])
            .drop_duplicates(subset=[id_col, ts_col])
            .reset_index(drop=True))

    # --- Señales temporales
    df["hour"]       = df[ts_col].dt.hour
    df["dow"]        = df[ts_col].dt.dayofweek   # 0 = lunes
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["month"]      = df[ts_col].dt.month

    # Cíclicas (24h)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Lags y rolling SIN fuga (por estación)
    df["y_lag1"] = df.groupby(id_col)[y_col].shift(1)
    df["y_lag2"] = df.groupby(id_col)[y_col].shift(2)
    df["y_ma3"]  = (df.groupby(id_col)[y_col]
                      .transform(lambda s: s.shift(1).rolling(3).mean()))

    # Tipos numéricos para derivadas
    for c in ["y_lag1", "y_lag2", "y_ma3"]:
        df[c] = df[c].astype(float)

    # Eliminar filas sin lags (primeras por estación)
    mask_ok = ~df[["y_lag1", "y_lag2", "y_ma3"]].isna().any(axis=1)
    dropped = int((~mask_ok).sum())
    if dropped:
        print(f"[INFO] Filas iniciales sin historial eliminadas: {dropped}")

    df_out = df.loc[mask_ok].copy()

    # --- Guardar parquet canónico
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    # (Opcional) exportar CSV como vista (no como fuente)
    if export_csv:
        os.makedirs(os.path.dirname(export_csv), exist_ok=True)
        df_out.to_csv(export_csv, index=False)

    # --- Logs útiles
    print(f"[OK] Guardado {out_path}")
    if export_csv:
        print(f"[OK] Exportado CSV (vista): {export_csv}")
    print(f"Filas/Cols: {df_out.shape[0]}/{df_out.shape[1]}")
    print("Columnas clave:", [id_col, ts_col, y_col])
    print("Primeras filas:")
    print(df_out[[id_col, ts_col, y_col, "y_lag1", "y_ma3"]].head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preparación de features sin fuga (lags/rolling) desde un Parquet crudo.")
    ap.add_argument("--in",   dest="in_path",   required=True, help="Parquet crudo (p.ej. status_clean.parquet)")
    ap.add_argument("--out",  dest="out_path",  required=True, help="Parquet listo para modelar")
    ap.add_argument("--ts",   dest="ts_col",    default="ts_local", help="Nombre de la columna de timestamp (default: ts_local)")
    ap.add_argument("--id",   dest="id_col",    default="station_id", help="ID de estación (default: station_id)")
    ap.add_argument("--y",    dest="y_col",     default="num_bikes_available", help="Target (default: num_bikes_available)")
    ap.add_argument("--csv",  dest="export_csv", default="", help="(Opcional) Ruta para exportar CSV de vista")
    args = ap.parse_args()

    export_csv = args.export_csv if args.export_csv.strip() else None
    main(args.in_path, args.out_path, args.ts_col, args.id_col, args.y_col, export_csv)