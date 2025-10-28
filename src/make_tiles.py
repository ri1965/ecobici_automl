# src/make_tiles.py
# ------------------------------------------------------------
# Genera tiles georreferenciados para dashboard Ecobici
# ------------------------------------------------------------
import os, sys, argparse
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Configuración de colores (puede moverse a config.yaml)
# ------------------------------------------------------------
COLOR_RULES = {
    "high":   "#2ECC71",  # verde
    "medium": "#F1C40F",  # amarillo
    "low":    "#E74C3C",  # rojo
    "na":     "#BDC3C7"   # gris
}

# ------------------------------------------------------------
# Función principal
# ------------------------------------------------------------
def main(pred_path: str, stations_path: str, out_path: str):
    # --- Validaciones iniciales
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Archivo de predicciones no encontrado: {pred_path}")
    if not os.path.exists(stations_path):
        raise FileNotFoundError(f"Archivo de estaciones no encontrado: {stations_path}")

    print(f"📂 Leyendo predicciones desde: {pred_path}")
    preds = pd.read_parquet(pred_path)
    print(f"📂 Leyendo estaciones desde: {stations_path}")
    stations = pd.read_parquet(stations_path)

    # --- Validación de columnas clave
    for col in ["station_id", "yhat", "timestamp_pred", "h"]:
        if col not in preds.columns:
            raise KeyError(f"Predicciones sin columna requerida: {col}")
    for col in ["station_id", "lat", "lon"]:
        if col not in stations.columns:
            raise KeyError(f"Estaciones sin columna requerida: {col}")

    # --- Merge
    tiles = preds.merge(stations, on="station_id", how="left")

    # --- Limpieza de valores
    tiles["yhat"] = pd.to_numeric(tiles["yhat"], errors="coerce").fillna(0)
    tiles["yhat"] = tiles["yhat"].clip(lower=0)

    # --- Niveles de disponibilidad
    def availability_level(x: float) -> str:
        if np.isnan(x):
            return "na"
        if x >= 10:
            return "high"
        elif x >= 3:
            return "medium"
        else:
            return "low"

    tiles["level"] = tiles["yhat"].apply(availability_level)
    tiles["color"] = tiles["level"].map(COLOR_RULES).fillna(COLOR_RULES["na"])

    # --- Columnas finales
    cols = [
        "station_id",
        "name" if "name" in stations.columns else None,
        "lat",
        "lon",
        "timestamp_pred",
        "h",
        "yhat",
        "level",
        "color",
    ]
    cols = [c for c in cols if c in tiles.columns]
    tiles = tiles[cols].copy()

    # --- Nombre de salida idempotente
    base, ext = os.path.splitext(out_path)
    out_final = f"{base}__tiles.parquet" if not ext else out_path

    # --- Guardar
    os.makedirs(os.path.dirname(out_final), exist_ok=True)
    tiles.to_parquet(out_final, index=False)

    # --- Logs finales
    print(f"✅ Tiles generados: {len(tiles)} filas, {tiles['station_id'].nunique()} estaciones.")
    print(f"📦 Guardados en: {out_final}")
    print(tiles.head(5))

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera tiles georreferenciados para dashboard Ecobici.")
    ap.add_argument("--pred", required=True, help="Archivo de predicciones (.parquet)")
    ap.add_argument("--stations", required=True, help="Archivo de información de estaciones (.parquet)")
    ap.add_argument("--out", required=True, help="Archivo de salida (.parquet)")
    args = ap.parse_args()

    try:
        main(pred_path=args.pred, stations_path=args.stations, out_path=args.out)
        print("[DONE] make_tiles OK")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] make_tiles: {e}")
        sys.exit(1)