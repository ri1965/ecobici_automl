# src/make_tiles.py
# ------------------------------------------------------------
# Genera tiles georreferenciados para dashboard Ecobici
# ------------------------------------------------------------
import os
import argparse
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Función principal
# ------------------------------------------------------------
def main(pred_path: str, stations_path: str, out_path: str):
    assert os.path.exists(pred_path), f"Archivo de predicciones no encontrado: {pred_path}"
    assert os.path.exists(stations_path), f"Archivo de estaciones no encontrado: {stations_path}"

    print(f"📂 Leyendo predicciones desde: {pred_path}")
    preds = pd.read_parquet(pred_path)
    print(f"📂 Leyendo estaciones desde: {stations_path}")
    stations = pd.read_parquet(stations_path)

    # --------------------------------------------------------
    # Unir coordenadas
    # --------------------------------------------------------
    if "station_id" not in preds.columns or "station_id" not in stations.columns:
        raise KeyError("Ambos archivos deben contener la columna 'station_id'.")

    tiles = preds.merge(stations, on="station_id", how="left")

    # --------------------------------------------------------
    # Derivar niveles de disponibilidad y colores
    # --------------------------------------------------------
    # Normalizar valores negativos o NaN
    tiles["yhat"] = tiles["yhat"].clip(lower=0).fillna(0)

    # Regla simple de color (puede adaptarse según umbrales de negocio)
    def availability_level(x):
        if x >= 10:
            return "high"
        elif x >= 3:
            return "medium"
        else:
            return "low"

    def availability_color(level):
        return {"high": "#2ECC71", "medium": "#F1C40F", "low": "#E74C3C"}.get(level, "#BDC3C7")

    tiles["level"] = tiles["yhat"].apply(availability_level)
    tiles["color"] = tiles["level"].apply(availability_color)

    # --------------------------------------------------------
    # Seleccionar columnas clave
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Guardar salida
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tiles.to_parquet(out_path, index=False)
    print(f"✅ Tiles guardados en: {out_path}")
    print(tiles.head(5))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera tiles georreferenciados para dashboard Ecobici.")
    ap.add_argument("--pred", required=True, help="Archivo de predicciones (.parquet)")
    ap.add_argument("--stations", required=True, help="Archivo de información de estaciones (.parquet)")
    ap.add_argument("--out", required=True, help="Archivo de salida (.parquet o .geojson)")
    args = ap.parse_args()

    main(pred_path=args.pred, stations_path=args.stations, out_path=args.out)