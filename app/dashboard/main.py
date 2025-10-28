# app/dashboard/main.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from pydeck.data_utils import viewport_helpers as pdk_view
from datetime import datetime, timezone

# --- imports robustos ---
try:
    from app.dashboard.utils import load_predictions, load_stations
except Exception:
    from utils import load_predictions, load_stations  # fallback

st.set_page_config(page_title="Ecobici AutoML — Dashboard", layout="wide")
st.title("🚲 Ecobici AutoML — Predicción de Disponibilidad")

# ===================== helpers =====================
def _infer_capacity_cols(df: pd.DataFrame) -> pd.Series:
    for c in ["capacity", "num_docks_total", "dock_count", "capacidad", "_capacity"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index)

def _color_and_label(pct):
    if pd.isna(pct):  return [160,160,160], "gris"
    if pct >= 50:     return [0,180,0], "verde"
    if pct >= 11:     return [240,200,0], "amarillo"
    return [220,0,0], "rojo"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def _freshness_minutes(ts: pd.Timestamp) -> float | None:
    if ts is None or pd.isna(ts):
        return None
    # Normalizar a naive si viene con tz
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None) if hasattr(ts, "tz_convert") else ts.tz_localize(None)
    now = pd.Timestamp.now(tz=None)
    return float((now - ts).total_seconds() / 60.0)

# ===================== datos base =====================
preds = load_predictions()  # ideal: devuelve un DataFrame + attrs con path y asof
stations = load_stations()

# Fuente del parquet y asof (si utils lo adjuntan)
src = preds.attrs.get("__source_path__")
asof_attr = preds.attrs.get("__asof_ts__")

top_bar = st.empty()
if src:
    st.caption(f"Fuente de predicciones: `{src}`")

# ===================== sidebar =====================
st.sidebar.header("Controles")
autorefresh = st.sidebar.checkbox("Auto-refresh cada 60s", value=False)
if autorefresh:
    st.experimental_rerun  # referencia para que Streamlit habilite el rerun
    st.experimental_set_query_params(_=str(pd.Timestamp.now().value))
    st.experimental_rerun  # (Streamlit maneja el bucle internamente)

# Filtro de horizonte
has_h = ("h" in preds.columns) and preds["h"].notna().any()
if has_h:
    horizons = sorted([int(h) for h in preds["h"].dropna().unique()])
    default_idx = 1 if 3 in horizons else 0
    h_sel = st.sidebar.selectbox("Horizonte (horas)", horizons, index=default_idx)
    preds_h = preds[preds["h"] == h_sel].copy()
else:
    st.sidebar.caption("No se encontró columna 'h' en predicciones.")
    h_sel = None
    preds_h = preds.copy()

# Estación foco + radio
st.sidebar.subheader("Estación foco (para cercanas)")
if not {"station_id","lat","lon"}.issubset(stations.columns):
    st.sidebar.error("Faltan columnas en stations (se requieren station_id, lat, lon).")
opt_df = stations[[c for c in ["station_id","name","lat","lon"] if c in stations.columns]].dropna().copy()
lab_has_name = "name" in opt_df.columns
if lab_has_name:
    opt_df["label"] = opt_df.apply(lambda r: f"{int(r['station_id'])} — {r['name']}", axis=1)
else:
    opt_df["label"] = opt_df["station_id"].astype(str)
focus_label = st.sidebar.selectbox("Elegí estación foco", options=opt_df["label"].tolist())
radius_m = st.sidebar.slider("Radio de búsqueda (m)", 200, 2000, 1000, step=100)
near_box = st.sidebar.container()

# ===================== avisos de calidad =====================
if has_h:
    test = preds.groupby("h")["yhat"].mean(numeric_only=True)
    if test.nunique(dropna=True) <= 1 and len(test) > 1:
        st.info("ℹ️ Las predicciones parecen iguales entre horizontes (1/3/6/12). Probable falta de historia.")

# Frescura (usa timestamp_pred más reciente si no viene asof en attrs)
asof_guess = None
if "timestamp_pred" in preds.columns and preds["timestamp_pred"].notna().any():
    asof_guess = pd.to_datetime(preds["timestamp_pred"]).max()
asof_show = asof_attr or asof_guess
fresh_min = _freshness_minutes(asof_show) if asof_show is not None else None
if fresh_min is not None:
    st.caption(f"Última predicción: {pd.to_datetime(asof_show)} (hace {fresh_min:0.0f} min)")
    if fresh_min > 90:
        top_bar.warning("⚠️ Predicciones desactualizadas (>90 min). Verificá jobs de ingest/predict.")
else:
    top_bar.info("No se pudo inferir la frescura de las predicciones.")

st.divider()

# ===================== mapa =====================
st.subheader("Mapa de disponibilidad — vista interactiva")

if not preds_h.empty and not stations.empty and {"station_id","lat","lon"}.issubset(stations.columns):
    df_map = preds_h.merge(stations, on="station_id", how="left")

    # Si hay varias filas por estación/ts, quedarnos con la última por estación
    if "timestamp_pred" in df_map.columns:
        df_map["timestamp_pred"] = pd.to_datetime(df_map["timestamp_pred"], errors="coerce")
        df_map = df_map.sort_values(["station_id","timestamp_pred"]).groupby("station_id", as_index=False).tail(1)

    # Derivadas
    df_map["_capacity"]   = _infer_capacity_cols(df_map)
    df_map["yhat"]        = pd.to_numeric(df_map.get("yhat"), errors="coerce")
    df_map["_docks_pred"] = (df_map["_capacity"] - df_map["yhat"]).where(df_map["_capacity"].notna())
    df_map["_pct_bikes"]  = (df_map["yhat"] / df_map["_capacity"] * 100).where(df_map["_capacity"] > 0)
    df_map["_pct_docks"]  = (df_map["_docks_pred"] / df_map["_capacity"] * 100).where(df_map["_capacity"] > 0)

    for col in ["yhat", "_docks_pred", "_pct_bikes", "_pct_docks"]:
        if col in df_map.columns:
            df_map[col] = df_map[col].round(1)

    # Color y etiqueta
    df_map["color"], df_map["_semaforo_label"] = zip(*df_map["_pct_bikes"].apply(_color_and_label))

    # Vista inicial
    try:
        view_params = pdk_view.compute_view(df_map[["lon","lat"]])
        view_state = pdk.ViewState(latitude=view_params["latitude"], longitude=view_params["longitude"], zoom=view_params["zoom"])
    except Exception:
        center_lon = float(df_map["lon"].mean()); center_lat = float(df_map["lat"].mean())
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)

    # Tooltip
    h_display = f"{h_sel} hora{'s' if h_sel not in (None,1) else ''}" if has_h else "no especificado"
    tooltip = {
        "html": (
            "<b>Estación:</b> {name}<br/>"
            "<b>ID:</b> {station_id}<br/>"
            f"<b>Horizonte:</b> {h_display}<br/>"
            "<b>Semáforo:</b> {_semaforo_label}<br/>"
            "<b>Capacidad (C):</b> {_capacity}<br/>"
            "<b>Bicis predichas (B):</b> {yhat}<br/>"
            "<b>Docks libres predichos (D):</b> {_docks_pred}<br/>"
            "<b>% bicis:</b> {_pct_bikes}%<br/>"
            "<i>(Identidad: B + D = C)</i>"
        ),
        "style": {"backgroundColor": "rgba(0,0,0,0.75)", "color": "white"},
    }

    # Capas
    osm = pdk.Layer("TileLayer", data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", min_zoom=0, max_zoom=19, tile_size=256)
    layer = pdk.Layer("ScatterplotLayer", data=df_map, get_position=["lon", "lat"], get_radius=90, get_color="color", pickable=True)

    deck = pdk.Deck(layers=[osm, layer], initial_view_state=view_state, tooltip=tooltip, map_style=None)
    st.pydeck_chart(deck, use_container_width=True)

    # ===================== Top-5 cercanas (sidebar) =====================
    foco = opt_df[opt_df["label"] == focus_label].iloc[0]
    lat0, lon0, id0 = float(foco["lat"]), float(foco["lon"]), int(foco["station_id"])
    near = df_map.copy()
    near["distance_m"] = haversine_m(lat0, lon0, near["lat"].values, near["lon"].values)
    near = near[
        (near["_semaforo_label"].isin(["verde","amarillo"])) &
        (near["station_id"] != id0) &
        (near["distance_m"] <= radius_m)
    ].sort_values("distance_m").head(5)

    with near_box:
        st.markdown("**Top-5 cercanas con disponibilidad**")
        if near.empty:
            st.caption("No hay estaciones verdes/amarillas dentro del radio seleccionado.")
        else:
            cols = [c for c in ["station_id","name","yhat","_docks_pred","_pct_bikes","_semaforo_label","distance_m"] if c in near.columns]
            out = near[cols].rename(columns={
                "station_id":"ID","name":"Estación","yhat":"Bicis (B)","_docks_pred":"Docks libres (D)",
                "_pct_bikes":"% bicis","_semaforo_label":"Semáforo","distance_m":"Distancia (m)"
            })
            if "Distancia (m)" in out.columns:
                out["Distancia (m)"] = out["Distancia (m)"].round(0).astype("Int64")
            st.dataframe(out, use_container_width=True, hide_index=True)

else:
    st.warning("Para ver el mapa necesitás `predictions/*.parquet` y `data/curated/station_information.parquet`.")