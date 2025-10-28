from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from pydeck.data_utils import viewport_helpers as pdk_view

# --- imports robustos ---
try:
    from app.dashboard.utils import (
        load_predictions, load_stations
    )
except Exception:
    from utils import (
        load_predictions, load_stations
    )

st.set_page_config(page_title="Ecobici AutoML — Dashboard", layout="wide")
st.title("🚲 Ecobici AutoML — Predicción de Disponibilidad")

# ----------------- helpers -----------------
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

# ----------------- datos base -----------------
preds = load_predictions()
stations = load_stations()

# Mostrar la fuente del parquet que se está usando
src = preds.attrs.get("__source_path__")
if src:
    st.caption(f"Fuente de predicciones: `{src}`")

# ----------------- sidebar -----------------
st.sidebar.header("Filtros")

# Horizonte (si existe h)
has_h = ("h" in preds.columns) and preds["h"].notna().any()
if has_h:
    horizons = sorted([int(h) for h in preds["h"].dropna().unique()])
    h_sel = st.sidebar.selectbox("Horizonte (horas)", horizons, index=0)
    preds_h = preds[preds["h"] == h_sel].copy()
else:
    st.sidebar.caption("No se encontró columna 'h' en predicciones.")
    h_sel = None
    preds_h = preds.copy()

# Estación foco + radio (para cercanas)
st.sidebar.subheader("Estación foco (para cercanas)")
opt_df = stations[["station_id","name","lat","lon"]].dropna().copy()
opt_df["label"] = opt_df.apply(lambda r: f"{int(r['station_id'])} — {r['name']}", axis=1)
focus_label = st.sidebar.selectbox("Elegí estación foco", options=opt_df["label"].tolist())
radius_m = st.sidebar.slider("Radio de búsqueda (m)", 200, 2000, 1000, step=100)

# Contenedor donde se renderiza el Top-5
near_box = st.sidebar.container()

# ----------------- aviso si los horizontes no cambian -----------------
if has_h:
    test = preds.groupby("h")["yhat"].mean(numeric_only=True)
    if test.nunique(dropna=True) <= 1 and len(test) > 1:
        st.info("ℹ️ Las predicciones parecen iguales entre horizontes (1/3/6/12). Revisá cómo se generan en tu pipeline.")

st.divider()

# ----------------- mapa -----------------
st.subheader("Mapa de disponibilidad — vista interactiva")

if not preds_h.empty and not stations.empty and {"station_id","lat","lon"}.issubset(stations.columns):
    # Merge predicciones + estaciones
    df_map = preds_h.merge(stations, on="station_id", how="left")

    # Si hay timestamps, quedarnos con la última por estación
    if "ts" in df_map.columns:
        df_map = df_map.sort_values("ts").groupby("station_id", as_index=False).tail(1)

    # Derivadas
    df_map["_capacity"]   = _infer_capacity_cols(df_map)
    df_map["yhat"]        = pd.to_numeric(df_map.get("yhat"), errors="coerce")
    df_map["_docks_pred"] = (df_map["_capacity"] - df_map["yhat"]).where(df_map["_capacity"].notna())
    df_map["_pct_bikes"]  = (df_map["yhat"] / df_map["_capacity"] * 100).where(df_map["_capacity"] > 0)
    df_map["_pct_docks"]  = (df_map["_docks_pred"] / df_map["_capacity"] * 100).where(df_map["_capacity"] > 0)

    for col in ["yhat", "_docks_pred", "_pct_bikes", "_pct_docks"]:
        if col in df_map.columns:
            df_map[col] = df_map[col].round(1)

    # Color y etiqueta de semáforo
    df_map["color"], df_map["_semaforo_label"] = zip(*df_map["_pct_bikes"].apply(_color_and_label))

    # Vista inicial
    try:
        view_params = pdk_view.compute_view(df_map[["lon","lat"]])
        view_state = pdk.ViewState(latitude=view_params["latitude"], longitude=view_params["longitude"], zoom=view_params["zoom"])
    except Exception:
        center_lon = float(df_map["lon"].mean()); center_lat = float(df_map["lat"].mean())
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)

    # Tooltip
    h_display = f"{h_sel} hora{'s' if h_sel != 1 else ''}" if has_h else "no especificado"
    tooltip = {
        "html": (
            "<b>Estación:</b> {name}<br/>"
            "<b>ID:</b> {station_id}<br/>"
            f"<b>Horizonte:</b> {h_display}<br/>"
            "<b>Semáforo:</b> {_semaforo_label}<br/>"
            "<b>Capacidad (C):</b> {_capacity}<br/>"
            "<b>Bicis predichas (B):</b> {yhat}<br/>"
            "<b>Docks libres predichos (D):</b> {_docks_pred}<br/>"
            "<b>% disp. de bicis:</b> {_pct_bikes}%<br/>"
            "<i>(Identidad: B + D = C)</i>"
        ),
        "style": {"backgroundColor": "rgba(0,0,0,0.75)", "color": "white"},
    }

    # === NUEVO: capas de foco y anillos de sugeridas ============================
    foco = opt_df[opt_df["label"] == focus_label].iloc[0]
    focus_df = pd.DataFrame([{
        "station_id": int(foco["station_id"]),
        "name": foco["name"],
        "lat": float(foco["lat"]),
        "lon": float(foco["lon"]),
    }])

    lat0, lon0, id0 = float(foco["lat"]), float(foco["lon"]), int(foco["station_id"])
    near_for_layer = df_map.copy()
    near_for_layer["distance_m"] = haversine_m(lat0, lon0, near_for_layer["lat"].values, near_for_layer["lon"].values)
    near_for_layer = near_for_layer[
        (near_for_layer["_semaforo_label"].isin(["verde", "amarillo"])) &
        (near_for_layer["station_id"] != id0) &
        (near_for_layer["distance_m"] <= radius_m)
    ].copy()

    focus_layer = pdk.Layer(
        "ScatterplotLayer",
        data=focus_df,
        get_position=["lon", "lat"],
        get_radius=140,
        get_fill_color=[30, 144, 255, 200],   # azul (DodgerBlue)
        stroked=True,
        get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=2,
        pickable=False
    )

    ring_layer = pdk.Layer(
        "ScatterplotLayer",
        data=near_for_layer,
        get_position=["lon", "lat"],
        get_radius=220,
        filled=False,
        stroked=True,
        get_line_color=[0, 200, 0, 120],  # verde semitransparente
        line_width_min_pixels=8,
        pickable=False
    )
    # ============================================================================

    # Capas base
    osm = pdk.Layer(
        "TileLayer",
        data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0, max_zoom=19, tile_size=256
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_radius=90,
        get_color="color",
        pickable=True
    )

    # Capas combinadas
    deck = pdk.Deck(
        layers=[osm, ring_layer, focus_layer, layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None
    )
    st.pydeck_chart(deck, use_container_width=True)

    # ----------------- Top-5 cercanas EN EL SIDEBAR -----------------
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
            cols = ["station_id","name","yhat","_docks_pred","_pct_bikes","_semaforo_label","distance_m"]
            out = near[cols].rename(columns={
                "station_id":"ID","name":"Estación","yhat":"Bicis (B)","_docks_pred":"Docks libres (D)",
                "_pct_bikes":"% bicis","_semaforo_label":"Semáforo","distance_m":"Distancia (m)"
            })
            out["Distancia (m)"] = out["Distancia (m)"].round(0).astype("Int64")
            st.dataframe(out, use_container_width=True, hide_index=True)

else:
    st.warning("Para ver el mapa necesitás `predictions/*.parquet` y `data/curated/station_information.parquet`.")