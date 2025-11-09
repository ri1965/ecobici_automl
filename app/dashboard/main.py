from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from pydeck.data_utils import viewport_helpers as pdk_view

# Mapa con captura de click
import folium
from streamlit_folium import st_folium

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

# ✅ Semáforo: rojo=0 bici, amarillo <50%, verde >=50%
def _color_and_label(pct, bikes):
    if pd.isna(pct):             return [160,160,160], "gris"
    try:
        b = float(bikes) if bikes is not None else np.nan
    except Exception:
        b = np.nan
    # Rojo: 0 bicis (o negativo por predicción)
    if pd.notna(b) and b <= 0:
        return [220,0,0], "rojo"
    # Verde / Amarillo por % de bicis
    if pct >= 50:
        return [0,180,0], "verde"
    if pct > 0:
        return [240,200,0], "amarillo"
    # Si % es 0 y no pudimos evaluar bikes, consideramos rojo
    return [220,0,0], "rojo"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

# Hex por semáforo para Folium
def _hex_from_label(label: str) -> str:
    return {
        "verde":   "#2ecc71",
        "amarillo":"#f1c40f",
        "rojo":    "#e74c3c",
        "gris":    "#95a5a6",
    }.get(label, "#95a5a6")

def _sanitize_for_fname(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in ("-", "_"))

# ----------------- datos base -----------------
preds = load_predictions()
stations = load_stations()

# Mostrar la fuente usada
src = getattr(preds, "attrs", {}).get("__source_path__")
if src:
    st.caption(f"Fuente de predicciones: `{src}`")

# Última actualización si existe
if "generated_at" in preds.columns:
    try:
        last_gen = pd.to_datetime(preds["generated_at"]).max()
        st.caption(f"Última actualización de predicciones: {last_gen}")
    except Exception:
        pass

# ----------------- sidebar -----------------
st.sidebar.header("Filtros")

# --- Slots (nuevo) o horizontes (compat) ---
has_slots = ("slot_label" in preds.columns) and preds["slot_label"].notna().any()
if has_slots:
    slots_avail = preds["slot_label"].dropna().unique().tolist()
    order_pref = ["09:00", "13:00", "16:00", "20:00"]
    slots_sorted = [s for s in order_pref if s in slots_avail] + [s for s in slots_avail if s not in order_pref]
    slot_sel = st.sidebar.selectbox("Franja horaria", slots_sorted, index=0)
    preds_h = preds[preds["slot_label"] == slot_sel].copy()
    h_sel = None
    mode_label = f"Franja: {slot_sel}"
else:
    has_h = ("h" in preds.columns) and preds["h"].notna().any()
    if has_h:
        horizons = sorted([int(h) for h in preds["h"].dropna().unique()])
        h_sel = st.sidebar.selectbox("Horizonte (horas)", horizons, index=0)
        preds_h = preds[preds["h"] == h_sel].copy()
        mode_label = f"Horizonte: {h_sel} h"
    else:
        st.sidebar.caption("No se encontró columna 'slot_label' ni 'h' en predicciones.")
        h_sel = None
        preds_h = preds.copy()
        mode_label = "Sin modo"

# Estación foco + radio (para cercanas)
st.sidebar.subheader("Estación foco (para cercanas)")
opt_df = stations[["station_id", "name", "lat", "lon"]].dropna().copy()
options = opt_df.to_dict("records")

# ====== Estado inicial (foco + vista del mapa) ======
if "station_focus_id" not in st.session_state:
    st.session_state["station_focus_id"] = int(options[0]["station_id"]) if options else None

# (opcional) bloquear vista del mapa entre acciones
lock_view = st.sidebar.checkbox("Mantener vista del mapa", value=True)

# init view defaults una sola vez
if "map_center" not in st.session_state:
    _ini = next((o for o in options if int(o["station_id"]) == int(st.session_state["station_focus_id"])), options[0])
    st.session_state["map_center"] = [float(_ini["lat"]), float(_ini["lon"])]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 13

# --- SYNC PRE-WIDGET ---
# Si venimos de un click, pedimos que el selectbox adopte el foco recién elegido
if st.session_state.get("__sync_focus__") is True:
    try:
        current = next(
            o for o in options
            if int(o["station_id"]) == int(st.session_state["station_focus_id"])
        )
        st.session_state["focus_row_sel"] = current
    except StopIteration:
        pass
    finally:
        st.session_state.pop("__sync_focus__", None)

# Si no hay valor cargado aún (primera vez), setear desde el foco
if "focus_row_sel" not in st.session_state and "station_focus_id" in st.session_state:
    try:
        current = next(
            o for o in options
            if int(o["station_id"]) == int(st.session_state["station_focus_id"])
        )
        st.session_state["focus_row_sel"] = current
    except StopIteration:
        pass

# Callback: cuando el usuario cambia el select, actualizar foco (y opcionalmente centrar)
def _on_focus_change():
    sel = st.session_state.get("focus_row_sel")
    if sel:
        st.session_state["station_focus_id"] = int(sel["station_id"])
        if not lock_view:
            st.session_state["map_center"] = [float(sel["lat"]), float(sel["lon"])]
            st.session_state["map_zoom"] = 14

focus_row = st.sidebar.selectbox(
    "Elegí estación foco",
    options=options,
    format_func=lambda r: r["name"],
    key="focus_row_sel",
    on_change=_on_focus_change
)

radius_m = st.sidebar.slider("Radio de búsqueda (m)", 200, 2000, 1000, step=100)

# Contenedor Top-5 en sidebar
near_box = st.sidebar.container()

# Aviso si los horizontes no cambian (solo modo legacy)
if ("h" in preds.columns) and preds["h"].notna().any():
    test = preds.groupby("h")["yhat"].mean(numeric_only=True)
    if test.nunique(dropna=True) <= 1 and len(test) > 1:
        st.info("ℹ️ Las predicciones parecen iguales entre horizontes (1/3/6/12). Revisá el pipeline.")

st.divider()

# ----------------- mapa -----------------
st.subheader("Mapa de disponibilidad — vista interactiva")

# CSS para asegurar ancho completo del contenedor
st.markdown("""
<style>
.leaflet-container { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

if not preds_h.empty and not stations.empty and {"station_id","lat","lon"}.issubset(stations.columns):
    # Merge predicciones + estaciones
    df_map = preds_h.merge(stations, on="station_id", how="left")

    # Elegir columna temporal (slots vs horizontes)
    ts_col = "t_slot" if "t_slot" in df_map.columns else ("ts" if "ts" in df_map.columns else None)

    # Quedarse con la última por estación
    if ts_col is not None:
        df_map = df_map.sort_values(ts_col).groupby("station_id", as_index=False).tail(1)

    # Derivadas
    df_map["_capacity"]   = _infer_capacity_cols(df_map)
    df_map["yhat"]        = pd.to_numeric(df_map.get("yhat"), errors="coerce")
    df_map["_docks_pred"] = (df_map["_capacity"] - df_map["yhat"]).where(df_map["_capacity"].notna())
    df_map["_pct_bikes"]  = (df_map["yhat"] / df_map["_capacity"] * 100).where(df_map["_capacity"] > 0)
    df_map["_pct_docks"]  = (df_map["_docks_pred"] / df_map["_capacity"] * 100).where(df_map["_capacity"] > 0)

    for col in ["yhat", "_docks_pred", "_pct_bikes", "_pct_docks"]:
        if col in df_map.columns:
            df_map[col] = df_map[col].round(1)

    # ✅ Color semáforo (usa % y cantidad de bicis)
    df_map["color"], df_map["_semaforo_label"] = zip(*df_map.apply(
        lambda r: _color_and_label(r.get("_pct_bikes"), r.get("yhat")),
        axis=1
    ))

    # === Foco vigente a partir del session_state ===
    try:
        foco = next(
            o for o in options
            if int(o["station_id"]) == int(st.session_state["station_focus_id"])
        )
    except StopIteration:
        foco = options[0]

    lat0, lon0, id0 = float(foco["lat"]), float(foco["lon"]), int(foco["station_id"])

    # ---- Mapa FOLIUM (usa vista guardada y fallback de tiles) ----
    center_lat, center_lon = st.session_state["map_center"]
    zoom_start = st.session_state["map_zoom"]
    try:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="CartoDB dark_matter")
        _ = m.to_dict()
    except Exception:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")

    # estaciones (círculos de color)
    for _, r in df_map.iterrows():
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=5,
            weight=1,
            color=_hex_from_label(r["_semaforo_label"]),
            fill=True,
            fill_opacity=0.9,
            tooltip=(
                f"<b>Estación:</b> {r.get('name','')}"
                f"<br><b>ID:</b> {int(r.get('station_id',0))}"
                f"<br><b>{mode_label}</b>"
                f"<br><b>Semáforo:</b> {r.get('_semaforo_label','')}"
                f"<br><b>Capacidad (C):</b> {int(r.get('_capacity') or 0)}"
                f"<br><b>Bicis (B):</b> {int((r.get('yhat') or 0))}"
                f"<br><b>Docks (D):</b> {int((r.get('_docks_pred') or 0))}"
                f"<br><b>% bicis:</b> {round((r.get('_pct_bikes') or 0),1)}%"
            ),
        ).add_to(m)

    # halo foco
    folium.CircleMarker(
        location=[lat0, lon0],
        radius=10, color="#00FFFF", fill=False, weight=3
    ).add_to(m)

    # halos verdes/amarillos para cercanas (respecto al foco)
    near_for_layer = df_map.copy()
    near_for_layer["distance_m"] = haversine_m(lat0, lon0, near_for_layer["lat"].values, near_for_layer["lon"].values)
    near_for_layer = near_for_layer[
        (near_for_layer["_semaforo_label"].isin(["verde", "amarillo"])) &
        (near_for_layer["station_id"] != id0) &
        (near_for_layer["distance_m"] <= radius_m)
    ].copy()

    for _, r in near_for_layer.iterrows():
        folium.Circle(
            location=[float(r["lat"]), float(r["lon"])],
            radius=220,
            color="#00aa00",
            fill=False,
            weight=2,
            opacity=0.7
        ).add_to(m)

    # --- RENDER Folium con ancho del contenedor y key dinámico ---
    map_state = st_folium(
        m,
        height=620,
        use_container_width=True,
        key=f"map_clickable_{st.session_state.get('station_focus_id')}_{st.session_state.get('map_zoom', 13)}",
        returned_objects=["last_clicked", "center", "zoom"]
    )

    # Guardar vista actual
    try:
        if "center" in map_state and map_state["center"]:
            st.session_state["map_center"] = [map_state["center"]["lat"], map_state["center"]["lng"]]
        if "zoom" in map_state and map_state["zoom"] is not None:
            st.session_state["map_zoom"] = map_state["zoom"]
    except Exception:
        pass

    # ---- CLICK: actualizar estación foco y rerun ----
    if map_state and map_state.get("last_clicked"):
        _lat = float(map_state["last_clicked"]["lat"])
        _lon = float(map_state["last_clicked"]["lng"])
        d = haversine_m(_lat, _lon, df_map["lat"].values, df_map["lon"].values)
        idx = int(np.argmin(d))
        new_focus = df_map.iloc[idx]
        st.session_state["station_focus_id"] = int(new_focus["station_id"])
        st.session_state["__sync_focus__"] = True
        if not lock_view:
            st.session_state["map_center"] = [float(new_focus["lat"]), float(new_focus["lon"])]
            st.session_state["map_zoom"] = 14
        st.rerun()

    # ----------------- 📍 Detalle Estación Foco (SIDEBAR) -----------------
    if "station_focus_id" in st.session_state and not stations.empty:
        focus_info = stations.loc[stations["station_id"] == st.session_state["station_focus_id"]]
        if not focus_info.empty:
            st.sidebar.markdown("### 📍 Detalle Estación Foco")
            row_focus = df_map[df_map["station_id"] == st.session_state["station_focus_id"]]
            if not row_focus.empty:
                foco_data = row_focus.iloc[0]
                st.sidebar.caption(f"{mode_label} — {foco_data.get('_semaforo_label','')}")
                c1, c2, c3 = st.sidebar.columns(3)
                c1.metric("Cap.", int(foco_data.get("_capacity", 0)))
                c2.metric("Bicis", int(foco_data.get("yhat", 0)))
                c3.metric("Docks libres", int(foco_data.get("_docks_pred", 0)))
                pct = foco_data.get("_pct_bikes", 0)
                try:
                    st.sidebar.progress(min(max(float(pct)/100, 0), 1), text=f"{pct:.1f}% bicis")
                except Exception:
                    pass
            st.sidebar.divider()

    # ----------------- Top-5 cercanas EN EL SIDEBAR -----------------
    near = df_map.copy()
    near["distance_m"] = haversine_m(lat0, lon0, near["lat"].values, near["lon"].values)
    # Filtro: semáforo con disponibilidad, dentro del radio y excluyendo foco
    near = near[
        (near["_semaforo_label"].isin(["verde", "amarillo"])) &
        (near["station_id"] != id0) &
        (near["distance_m"] <= radius_m)
    ].copy()

    # ✅ Orden: Bicis (desc) y, a igualdad, Distancia (asc)
    near = near.sort_values(by=["yhat", "distance_m"], ascending=[False, True]).head(5)

    with near_box:
        st.markdown("**Top-5 cercanas con disponibilidad** _(orden: Bicis ↓, Distancia ↑)_")
        if near.empty:
            st.caption("No hay estaciones verdes/amarillas dentro del radio seleccionado.")
        else:
            cols = ["name", "yhat", "_docks_pred", "_pct_bikes", "_semaforo_label", "distance_m"]
            out = near[cols].rename(columns={
                "name": "Estación",
                "yhat": "Bicis",
                "_docks_pred": "Docks libres",
                "_pct_bikes": "% bicis",
                "_semaforo_label": "Semáforo",
            })

            # Redondeos para presentación
            out["Bicis"] = out["Bicis"].round(0).astype("Int64")
            out["Docks libres"] = out["Docks libres"].round(0).astype("Int64")
            out["% bicis"] = out["% bicis"].round(1)
            out["Distancia (m)"] = out.pop("distance_m").round(0).astype("Int64")

            st.dataframe(out, use_container_width=True, hide_index=True)

            # --- Descarga CSV (mismo orden mostrado) ---
            suffix = _sanitize_for_fname(slot_sel.replace(":", "")) if has_slots else f"h{h_sel}"
            csv_data = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name=f"top5_{id0}_{suffix}.csv",
                mime="text/csv"
            )

else:
    st.warning(
        "Para ver el mapa necesitás predicciones y estaciones.\n\n"
        "• Modo franjas: `predictions/slots/*/predictions.parquet` "
        "(columnas: station_id, t_slot, slot_label, yhat)\n"
        "• Modo horizonte: `predictions/*.parquet` con columna `h`\n"
        "• Estaciones: `data/curated/station_information.parquet`"
    )