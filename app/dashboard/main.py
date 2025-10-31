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

# --- Ajuste CSS para reducir padding del sidebar y evitar scroll ---
st.markdown("""
<style>
section[data-testid="stSidebar"] div.stSidebarContent {
    padding-top: 0rem !important;       /* elimina casi todo el margen superior */
    padding-bottom: 0.3rem !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    margin-top: 0.4rem !important;      /* títulos más compactos */
    margin-bottom: 0.3rem !important;
}
div[data-testid="stSidebarNav"] { 
    display: none !important;           /* oculta la pequeña franja de título vacía */
}
</style>
""", unsafe_allow_html=True)

# CSS: reducir tamaño de métricas solo en la mini ficha del sidebar
st.markdown("""
<style>
.mini-ficha div[data-testid="stMetric"] label { 
    font-size: 0.80rem !important;   /* etiqueta más chica */
}
.mini-ficha div[data-testid="stMetricValue"] { 
    font-size: 1.20rem !important;   /* valor ~35-40% más chico */
    line-height: 1.0 !important;
}
</style>
""", unsafe_allow_html=True)

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

# Hex por semáforo para Folium
def _hex_from_label(label: str) -> str:
    return {
        "verde":   "#2ecc71",
        "amarillo":"#f1c40f",
        "rojo":    "#e74c3c",
        "gris":    "#95a5a6",
    }.get(label, "#95a5a6")

# Casteo seguro a int (NaN -> 0)
def _int0(x) -> int:
    try:
        return int(np.nan_to_num(float(x), nan=0.0))
    except Exception:
        return 0

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

    # 1) Renombrado: Franja -> Horario de Predicción
    slot_sel = st.sidebar.selectbox("Horario de Predicción", slots_sorted, index=0)
    preds_h = preds[preds["slot_label"] == slot_sel].copy()
    h_sel = None
    mode_label = f"Horario: {slot_sel}"

    # 2) Radio de Búsqueda DEBAJO del horario  + contenedor CSV justo debajo
    radius_m = st.sidebar.slider("Radio de Búsqueda (m)", 200, 2000, 1000, step=100)
    csv_box = st.sidebar.container()  # botón CSV va aquí, debajo del slider

else:
    has_h = ("h" in preds.columns) and preds["h"].notna().any()
    if has_h:
        horizons = sorted([int(h) for h in preds["h"].dropna().unique()])
        h_sel = st.sidebar.selectbox("Horizonte (horas)", horizons, index=0)
        preds_h = preds[preds["h"] == h_sel].copy()
        mode_label = f"Horizonte: {h_sel} h"

        radius_m = st.sidebar.slider("Radio de Búsqueda (m)", 200, 2000, 1000, step=100)
        csv_box = st.sidebar.container()
    else:
        st.sidebar.caption("No se encontró columna 'slot_label' ni 'h' en predicciones.")
        h_sel = None
        preds_h = preds.copy()
        mode_label = "Sin modo"
        radius_m = st.sidebar.slider("Radio de Búsqueda (m)", 200, 2000, 1000, step=100)
        csv_box = st.sidebar.container()

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

# Contenedor Top-5 en sidebar (lo usamos para la ficha y el top-5)
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

    # Color semáforo
    df_map["color"], df_map["_semaforo_label"] = zip(*df_map["_pct_bikes"].apply(_color_and_label))

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
                f"<br><b>ID:</b> {_int0(r.get('station_id'))}"
                f"<br><b>{mode_label}</b>"
                f"<br><b>Semáforo:</b> {r.get('_semaforo_label','')}"
                f"<br><b>Capacidad (C):</b> {_int0(r.get('_capacity'))}"
                f"<br><b>Bicis (B):</b> {_int0(r.get('yhat'))}"
                f"<br><b>Docks (D):</b> {_int0(r.get('_docks_pred'))}"
                f"<br><b>% bicis:</b> {round(float(r.get('_pct_bikes') or 0),1)}%"
            ),
        ).add_to(m)

    # halo foco
    folium.CircleMarker(
        location=[lat0, lon0],
        radius=10, color="#00FFFF", fill=False, weight=3
    ).add_to(m)

    # halos verdes para cercanas (respecto al foco)
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
        key=f"map_clickable",
        returned_objects=["last_clicked", "center", "zoom"]
    )

    # Guardar siempre la vista actual (para no perder zoom/centro)
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

    # === Botón CSV (render EN la sidebar debajo del slider) ===
    with csv_box:
        st.markdown("**Exportar CSV – Top-5 vecinas de estaciones rojas**")

        # 1) Solo usamos el df del slot/horario actual, como el Top-5
        df_base = df_map.copy()

        # 2) Estaciones rojas (origen para el CSV)
        rojas = df_base[df_base["_semaforo_label"] == "rojo"].copy()

        if rojas.empty:
            st.caption("No hay estaciones rojas dentro del conjunto actual.")
        else:
            export_rows = []
            for _, r0 in rojas.iterrows():
                lat_r, lon_r = float(r0["lat"]), float(r0["lon"])
                id_r, name_r = int(r0["station_id"]), r0["name"]

                # 3) Vecinas: **mismo filtro que el Top-5** → solo verde/amarillo, distinto id y dentro del radio
                df_tmp = df_base.copy()
                df_tmp["distance_m"] = haversine_m(lat_r, lon_r, df_tmp["lat"].values, df_tmp["lon"].values)
                df_tmp = df_tmp[
                    (df_tmp["_semaforo_label"].isin(["verde", "amarillo"])) &
                    (df_tmp["station_id"] != id_r) &
                    (df_tmp["distance_m"] <= radius_m)
                ].sort_values("distance_m").head(5)

                # 4) Armado de filas (mismos redondeos que el Top-5)
                for _, n in df_tmp.iterrows():
                    export_rows.append({
                        "estacion_roja_id": id_r,
                        "estacion_roja_nombre": name_r,
                        "vecina_id": int(n["station_id"]),
                        "vecina_nombre": n["name"],
                        "distancia_m": int(round(float(n["distance_m"]), 0)),
                        "bicis_pred": int(round(float(n["yhat"])) if pd.notna(n["yhat"]) else 0),
                        "docks_pred": int(round(float(n["_docks_pred"])) if pd.notna(n["_docks_pred"]) else 0),
                        "pct_bicis": round(float(n["_pct_bikes"]) if pd.notna(n["_pct_bikes"]) else 0.0, 1),
                        "semaforo": n["_semaforo_label"],
                        "horario_prediccion": slot_sel if has_slots else None,
                        "horizonte_h": h_sel if (not has_slots) else None,
                        "radio_m": radius_m,
                    })

            if export_rows:
                df_export = pd.DataFrame(export_rows)
                fname = (
                    f"top5_vecinas_rojas_{str(slot_sel).replace(':','')}_{radius_m}m.csv"
                    if has_slots else
                    f"top5_vecinas_rojas_h{h_sel}_{radius_m}m.csv"
                )
                st.download_button(
                    label="📥 Descargar CSV",
                    data=df_export.to_csv(index=False).encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                )
    # -------- Tarjeta de datos de la Estación Foco (en el cuerpo) --------
    focus_row_full = df_map[df_map["station_id"] == id0].copy()
    if not focus_row_full.empty:
        row = focus_row_full.iloc[0]
        st.markdown("### 📍 Estación Foco — Detalle")
        with st.container(border=True):
            st.markdown(f"**{row['name']}**")
            st.caption(f"Ubicación: {row['lat']:.5f}, {row['lon']:.5f}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Capacidad", f"{_int0(row['_capacity'])}")
            c2.metric("Bicis (pred.)", f"{_int0(row['yhat'])}")
            c3.metric("Docks libres (pred.)", f"{_int0(row['_docks_pred'])}")

    # ----------------- FICHA COMPACTA EN SIDEBAR + TOP-5 -----------------
    # 1) Ficha compacta de estación foco (en sidebar, arriba del Top-5)
    with near_box:
        st.markdown("**📍 Detalle Estación Foco**")
        focus_row_sidebar = df_map[df_map["station_id"] == id0]
        if not focus_row_sidebar.empty:
            r = focus_row_sidebar.iloc[0]

            # Solo mostramos hora + semáforo (sin nombre ni ID)
            st.caption(f"{mode_label} — {str(r['_semaforo_label']).capitalize()}")

            # Envolvemos con una clase para aplicar el CSS de tamaños
            st.markdown('<div class="mini-ficha">', unsafe_allow_html=True)
            c1s, c2s, c3s = st.columns(3)
            c1s.metric("Cap.", f"{_int0(r['_capacity'])}")
            c2s.metric("Bicis", f"{_int0(r['yhat'])}")
            c3s.metric("Docks libres", f"{_int0(r['_docks_pred'])}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Barra de % bicis
            pct = float(r["_pct_bikes"] or 0.0)
            st.progress(max(0.0, min(1.0, pct/100.0)), text=f"{pct:.1f}% bicis")

    # 2) Top-5 cercanas (en sidebar, debajo de la ficha)
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
            # Seleccionar y renombrar columnas
            cols = ["name", "yhat", "_docks_pred", "_pct_bikes", "_semaforo_label", "distance_m"]
            out = near[cols].rename(columns={
                "name": "Estación",
                "yhat": "Bicis",
                "_docks_pred": "Docks libres",
                "_pct_bikes": "% bicis",
                "_semaforo_label": "Semáforo",
            })

            # Redondear y formatear
            out["Bicis"] = out["Bicis"].round(0).astype("Int64")
            out["Docks libres"] = out["Docks libres"].round(0).astype("Int64")
            out["% bicis"] = out["% bicis"].round(1)
            out["Distancia (m)"] = out.pop("distance_m").round(0).astype("Int64")

            # === estilo compacto ===
            st.markdown(
                """
                <style>
                .small-table td, .small-table th {
                    font-size: 13px !important;
                    padding: 4px 6px !important;
                    text-align: center !important;
                    white-space: nowrap;
                }
                .small-table table {
                    width: 100% !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.dataframe(
                out,
                use_container_width=True,
                hide_index=True
            )

else:
    st.warning(
        "Para ver el mapa necesitás predicciones y estaciones.\n\n"
        "• Modo franjas: `predictions/slots/*/predictions.parquet` "
        "(columnas: station_id, t_slot, slot_label, yhat)\n"
        "• Modo horizonte: `predictions/*.parquet` con columna `h`\n"
        "• Estaciones: `data/curated/station_information.parquet`"
    )