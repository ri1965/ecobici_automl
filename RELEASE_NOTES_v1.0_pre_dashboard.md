# 🚲 Release v1.0_pre_dashboard — Ecobici AutoML

## 📝 Descripción
Primera versión consolidada del flujo completo **hasta la etapa de predicciones batch**, previa al desarrollo del dashboard visual.  
Esta release integra todos los componentes del pipeline de modelado y despliegue automático de predicciones horarias de disponibilidad de bicicletas del sistema Ecobici (GCBA).

Incluye:
- Estructura de carpetas normalizada (`data/`, `src/`, `notebooks/`, `models/`, `reports/`, `docs/`).
- Entorno reproducible `environment.yml` y verificación con `make verify-env`.
- Pipelines de modelado AutoML (PyCaret, H2O, FLAML).
- Comparativa de métricas y selección del mejor modelo (`03D_Comparativa`).
- Generación de predicciones batch (`04C_Predict_Batch`).
- Árbol de repositorio y documentación de flujo (`docs/dev/repo_tree.txt`, `Explicacion_Flujo.docx`).
- Integración con MLflow local.

---

## ✅ Checklist de esta versión
- [x] Entorno `ecobici_automl` validado y estable.  
- [x] Ingesta, limpieza y curación de datos en Parquet (`data/curated/`).  
- [x] Modelado AutoML completo (PyCaret / H2O / FLAML).  
- [x] Comparativa de resultados y exportación de métricas.  
- [x] Script batch de predicciones funcionando (`src/predict_batch.py`).  
- [x] Outputs generados en `predictions/` y `reports/`.  
- [x] Estructura reproducible con `Makefile` (`make repo-tree`, `make verify-env`).  

---

## 📊 Próximos pasos — Etapa Dashboard
1. **Definir fuente visual**: Streamlit (versión inicial) → Tableau / Power BI (fase 2).  
2. **Consumir datos desde `predictions/*.parquet` y `station_information.parquet`.**  
3. **Implementar mapa de calor** con semáforo de disponibilidad (bicis/docks).  
4. **Agregar selector de horizonte (1h, 3h, 6h, 12h)** según resultados de `backtest_metrics_by_horizon.csv`.  
5. **Publicar visualización interactiva** en `app/dashboard/` (Streamlit).  

---

## 📁 Archivos clave para visualización
Estos archivos garantizan que ya está disponible la base necesaria para construir un dashboard (en **Streamlit**, **Tableau** o **Power BI**):

| Archivo | Descripción |
|----------|--------------|
| `predictions/*.parquet` | Predicciones listas para visualización. |
| `data/curated/station_information.parquet` | Metadatos espaciales (lat, lon, nombre). |
| `reports/backtest_metrics_by_horizon.csv` | Errores por horizonte. |
| `reports/holdout_month_metrics.csv` | Validación externa. |
| `reports/error_by_hour_and_horizon.csv` | Análisis temporal detallado. |

✅ Con esto **ya se dispone del dataset final y estructurado para visualización**, habilitando la creación de dashboards en Streamlit o herramientas externas (Tableau / Power BI).

---

🗓 **Release:** `v1.0_pre_dashboard`  
📅 **Fecha:** Octubre 2025  
👤 **Autor:** Equipo Ecobici-AutoML  