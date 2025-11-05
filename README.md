# 🚴‍♂️ Ecobici-AutoML Dashboard  
> Predicción horaria de disponibilidad de bicicletas (Buenos Aires)  
> *PyCaret + FLAML + MLflow + Streamlit — Versión v1.0 (Etapa 8 consolidada)*  

---

### 📘 Descripción general  
**Ecobici-AutoML** es un pipeline integral de *Machine Learning* para predecir la disponibilidad horaria de bicicletas en el sistema público de la Ciudad de Buenos Aires.  

Integra todas las etapas del ciclo de vida del modelo:  
**ingesta → features → entrenamiento → selección → predicción → visualización**,  
automatizando la comparación entre frameworks **PyCaret** y **FLAML**, con registro en **MLflow** y despliegue mediante **Streamlit**.

---

## 🧭 Flujo resumido  

| Etapa | Descripción | Output |
|:--|:--|:--|
| **1️⃣ Ingesta** | `data/raw → data/curated` | `status_clean.parquet` |
| **2️⃣ Features** | `curated → ecobici_model_ready.parquet` | variables temporales, lags |
| **3️⃣ Entrenamiento** | AutoML (PyCaret & FLAML) | `models/03*/` |
| **4️⃣ Selección Champion** | compara métricas `val_rmse` y registra en MLflow | `models/Champion/best_model.pkl` |
| **5️⃣ Predicción** | `predict_batch.py` genera forecasts multi-horizonte | `predictions/latest.parquet` |
| **6️⃣ Dashboard** | visualización interactiva (Streamlit) | `tiles/tiles_*.parquet` |

---

## ⚙️ Instalación  

### 1️⃣ Clonar el repositorio  
```bash
git clone https://github.com/ri1965/ecobici_automl.git
cd ecobici_automl

2️⃣ Crear y activar entorno

Con Conda (recomendado):

conda create -n ecobici_automl python=3.10 -y
conda activate ecobici_automl
pip install -r requirements.txt

O alternativamente:

conda env create -f environment.yml
conda activate ecobici_automl


⸻

📦 Estructura mínima esperada

ecobici-automl/
├── data/
│   ├── raw/status_clean.parquet
│   └── curated/station_information.parquet
├── models/
│   ├── 03A_pycaret/
│   ├── 06_flaml/
│   └── Champion/best_model.pkl
├── reports/
│   ├── automl_metrics_by_split.csv
│   ├── model_selection.csv
│   └── champion_selection.json
├── predictions/latest.parquet
├── tiles/tiles_YYYYMMDD-HHMMSS.parquet
├── app/dashboard/main.py
├── run.py
└── Makefile


⸻

🚀 Ejecución rápida

🔸 Pipeline completo (entrenar + seleccionar + predecir)

make run

🔸 Sólo predicción (usando Champion actual)

make predict

🔸 Abrir interfaz de MLflow

make mlflow-ui

→ http://127.0.0.1:5001

🔸 Abrir el dashboard interactivo

make app

→ http://localhost:8501

⸻

📊 Archivos de salida principales

Tipo	Ruta	Descripción
Modelo Champion	models/Champion/best_model.pkl	Mejor modelo en producción
Manifiesto Champion	reports/champion_selection.json	Metadata de selección
Predicciones batch	predictions/latest.parquet	Forecast multi-horizonte
Tiles dashboard	tiles/tiles_*.parquet	Datos listos para mapa de calor
Benchmark AutoML	reports/automl_metrics_by_split.csv	Métricas por split
MLflow tracking	mlruns/	Historial de experimentos


⸻

🧩 Verificación rápida

Modelo Champion presente

ls -lh models/Champion/best_model.pkl

Predicciones válidas

python - <<'PY'
import pandas as pd
df = pd.read_parquet("predictions/latest.parquet")
print({"cols_ok": {"station_id","timestamp_pred","h","yhat"}.issubset(df.columns),
       "rows": len(df), "stations": df["station_id"].nunique()})
print(df.head(3))
PY

Últimos tiles generados

python - <<'PY'
from pathlib import Path
import pandas as pd
tiles = sorted(Path("tiles").glob("tiles_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
print("Último tiles:", tiles[0] if tiles else "(no hay)")
if tiles: print(pd.read_parquet(tiles[0]).head())
PY


⸻

🔁 Actualización de datos

Si se reemplazan los datos base (data/raw/status_clean.parquet), ejecutar nuevamente:

make run

Esto reentrena PyCaret + FLAML, selecciona nuevo Champion y actualiza predicciones / tiles.

⸻

🧹 Comandos útiles (Makefile)

Comando	Descripción
make verify-env	Verifica entorno y librerías clave
make run	Pipeline completo (1→4)
make predict	Genera predicciones con Champion actual
make mlflow-ui	Abre interfaz MLflow local
make app	Inicia dashboard Streamlit
make show-latest-tiles	Muestra el último tiles generado
make clean	Limpieza ligera de artefactos locales


⸻

🧠 Créditos y versión
	•	Proyecto desarrollado como Trabajo Práctico Final – Fundamentos de Aprendizaje Automático (Maestría en Ciencia de Datos, Univ. Austral)
	•	Autor: Adrián Firpo - Roberto Inza - Juan Manuel Lucero - Nicolás Souto
	•	Fecha: Noviembre 2025

⸻


