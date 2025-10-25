¡genial! acá tenés un README.md listo para pegar en la raíz del repo. Está pensado para que quien haga fork/clone pueda levantar todo desde cero y reproducir el pipeline end-to-end, con notas de AutoML, predicción batch, tiles y (opcional) dashboard.

⸻

Ecobici-AutoML

Pipeline reproducible para pronóstico de disponibilidad de bicicletas por estación y múltiples horizontes horarios. Incluye preparación de datos sin fuga, split temporal, baseline, AutoML (PyCaret/FLAML), backtesting con MLflow, predicción batch multi-horizonte y generación de tiles para un heatmap.

🔧 Empezar desde cero (fork/clone)
	1.	Clonar el repo

git clone https://github.com/<tu_usuario>/ecobici_automl.git
cd ecobici_automl

	2.	Crear entorno (Conda recomendado)

conda env create -f environment.yml
conda activate ecobici_automl

	3.	Estructura mínima de carpetas

make setup

Esto crea data/{raw,interim,curated,...}, reports/, models/, predictions/, tiles/, etc.
	4.	Colocar datos de entrada

	•	Ubicá tu parquet “crudo” en data/raw/status_clean.parquet.
	•	(Opcional) data/curated/station_information.parquet con station_id, lat, lon, name (para tiles).

Si no tenés station_information.parquet, podés empezar sin tiles y agregarlo después.

	5.	Verificar entorno

make verify-env

	6.	Correr el pipeline base (Pasos 1→2→3)

make all

Esto:
	•	Prepara features sin fuga → data/curated/ecobici_model_ready.parquet
	•	Crea splits temporales → data/splits/{train,val,test}.parquet
	•	Entrena baseline (RandomForest) → modelo en models/03D y métricas en reports/

	7.	Backtesting + MLflow (Paso 5)

make backtest
make mlflow-ui   # abre la UI en http://127.0.0.1:5001

	8.	AutoML (Paso 6)

	•	PyCaret (script):

make automl-pycaret

Guarda modelo ganador en models/03A_pycaret/pycaret_best_model.pkl
Métricas acumuladas en reports/automl_metrics_by_split.csv
	•	FLAML (opcional):
si activás el target FLAML, generará models/06_flaml/flaml_automl.pkl y actualizará reports/automl_bench.csv.

	9.	Predicción batch + tiles (Paso 8)

make predict
make tiles

	•	make predict genera predictions/predictions_YYYY-MM-DD-HHh.parquet y un alias predictions/latest.parquet.
	•	make tiles fusiona predicciones con coordenadas para heatmap → tiles/tiles_YYYY-MM-DD-HHh.parquet con alias tiles/latest.parquet.

	10.	Dashboard (opcional)
Si añadís la app de Streamlit:

make app


⸻

🗂️ Estructura (resumen)

ecobici_automl/
├── app/                      # (opcional) dashboard Streamlit
├── data/
│   ├── raw/                  # status_clean.parquet (input crudo)
│   ├── curated/              # ecobici_model_ready.parquet, station_information.parquet
│   └── splits/               # train/val/test.parquet
├── models/
│   ├── 03D/                  # baseline RF
│   └── 03A_pycaret/          # pycaret_best_model.pkl
├── predictions/              # predicciones batch + latest.parquet
├── tiles/                    # tiles para heatmap + latest.parquet
├── notebooks/                # paso a paso reproducible
├── reports/                  # métricas, backtest, automl_bench, etc.
├── src/
│   ├── prepare_features.py   # Paso 1
│   ├── split_time.py         # Paso 2
│   ├── train.py              # Paso 3 (baseline)
│   ├── backtest_mlflow.py    # Paso 5
│   ├── automl_pycaret.py     # Paso 6 (PyCaret)
│   ├── automl_flaml.py       # Paso 6 (FLAML) [opcional]
│   ├── model_select.py       # Paso 7 (registry/promoción) [opcional]
│   ├── predict_batch.py      # Paso 8.1
│   └── make_tiles.py         # Paso 8.2
├── environment.yml
└── Makefile


⸻

🧵 Flujo por pasos (resumen)
	1.	Paso 1 — Features: limpieza mínima + expansión segura de columnas tipo JSON y one-hot de baja cardinalidad → solo numéricas para modelos tradicionales.
	2.	Paso 2 — Split temporal: train/val/test respetando cronología (sin fuga).
	3.	Paso 3 — Baseline: RandomForest + baseline lag1 comparativo.
	4.	Paso 4 — Predicción recursiva (en notebooks): prueba multi-horizonte usando lags simulados.
	5.	Paso 5 — Backtesting + MLflow: expanding window y tracking de métricas/artefactos.
	6.	Paso 6 — AutoML: PyCaret / FLAML con validación temporal y registro en reports/.
	7.	Paso 7 — Selección/Registry (opcional): elegir mejor por métrica y registrar en MLflow Model Registry.
	8.	Paso 8 — Serving batch + tiles: predict_batch.py → make_tiles.py para dashboard.
	9.	Paso 9 — Orquestación/CI (opcional): GitHub Actions/Jenkins + checks de drift/calidad.
	10.	Paso 10 — DataOps/DVC (opcional): versionado de datasets/artefactos y remotes (S3/DagsHub).

⸻

🧪 Comandos útiles (Makefile)

make help          # ver targets disponibles
make setup         # crear estructura mínima
make all           # Paso 1→2→3
make backtest      # Paso 5
make mlflow-ui     # abrir UI
make automl-pycaret# Paso 6 (PyCaret)
make predict       # Paso 8.1 — batch multi-horizonte
make tiles         # Paso 8.2 — tiles heatmap
make clean         # limpiar caches livianos


⸻

🔍 Detalles clave

ASOF

Timestamp de referencia (p.ej. “2025-10-25 13:00:00”) desde el cual se generan predicciones hacia adelante. predict_batch.py armoniza timezone con tus datos para evitar errores tz-aware vs tz-naive.

Tiles

Archivo parquet con filas a nivel (station_id, timestamp_pred, h) y columnas lat, lon, yhat, etc. Se usa directo para mapas/heatmaps.

⸻

🧯 Troubleshooting común
	•	tz-naive vs tz-aware en comparaciones
Asegurate de no mezclar timestamps con y sin zona horaria. El pipeline ya localiza ASOF según el tz de tus datos.
	•	Feature names must match al predecir
Ocurre si el modelo fue entrenado con un set de features distinto.
Solución aplicada: predict_batch.py reindexa las features runtime al esquema exacto del train split (columnas y orden), rellena faltantes con 0 y descarta extras.
	•	latest.parquet no encontrado al hacer tiles
Usá make predict antes de make tiles. El Makefile crea/actualiza el alias predictions/latest.parquet.

⸻

🗄️ Git, datos y privacidad

Por defecto el .gitignore excluye:
	•	*.parquet, *.pkl, mlruns/, predictions/, tiles/, data/raw/, data/interim/.
	•	Subí a Git código y configs, no datos sensibles.
Si usás DVC (opcional), podés versionar metadata de datasets y empujar a S3/DagsHub.

⸻

📊 MLflow

Todos los entrenamientos y backtests registran parámetros, métricas y artefactos.
Abrí la UI con:

make mlflow-ui
# http://127.0.0.1:5001


⸻

🧠 AutoML
	•	PyCaret: validación temporal (fold_strategy="timeseries") y sólo features numéricas para evitar problemas con tipos complejos.
	•	FLAML: alternativa liviana con time budget configurable y logging a archivo.

Podés comparar ambos (y baseline) leyendo reports/automl_bench.csv / reports/automl_metrics_by_split.csv.

⸻

🗺️ Dashboard (opcional)

Con tiles/latest.parquet y predictions/latest.parquet podés construir:
	•	Heatmap por estación, selector de horizonte (1,3,6,12h), tooltips con yhat.
	•	Panel lateral con top estaciones por demanda/escasez.

⸻

📜 Licencia

Elegí y añadí una licencia en el repo (MIT/Apache-2.0, etc.) según corresponda.

⸻

¿Querés que te lo deje también como archivo listo para commit con un mensaje sugerido?