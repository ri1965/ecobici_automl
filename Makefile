# ==========================================================
# Ecobici-AutoML — Makefile v1.3 (tabs fixed)
# Flujo: prepare_features → predict_batch → make_tiles → app
# ==========================================================

PY := python
PORT := 8501
BACKEND_URI := ./mlruns

.PHONY: help verify-env features predict tiles app mlflow-ui clean

help:
	@echo "Comandos disponibles:"
	@echo "  make verify-env   - Verifica entorno y librerías base"
	@echo "  make features     - Genera dataset de features desde status_clean"
	@echo "  make automl       - AutoML (PyCaret, sin tuning)"
	@echo "  make automl-tune  - AutoML (PyCaret, con tuning)"
	@echo "  make predict      - Ejecuta predicciones multi-horizonte"
	@echo "  make tiles        - Genera tiles georreferenciados (último parquet)"
	@echo "  make app          - Lanza dashboard Streamlit (puerto $(PORT))"
	@echo "  make mlflow-ui    - Abre interfaz de MLflow (puerto 5001)"
	@echo "  make clean        - Limpia artefactos locales"
	@echo "  make train        - Orquestador: sólo entrenamiento (PyCaret/FLAML/both)"
	@echo "  make promote      - Orquestador: elige y registra Champion"
	@echo "  make refit        - Orquestador: reentrena Champion en full data"
	@echo "  make all          - Orquestador: pipeline completo"

verify-env:
	@$(PY) --version
	@$(PY) -c "import yaml, pandas, numpy, sklearn; print('✅ Entorno OK')"

features:
	@echo "🚧 Generando features (lags, señales temporales)..."
	@$(PY) src/prepare_features.py --in "data/raw/status_clean.parquet" --out "data/curated/ecobici_model_ready.parquet"
	@echo "✅ Features actualizados en data/curated/ecobici_model_ready.parquet"

predict:
	@echo "🚀 Ejecutando predicciones..."
	@$(PY) src/predict_batch.py
	@echo "✅ Archivo generado en predictions/latest.parquet"

tiles:
	@latest=$$(ls -t predictions/*.parquet 2>/dev/null | head -n1); \
	if [ -z "$$latest" ]; then echo "❌ No hay archivos en ./predictions"; exit 1; fi; \
	echo "📊 Usando $$latest"; \
	$(PY) src/make_tiles.py --pred "$$latest" --stations "data/curated/station_information.parquet" --out "tiles/latest_tiles.parquet"
	@echo "✅ Tiles generados en tiles/latest_tiles.parquet"

app:
	@echo "🌐 Abriendo dashboard en http://localhost:$(PORT) ..."
	@STREAMLIT_SERVER_PORT=$(PORT) streamlit run app/dashboard/main.py

mlflow-ui:
	@echo "🧠 Abriendo MLflow UI en http://127.0.0.1:5001 ..."
	@$(PY) -m mlflow ui --backend-store-uri $(BACKEND_URI) --host 127.0.0.1 --port 5001

clean:
	@rm -rf predictions/* tiles/* reports/logs/* 2>/dev/null || true
	@echo "🧹 Limpieza completa"

.PHONY: preprocess-slots features-slots

preprocess-slots:
	python src/preprocess_slots.py \
	--input data/curated/ecobici_model_ready.parquet \
	--output data/curated/ecobici_slots_train.parquet \
	--slots 09:00 13:00 16:00 20:00 \
	--how asof --asof_tolerance 5min --lead_minutes 10 \
	--ycol num_bikes_available --tscol ts_local

features-slots:
	python src/features_slots.py \
	--slots_input data/curated/ecobici_slots_train.parquet \
	--raw_input data/curated/ecobici_model_ready.parquet \
	--output data/curated/ecobici_slots_train_features.parquet \
	--tscol_raw ts_local --ycol_raw num_bikes_available \
	--max_lag 12 --ma_windows 3 6 12	

# -------- AutoML (PyCaret) --------
TUNE ?= 0
TUNE_ITERS ?= 50
METRIC ?= RMSE
FOLDS ?= 3

.PHONY: automl automl-tune
automl:
	@echo "▶️ AutoML (sin tuning)"
	python automl_pycaret.py \
		--train data/splits/train.parquet \
		--val   data/splits/val.parquet \
		--test  data/splits/test.parquet \
		--metric $(METRIC) \
		--folds $(FOLDS)

automl-tune:
	@echo "▶️ AutoML (con tuning: $(TUNE_ITERS) iter)"
	python automl_pycaret.py \
		--train data/splits/train.parquet \
		--val   data/splits/val.parquet \
		--test  data/splits/test.parquet \
		--metric $(METRIC) \
		--folds $(FOLDS) \
		--tune --tune-iters $(TUNE_ITERS)

# ========= Orquestador (run.py) =========
.PHONY: all train promote refit

all:
	@echo "🚀 Pipeline completo (prepare → train → promote → refit → predict)"
	@$(PY) run.py --mode all

train:
	@echo "🏋️ Entrenamiento (según automl.framework en config.yaml)"
	@$(PY) run.py --mode train

promote:
	@echo "👑 Selección de Champion (compare_and_register)"
	@$(PY) run.py --mode promote

refit:
	@echo "♻️  Reentrenando Champion en full data (retrain.mode)"
	@$(PY) run.py --mode refit

