# ======================================================
# Ecobici-AutoML — Makefile base (repo limpio)
# ======================================================

# ---- Variables
PY         ?= python
CFG        ?= config/config.yaml
ENV_YML    ?= environment.yml

# ---- Ayuda (target por defecto)
.DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "Targets disponibles:"
	@echo "  make setup          # Crear entorno y estructura mínima"
	@echo "  make verify-env     # Verifica imports clave y versión de Python"
	@echo "  make repo-tree      # Guarda árbol del repo en docs/dev/repo_tree.txt"
	@echo "  make audit          # Snapshot corto de estado git e ignores"
	@echo "  make train          # Entrena modelo (usa src/train_model.py)"
	@echo "  make predict        # Predicción batch (usa src/predict_batch.py)"
	@echo "  make mlflow-ui      # Inicia MLflow UI (puerto 5000)"
	@echo "  make app            # Levanta la app de Streamlit"
	@echo "  make clean          # Limpia artefactos locales"

# ---- Setup de entorno y estructura
.PHONY: setup
setup:
	@echo "🔧 Creando estructura mínima…"
	@mkdir -p data/{raw,interim,curated,external,processed}
	@mkdir -p outputs/{models,predictions,tiles,visuals,metrics}
	@mkdir -p reports/{experiments,logs,monitoring}
	@mkdir -p docs/dev envs models notebooks scripts tools app src
	@touch src/__init__.py
	@echo "✅ Setup listo."

# ---- Verificación rápida del entorno
.PHONY: verify-env
verify-env:
	@echo "🩺 Python:"; \
	$(PY) -c "import sys; print(sys.version)"
	@echo "🩺 Librerías clave:"; \
	$(PY) -c "import yaml, pandas, numpy, sklearn; print('yaml/pandas/numpy/sklearn: OK')"
	@echo "🩺 Leyendo config: $(CFG)"; \
	$(PY) -c "import yaml,sys; p='$(CFG)'; f=open(p); cfg=yaml.safe_load(f); f.close(); print('Config OK. project:', cfg.get('project',{}).get('name','(sin nombre)'))"

# ---- Árbol del repositorio (excluye carpetas pesadas)
.PHONY: repo-tree
repo-tree:
	@echo "Generando árbol del repositorio…"
	@mkdir -p docs/dev
	@tree -a -I '.git|data|outputs|mlruns|archive|__pycache__|.ipynb_checkpoints' > docs/dev/repo_tree.txt || true
	@echo "✅ Árbol actualizado en docs/dev/repo_tree.txt"

# ---- Limpieza
.PHONY: clean
clean:
	@echo "🧹 Limpiando caches y logs ligeros…"
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -f logs.log 2>/dev/null || true
	@echo "✅ Limpieza completa."


# ======================================================
# Etapa de Datos y Modelado — Pasos 1 a 3
# ======================================================

.PHONY: all prepare split train

# ---- Variables de datos
RAW         = data/raw/status_clean.parquet
READY       = data/curated/ecobici_model_ready.parquet
EXPORT_CSV  = data/exports/ecobici_model_ready.csv
SPLITDIR    = data/splits
MODELDIR    = models/03D
REPORTS     = reports

# ---- Columnas clave
TS_COL = ts_local
ID_COL = station_id
Y_COL  = num_bikes_available

# ---- Paso 1: Preparar features sin fuga
prepare:
	@echo "▶️ Paso 1: preparando features desde $(RAW) → $(READY)"
	$(PY) src/prepare_features.py \
	  --in $(RAW) \
	  --out $(READY) \
	  --ts $(TS_COL) --id $(ID_COL) --y $(Y_COL) \
	  --csv $(EXPORT_CSV)

# ---- Paso 2: Split temporal (train/val/test)
split:
	@echo "✂️ Paso 2: creando splits temporales en $(SPLITDIR)"
	$(PY) src/split_time.py \
	  --in $(READY) \
	  --out $(SPLITDIR) \
	  --ts $(TS_COL) \
	  --ptrain 0.70 --pval 0.15 --minrows 100

# ---- Paso 3: Entrenamiento base (RandomForest)
train:
	@echo "🤖 Paso 3: entrenando modelo base RandomForest"
	$(PY) src/train.py \
	  --train $(SPLITDIR)/train.parquet \
	  --val   $(SPLITDIR)/val.parquet \
	  --test  $(SPLITDIR)/test.parquet \
	  --out   $(MODELDIR) \
	  --y $(Y_COL) --id $(ID_COL) --ts $(TS_COL) \
	  --reports $(REPORTS)

# ---- Pipeline completo (1→2→3)
all: prepare split train
	@echo "✅ Pipeline completo ejecutado correctamente."

# ---- Paso 4: Predicción batch multi-horizonte
.PHONY: predict

# Variables de predicción
PRED_DIR  = predictions
HORIZONS  = 1 3 6 12
FREQ_MIN  = 60
# ASOF por defecto = hora redondeada actual (podés sobreescribir: ASOF="2025-10-24 10:00:00")
ASOF ?= $(shell python - <<'PY'\nfrom datetime import datetime\nprint(datetime.now().strftime("%Y-%m-%d %H:00:00"))\nPY)

predict:
	@echo "🧮 Paso 4: predicción multi-horizonte → $(PRED_DIR)/predictions.parquet (ASOF=$(ASOF))"
	$(PY) src/predict_batch.py \
	  --model $(MODELDIR)/best_model.pkl \
	  --input $(READY) \
	  --train_split $(SPLITDIR)/train.parquet \
	  --asof "$(ASOF)" \
	  --horizons $(HORIZONS) \
	  --freq_minutes $(FREQ_MIN) \
	  --out_pred $(PRED_DIR) \
	  --outfile predictions.parquet

# —— Pipeline completo 1→4
.PHONY: all4
all4: prepare split train predict
	@echo "✅ Pipeline completo (1→4) ejecutado correctamente."

# ---- Paso 5: Evaluación temporal (Backtesting + MLflow)
backtest:
	@echo "📈 Paso 5: ejecutando backtesting + tracking MLflow"
	$(PY) src/backtest_mlflow.py \
	  --curated $(READY) \
	  --reports_dir $(REPORTS) \
	  --experiment ecobici_backtest_rf \
	  --y $(Y_COL) --id $(ID_COL) --ts $(TS_COL) \
	  --n_splits 3 --p_train_start 0.50 --p_train_end 0.75 --p_val 0.15 \
	  --n_estimators 300 --seed 42
	@echo "✅ Backtesting completado: ver resultados en $(REPORTS)/backtest_summary.csv" 

# ---- MLflow UI (explorar resultados del backtesting o entrenamiento)
.PHONY: mlflow-ui
mlflow-ui:
	@echo "🌐 Iniciando MLflow UI en http://127.0.0.1:5001 ..."
	conda activate ecobici_automl >/dev/null 2>&1 || true
	$(PY) -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001				


# ---- Paso 6: AutoML (PyCaret + FLAML)

automl-pycaret:
	@echo "🤖 Paso 6A: ejecutando AutoML con PyCaret"
	$(PY) src/automl_pycaret.py \
	  --train data/splits/train.parquet \
	  --val   data/splits/val.parquet \
	  --test  data/splits/test.parquet \
	  --outdir models/03A_pycaret \
	  --reports reports \
	  --y num_bikes_available --id station_id --ts ts_local \
	  --folds 3 --metric RMSE --seed 42 \
	  --mlflow-uri mlruns --experiment ecobici_pycaret_automl
	@echo "✅ PyCaret AutoML completado: ver métricas en reports/automl_metrics_by_split.csv"

automl-flaml:
	@echo "🧠 Paso 6B: ejecutando AutoML con FLAML"
	$(PY) src/automl_flaml.py \
	  --train data/splits/train.parquet \
	  --val   data/splits/val.parquet \
	  --test  data/splits/test.parquet \
	  --outdir models/06_flaml \
	  --reports reports \
	  --y num_bikes_available --id station_id --ts ts_local \
	  --time-budget 600 --metric RMSE --seed 42 \
	  --mlflow-uri mlruns --experiment ecobici_automl_flaml
	@echo "✅ FLAML AutoML completado: ver benchmark en reports/automl_bench.csv"

# ---- Alias general para correr ambos
automl-all: automl-pycaret automl-flaml
	@echo "🚀 AutoML (PyCaret + FLAML) finalizado; comparar resultados en reports/automl_bench.csv"


# ---- Paso 7: Comparación y registro del mejor modelo ----
select-best:
	@echo "⚖️  Paso 7: comparando y registrando el mejor modelo..."
	$(PY) src/register_best.py \
	  --bench_csv reports/automl_bench.csv \
	  --mlflow-uri mlruns \
	  --registry_name ecobici_best \
	  --out_selection reports/model_selection.csv


# ==========================================================
# 🧩 Paso 8 — Predicción servida & generación de tiles
# ==========================================================

# ---- Paso 8.1: Predicción batch multi-horizonte
.PHONY: predict
predict:
	@echo "🚴 Paso 8.1: generando predicciones batch (ASOF actual)"
	$(PY) src/predict_batch.py \
	  --model models/03A_pycaret/pycaret_best_model.pkl \
	  --input data/curated/ecobici_model_ready.parquet \
	  --train_split data/splits/train.parquet \
	  --out_pred predictions \
	  --outfile predictions_$$(date '+%Y-%m-%d-%Hh').parquet \
	  --asof "$$(date '+%Y-%m-%d %H:00:00')" \
	  --horizons 1 3 6 12 \
	  --freq_minutes 60
	@mkdir -p predictions
	@latest=$$(ls -t predictions/predictions_*.parquet 2>/dev/null | head -n1); \
	if [ -z "$$latest" ]; then \
	  echo "❌ No se encontró predictions/predictions_*.parquet. ¿Falló el paso anterior?"; exit 1; \
	fi; \
	cp -f "$$latest" predictions/latest.parquet; \
	echo "✅ Predicciones generadas → predictions/latest.parquet -> $$latest"

# ---- Paso 8.2: Generación de tiles para el dashboard
.PHONY: tiles
tiles:
	@echo "🗺️ Paso 8.2: generando tiles desde la última predicción"
	@mkdir -p tiles
	@test -f predictions/latest.parquet || (echo "❌ Falta predictions/latest.parquet. Corré 'make predict' primero." && exit 1)
	$(PY) src/make_tiles.py \
	  --pred predictions/latest.parquet \
	  --stations data/curated/station_information.parquet \
	  --out tiles/tiles_$$(date '+%Y-%m-%d-%Hh').parquet
	@latest_tile=$$(ls -t tiles/tiles_*.parquet 2>/dev/null | head -n1); \
	if [ -z "$$latest_tile" ]; then \
	  echo "❌ No se generó ningún tiles_*.parquet"; exit 1; \
	fi; \
	cp -f "$$latest_tile" tiles/latest.parquet; \
	echo "✅ Tiles listos → tiles/latest.parquet -> $$latest_tile"	


# ---- Smoke check: verificación rápida sin correr el pipeline completo
.PHONY: smoke
smoke:
	@echo "🧪 Smoke check: verificando entorno, benchmark, modelos y predicción mínima..."
	$(PY) tools/smoke_check.py	