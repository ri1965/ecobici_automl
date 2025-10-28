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
	@echo "  make predict      - Ejecuta predicciones multi-horizonte"
	@echo "  make tiles        - Genera tiles georreferenciados (último parquet)"
	@echo "  make app          - Lanza dashboard Streamlit (puerto $(PORT))"
	@echo "  make mlflow-ui    - Abre interfaz de MLflow (puerto 5001)"
	@echo "  make clean        - Limpia artefactos locales"

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
