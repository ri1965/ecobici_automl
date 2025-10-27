# Ecobici-AutoML — Makefile (v1.0 estable)

# ---- Variables
PY ?= python
BACKEND_URI ?= ./mlruns
PORT ?= 8501

# ---- Targets "phony"
.PHONY: help verify-env run predict mlflow-ui app show-latest-tiles clean

# ---- Ayuda
.DEFAULT_GOAL := help
help:
	@echo "Comandos:"
	@echo "  make verify-env         # Verifica entorno y librerías base"
	@echo "  make run                # Ejecuta el pipeline completo (1→4)"
	@echo "  make predict            # Solo genera nuevas predicciones"
	@echo "  make mlflow-ui          # Abre MLflow UI (puerto 5001)"
	@echo "  make app                # Abre el dashboard Streamlit (puerto $(PORT))"
	@echo "  make show-latest-tiles  # Muestra el último tiles_* generado"
	@echo "  make clean              # Limpia artefactos locales"

# ---- Verificación del entorno
verify-env:
	@$(PY) --version
	@$(PY) -c "import yaml, pandas, numpy, sklearn; print('Entorno OK')"

# ---- Ejecución principal (end-to-end)
run:
	@echo "Ejecutando pipeline completo..."
	@$(PY) run.py

# ---- Solo predicción (usa modelo Champion actual)
predict:
	@echo "Predicción con Champion (sin reentrenar)…"
	@$(PY) run.py --mode predict

# ---- MLflow UI
mlflow-ui:
	@echo "Abriendo MLflow UI en http://127.0.0.1:5001 ..."
	@$(PY) -m mlflow ui --backend-store-uri $(BACKEND_URI) --host 127.0.0.1 --port 5001

# ---- Dashboard Streamlit
app:
	@echo "Abriendo dashboard en http://localhost:$(PORT) ..."
	@STREAMLIT_SERVER_PORT=$(PORT) streamlit run app/dashboard/main.py

# ---- Mostrar últimos tiles generados (sin heredoc)
show-latest-tiles:
	@latest=$$(ls -t tiles/tiles_*.parquet 2>/dev/null | head -n1); \\
	if [ -z "$$latest" ]; then echo "No hay tiles en ./tiles"; exit 1; fi; \\
	echo "Último tiles: $$latest"; \\
	$(PY) -c "import pandas as pd,sys; p='$$latest'; df=pd.read_parquet(p); print(df.head()); print(f'Rows={len(df)} | Stations={{df.station_id.nunique() if \"station_id\" in df.columns else \"?\"}} | Horizons={{sorted(df.h.unique().tolist()) if \"h\" in df.columns else \"?\"}}')"

# ---- Limpieza ligera
clean:
	@rm -rf predictions/* tiles/* reports/logs/* 2>/dev/null || true
	@echo "Limpieza completa"
