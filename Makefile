# ======================================================
# Ecobici-AutoML — Makefile (v1.0 estable)
# ======================================================

# ---- Variables generales
PY = python
BACKEND_URI = ./mlruns

# ---- Ayuda
.DEFAULT_GOAL := help
help:
	@echo "🎛️  Comandos disponibles:"
	@echo "  make verify-env     # Verifica entorno y librerías base"
	@echo "  make run            # Ejecuta el pipeline completo (1→4)"
	@echo "  make predict        # Solo genera nuevas predicciones"
	@echo "  make mlflow-ui      # Abre interfaz MLflow local (puerto 5001)"
	@echo "  make clean          # Limpia artefactos locales"

# ---- Verificación del entorno
verify-env:
	$(PY) --version
	$(PY) -c "import yaml, pandas, numpy, sklearn; print('✅ Entorno y librerías base OK')"

# ---- Ejecución principal
run:
	@echo "🚀 Ejecutando pipeline completo Ecobici-AutoML..."
	$(PY) run.py

# ---- Solo predicción (usa modelo Champion actual)
predict:
	@echo "🧮 Predicción con Champion (sin reentrenar)…"
	python run.py --mode predict

# ---- MLflow UI (seguimiento de experimentos)
mlflow-ui:
	@echo "🌐 Abriendo MLflow UI en http://127.0.0.1:5001 ..."
	$(PY) -m mlflow ui --backend-store-uri $(BACKEND_URI) --host 127.0.0.1 --port 5001

# ---- Limpieza ligera
clean:
	rm -rf predictions/* tiles/* reports/logs/* || true
	@echo "🧹 Limpieza completa"

# ---- Mostrar últimos tiles generados
show-latest-tiles:
	@latest=$$(ls -t tiles/tiles_*.parquet | head -n1); \
	echo "Último tiles: $$latest"; \
	python - <<'PY' $$latest
import sys, pandas as pd
print(pd.read_parquet(sys.argv[1]).head())
PY	