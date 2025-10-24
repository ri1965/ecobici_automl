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