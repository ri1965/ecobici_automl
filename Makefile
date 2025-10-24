.PHONY: mlflow-ui smoke-mlflow env-info

# Abre la UI de MLflow en el puerto 5001 (5000 suele estar ocupado por ControlCenter en macOS)
mlflow-ui:
	/Users/ri1965/anaconda3/envs/ecobici_automl/bin/mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001

# Corre el smoke test que deja un run en mlruns/
smoke-mlflow:
	conda run -n ecobici_automl python tools/mlflow_smoke_test.py

# Muestra versiones clave del entorno
env-info:
	conda run -n ecobici_automl python -c "import pycaret, flaml, mlflow, pandas, xgboost, lightgbm, sklearn; \
print('PyCaret:', pycaret.__version__); \
print('FLAML:', flaml.__version__); \
print('MLflow:', getattr(mlflow, '__version__', 'n/a')); \
print('pandas:', pandas.__version__); \
print('xgboost OK:', hasattr(xgboost, 'XGBRegressor')); \
print('lightgbm OK:', hasattr(lightgbm, 'LGBMRegressor')); \
print('sklearn:', sklearn.__version__)"

# ============================================
# UTILIDADES — Monitoreo y documentación
# ============================================

repo-tree:
	@echo "🗂️  Generando estructura actual del repositorio..."
	@tree -a -I '.git|data|outputs|mlruns|__pycache__|.ipynb_checkpoints' > docs/dev/repo_tree.txt || true
	@echo "✅ Árbol actualizado en docs/dev/repo_tree.txt"






🧩 Paso 02 — Higiene del Repositorio + Camino Dorado


🎯 Objetivo general

Garantizar que el proyecto Ecobici-AutoML tenga una estructura limpia, reproducible y mantenible, siguiendo buenas prácticas de ingeniería de datos y ciencia de datos aplicada.
El “Camino Dorado” define la ruta mínima y estandarizada que todo desarrollador puede seguir para ejecutar, mantener o extender el proyecto sin depender de archivos sueltos, configuraciones ocultas ni rutas locales.

⸻

🧱 Objetivos específicos
	1.	Normalizar la estructura del repositorio
Alinear todas las carpetas con la convención Data Science Cookiecutter (datos, código, artefactos, reportes, documentación).
Se eliminaron duplicados, versiones antiguas y rutas obsoletas.
	2.	Separar responsabilidades
	•	src/ y scripts/ → código fuente y ejecución.
	•	data/ → datasets organizados por nivel (raw, interim, curated, external).
	•	outputs/ → resultados de modelo (predicciones, tiles, métricas, visuales).
	•	reports/ → logs, monitoring y experimentos.
	•	envs/ → entornos y dependencias (environment.yml).
	•	app/ → despliegue y dashboards.
	•	archive/ → respaldo histórico y seguro de artefactos antiguos.
	•	docs/dev/ → documentación técnica y árbol de proyecto.
	3.	Limpieza e ignorados
Se incorporó un .gitignore completo que excluye datos crudos, resultados, checkpoints y caches (data/, outputs/, mlruns/, __pycache__/, .DS_Store, etc.), evitando contaminación del repositorio con archivos pesados o locales.
	4.	Archivado controlado (“No borrar, sino guardar”)
Todo elemento prescindible fue trasladado a archive/ARCH_YYYYMMDD/, preservando trazabilidad sin ensuciar el flujo activo de trabajo.
	5.	Automatización de control de estructura
Se agregó un target repo-tree al Makefile que genera automáticamente el mapa del repositorio (docs/dev/repo_tree.txt) cada vez que se ejecuta:

make repo-tree


	6.	Verificación de entorno y sincronización con Git
	•	Entorno ecobici_automl reconstruido desde environment.yml.
	•	Confirmación de que Git rastrea solo los elementos pertinentes y que el estado local y remoto están alineados.

⸻

📈 Resultado final

El repositorio quedó:
	•	Limpio: sin residuos, duplicados ni rutas rotas.
	•	Estandarizado: compatible con buenas prácticas reproducibles (MLOps, Data Ops).
	•	Seguro: artefactos antiguos respaldados.
	•	Versionado: Git + Makefile + tags listos para CI/CD o dashboard.

⸻

🧭 Salida esperada

Estructura dorada final (nivel 2):

.
├── app/
├── archive/ARCH_YYYYMMDD/
├── config/
├── data/{raw,interim,curated,external}/
├── docs/dev/
├── envs/
├── models/
├── notebooks/
├── outputs/{models,predictions,tiles,visuals,metrics}/
├── reports/{experiments,logs,monitoring}/
├── scripts/
├── src/
└── tools/


# Auditoría Paso 02.1 (no destructivo)
audit-02-1:
	@mkdir -p docs/dev
	@tree -a -L 4 -I '.git|mlruns|data|outputs|__pycache__|.ipynb_checkpoints' > docs/dev/ESTRUCTURA_ACTUAL.txt || true
	@{ echo "=== git status ==="; git status; echo; echo "=== git status --ignored ==="; git status --ignored; } > docs/dev/GIT_STATUS_02_1.txt
	@{ echo "Carpetas espejo dentro de notebooks/:"; \
	for d in data models monitoring notebooks predictions reports tiles visuals; do \
	  [ -d notebooks/$$d ] && echo " - notebooks/$$d"; \
	done; } > docs/dev/DUPLICADOS_NOTEBOOKS.txt
	@{ echo "Archivos > 25MB (excluye .git, data, outputs, mlruns):"; \
	find . -type f \( ! -path "./.git/*" ! -path "./data/*" ! -path "./outputs/*" ! -path "./mlruns/*" \) -size +25M -print 2>/dev/null; } > docs/dev/ARCHIVOS_PESADOS.txt
	@{ echo "Coincidencias de rutas absolutas típicas (/Users/ y C:\\):"; \
	echo " - /Users/"; grep -RIn --exclude-dir=".git" "/Users/" || true; \
	echo; echo " - C:\\ estilo Windows"; grep -RIn --exclude-dir=".git" -E "([A-Za-z]:\\\\)" || true; } > docs/dev/RUTAS_ABSOLUTAS.txt
	@cat docs/dev/CAMINO_DORADO.txt 2>/dev/null || { \
	echo "CAMINO DORADO (referencia)" > docs/dev/CAMINO_DORADO.txt; }
	@echo "✅ Audit 02.1 listo. Revisá docs/dev/*"