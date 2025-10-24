# Guía APB (A Prueba de Balas) – Entorno + MLflow

Este documento explica cómo **levantar el entorno**, **probar MLflow** y **abrir la UI** sin parches manuales.

## Requisitos
- Anaconda/Miniconda instalado
- Git instalado

## 0) Preparación del entorno (reproducible)
1. Clonar el repo y entrar a la carpeta:
   ```bash
   git clone <URL_DEL_REPO>.git
   cd ecobici-automl

2.	Crear y activar el entorno:   
    ```bash
    conda env create -f environment.yml
    conda activate ecobici_automl

3.	Verificación rápida de librerías:
    ```bash
    make env-info
    pip check   # Debe mostrar: No broken requirements found.

## 1) MLflow: smoke test + UI local
1.	Ejecutar el smoke test (loggea un run mínimo):    
    ```bash
    make smoke-mlflow
Resultado esperado: Smoke test OK y un run en ./mlruns.

2) Estructura relevante del repo (resumen)
	•	environment.yml → define el entorno (con pines estables).
	•	tools/mlflow_smoke_test.py → prueba mínima de MLflow.
	•	mlruns/ → almacén local de experimentos de MLflow.
	•	Makefile → atajos: mlflow-ui, smoke-mlflow, env-info.

3) Solución de problemas comunes
	•	El puerto 5000 está ocupado: es normal en macOS por ControlCenter. Usamos 5001.
	•	No existe el comando mlflow: usamos el binario con ruta completa en el Makefile. Alternativa: python -c "from mlflow.server import app; app.run()".
	•	pip check reporta conflictos: están resueltos con los pines de environment.yml. Asegurate de correr:
    ```bash
    conda env update -f environment.yml --prune

    •	pkg_resources warning: es un aviso de deprecación. Mitigado con setuptools<81 en environment.yml.    

Una vez completado este paso, el proyecto está listo para avanzar al Paso 02: Higiene del repo + Camino Dorado (estructurar data/gold, mover lógica a src/, y targets de Makefile para preprocess/train/evaluate con logging en MLflow).

---

# Mini-commit sugerido
```bash
git checkout -b chore/docs-paso01
git add Makefile README_NUEVO.md tools/mlflow_ui.py 2>/dev/null || true
git commit -m "Docs APB Paso 01 + Makefile (mlflow-ui, smoke-mlflow, env-info)"

Perfecto 👌
Acá tenés el resumen conceptual y técnico del Paso 02 — “Higiene del repositorio + Camino Dorado”, para que lo incluyas en tu bitácora o documento principal (por ejemplo en docs/dev/bitacora.md o dentro del informe del TP).

⸻

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


