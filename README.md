🧭 Ecobici-AutoML

Repositorio limpio y reproducible — Etapa 2: Camino Dorado

Proyecto orientado al modelado automatizado de disponibilidad de bicicletas del sistema Ecobici (Buenos Aires), usando pipelines reproducibles basados en Python + Makefile.

Esta versión marca el punto de partida “Golden Path”: entorno validado, estructura consolidada y comandos mínimos para reproducir el flujo completo.

⸻

⚙️ 1. Activar entorno (A)

1️⃣ Crear el entorno de Conda:

conda env create -f environment.yml
conda activate ecobici_automl

💡 El archivo environment.yml contiene las dependencias base (Python ≥3.10, pandas, numpy, sklearn, pyyaml, etc.).

2️⃣ (Opcional) Verificar librerías específicas:

pip install -r requirements.txt


⸻

🧩 2. Preparar estructura (P)

Ejecutá el setup automático del proyecto:

make setup

Esto crea las carpetas necesarias:

data/{raw,interim,curated,external,processed}
outputs/{models,predictions,tiles,visuals,metrics}
reports/{experiments,logs,monitoring}
docs/dev, envs, models, notebooks, scripts, src, tools, app


⸻

🔍 3. Verificar entorno y configuración (B)

Ejecutá los chequeos automáticos:

make verify-env

Salida esperada:

🩺 Python:
3.10.x ...
🩺 Librerías clave:
yaml/pandas/numpy/sklearn: OK
🩺 Leyendo config: config/config.yaml
Config OK. project: Ecobici-AutoML

Luego generá el árbol del repositorio:

make repo-tree

Esto crea docs/dev/repo_tree.txt con la estructura actualizada.

⸻

🗂️ Estructura mínima esperada

ecobici-automl/
├── app/
├── archive/
├── config/
│   └── config.yaml
├── data/
│   └── raw/
├── docs/
├── envs/
├── models/
├── notebooks/
├── outputs/
├── reports/
├── scripts/
├── src/
│   └── __init__.py
├── tools/
│
├── .gitignore
├── Makefile
├── environment.yml
├── requirements.txt
├── README.md


⸻

🧹 Limpieza rápida

Para borrar artefactos temporales y dejar el repo listo:

make clean


⸻

📘 Próximos pasos (Etapa 3)
	•	Agregar targets train, predict, mlflow-ui y app en el Makefile.
	•	Conectar pipeline de entrenamiento y despliegue con MLflow.
	•	Documentar el flujo completo en el README principal.

⸻

🏁 Estado actual:
	•	Entorno validado y reproducible.
	•	Estructura “Golden Path” lista.
	•	Flujo de setup y verificación automatizado vía Makefile.

⸻

¿Querés que te lo deje ya con los íconos y formato markdown final (para copiar y pegar en tu README.md con estilo visual, emojis y resaltado de secciones)?