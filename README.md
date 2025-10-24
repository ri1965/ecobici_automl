🚲 Proyecto EcoBici AutoML

Este proyecto implementa un pipeline end-to-end que combina:
	•	Ingesta de datos desde la API pública de EcoBici GCBA
	•	Análisis exploratorio automático (AutoEDA)
	•	Generación automática de variables (AutoFeature Engineering)
	•	Entrenamiento de modelos predictivos con AutoML (PyCaret)
	•	Y una app interactiva en Streamlit para usuarios
	
⸻	
	
🧩 0. Qué necesitás tener antes

Antes de tocar nada, asegurate de tener instalados:
	•	🐍 Anaconda o Miniconda
	•	💻 Visual Studio Code o cualquier editor de texto
	•	📘 Jupyter Notebook / JupyterLab (viene con Anaconda)
	•	🌐 Conexión a internet (para instalar librerías y dependencias)
	•	📁 Al menos un archivo .parquet con datos de Ecobici en la carpeta data/

⸻	
	
🏗️ 1. Crear el proyecto base

Abrí la Terminal y copiá/pegá esto línea por línea:

	# Crear carpeta raíz del proyecto y entrar	
	mkdir ecobici-automl
	cd ecobici-automl

	# Crear subcarpetas
	mkdir data notebooks src models app

	# Crear archivos base
	touch README.md requirements.txt setup_environment.sh

	# Crear archivos dentro de src/
	touch src/ingest.py src/build_features.py src/train_model.py src/predict_service.py src/utils.py

	# Crear archivo base de la app
	touch app/streamlit_app.py		

💡 Esto genera toda la estructura vacía lista para empezar.

⸻

⚙️ 2. Configurar el entorno automático

Abrí el archivo setup_environment.sh (con VS Code o bloc de notas) y pegá este contenido:

	#!/bin/bash
	echo "⚙️ Creando entorno conda ecobici_automl (Python 3.10)..."
	conda create -n ecobici_automl python=3.10 -y

	echo "📦 Activando entorno e instalando dependencias..."
	source $(conda info --base)/etc/profile.d/conda.sh
	conda activate ecobici_automl

	pip install --upgrade pip setuptools wheel
	pip install ipykernel
	python -m ipykernel install --user --name ecobici_automl --display-name "EcoBici (Py3.10)"

	echo "📚 Instalando librerías del proyecto..."
	pip install -r requirements.txt

	echo "✅ Entorno EcoBici listo para usar."

Guardá el archivo (Ctrl+S o Cmd+S) y luego ejecutá:

	chmod +x setup_environment.sh   # solo la primera vez
	./setup_environment.sh

⸻

📦 3. Instalar librerías (requirements.txt)

Abrí el archivo requirements.txt y pegá este contenido:

	# Librerías principales del proyecto
	pandas==2.1.4
	numpy==1.26.4
	scipy==1.11.4
	scikit-learn==1.4.2
	matplotlib==3.7.5
	pyarrow
	jinja2>=3.1

	# AutoEDA
	ydata-profiling
	autoviz
	sweetviz
	dtale

	# AutoFeature Engineering
	featuretools

	# AutoML
	pycaret==3.3.2
	lightgbm==4.6.0
	sktime==0.26.0
	scikit-base<0.8

	# App y visualización
	streamlit
	plotly>=5.14
	requests>=2.31

💬 No hace falta instalar nada manualmente: el script setup_environment.sh ya lo hará todo.

⸻

📊 4. Cargar datos de prueba

Colocá tus archivos .parquet (descargados o preparados) dentro de la carpeta data/.

Ejemplo:

	ecobici-automl/
 	├── data/
 	│    ├── eco_snapshot_2024_01.parquet
 	│    └── eco_snapshot_2024_02.parquet

⸻

🧮 5. Ejecutar el script de preparación (limpieza y dataset base)

Creá el archivo:

	src/prepare_from_parquet.py
	
y pegá el código del preprocesamiento (el que se te proporcionó en clase o en este repositorio).

Luego ejecutá en la terminal:

	conda activate ecobici_automl
	python src/prepare_from_parquet.py --glob "data/*.parquet" --emit-train --horizons "10,20"	
	
Esto genera los archivos:

	data/curated/ecobici_timeseries.parquet
	data/training/train_10min.parquet
	data/training/train_20min.parquet	
	

⸻

🧠 6. Verificar que todo funcione

Abrí Jupyter Notebook y seleccioná el kernel EcoBici (Py3.10).

Probá con este código:

	import pandas as pd

	base = pd.read_parquet("data/curated/ecobici_timeseries.parquet")
	train10 = pd.read_parquet("data/training/train_10min.parquet")

	print(base.shape, train10.shape)
	base.head()	
	
Si ves columnas como hour, dow, capacity, etc. → ✅ ¡Todo está OK!

⸻

🔧 Anexo técnico: comandos manuales (solo si el script falla)

Estos pasos son equivalentes al script setup_environment.sh, pero ejecutados manualmente.

	conda create -n ecobici_automl python=3.10 -y
	conda activate ecobici_automl
	pip install ipykernel
	python -m ipykernel install --user --name ecobici_automl --display-name "EcoBici (Py3.10)"
	pip install -r requirements.txt	
	
📍 Usar solo si el archivo setup_environment.sh no se puede ejecutar correctamente.


⸻

✅ Estado actual
✔️ Estructura de carpetas creada
✔️ Entorno reproducible configurado
✔️ Librerías instaladas
✔️ Dataset limpio y listo para AutoEDA	

ecobici-automl/
├── data/                 # Datasets descargados o procesados
├── notebooks/            # Notebooks exploratorios / experimentales
├── src/                  # Código fuente principal
│   ├── ingest.py         # Ingesta desde API (pendiente si no se usa parquet)
│   ├── build_features.py # Ingeniería automática de características (AutoFE)
│   ├── train_model.py    # Entrenamiento de modelos (AutoML)
│   ├── predict_service.py# Predicciones y lógica de servicio
│   └── utils.py          # Funciones auxiliares reutilizables
├── models/               # Modelos entrenados o artefactos generados
├── app/                  # Aplicación final (Streamlit)
│   └── streamlit_app.py
├── requirements.txt      # Dependencias del entorno
├── setup_environment.sh  # Script para reproducir el entorno automáticamente
└── README.md             # Documentación general del proyecto
