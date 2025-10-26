Excelente 👌 — acá tenés el archivo ONBOARDING.md listo para colgar en la raíz del repo.
Está diseñado para que un nuevo colaborador entienda qué hace el proyecto, cómo correrlo en 5 minutos y cómo meterse más a fondo (flujo completo y contribución).

⸻


# 🚲 Proyecto Ecobici-AutoML — Guía de Onboarding

> **Objetivo:** predecir la disponibilidad de bicicletas en las estaciones de Ecobici Buenos Aires (8 h – 22 h) mediante modelos AutoML, y visualizar resultados en un dashboard interactivo (Streamlit).

---

## 🧭 1. Contexto general

Este proyecto forma parte del TP – Fundamentos de Aprendizaje Automático.  
Utiliza datos abiertos del GCBA (APIs de Ecobici) y modelos entrenados con **PyCaret**, **H2O** y **FLAML** para predecir la disponibilidad horaria de bicicletas por estación.  
El pipeline completo cubre desde la ingesta y generación de features hasta la predicción y despliegue en una app local.

Estructura típica de flujo:

data/raw → data/curated → features → model-ready → AutoML → modelo final → predicciones → tiles → dashboard

---

## ⚙️ 2. Instalación y primer arranque (5 minutos)

### 2.1. Clonar el repositorio
```bash
git clone <URL_DEL_REPO>
cd ecobici-automl

2.2. Crear y activar el entorno

conda env create -f environment.yml
conda activate ecobici_automl

2.3. Verificar entorno y estructura

make verify-env
make repo-tree

✅ Esto confirma versión de Python, librerías clave y genera el árbol del proyecto en docs/dev/repo_tree.txt.

2.4. Revisar archivos necesarios

Asegurate de tener:
	•	data/curated/ → con los .parquet ya curados.
	•	models/03D/best_model.pkl → modelo entrenado.
	•	config/config.yaml → configuración por defecto.

2.5. Ejecutar la app

streamlit run app/main.py

Luego abrí http://localhost:8501
Ahí vas a ver el mapa de calor y la disponibilidad de bicis por estación.

Si el puerto 8501 está ocupado, usá:
streamlit run app/main.py --server.port 8502

⸻

🧩 3. Entender el flujo completo (para quien quiera “meter mano”)

3.1. Lecturas recomendadas (orden sugerido)
	1.	README.md → guía rápida del proyecto.
	2.	docs/FAA_avanzado.doc → estructura avanzada, carpetas y scripts.
	3.	docs/Explicacion_Flujo.docx → descripción detallada del pipeline.
	4.	docs/dev/repo_tree.txt → mapa actualizado del repositorio.

⸻

3.2. Convención de notebooks y scripts

Etapa	Prefijo	Descripción	Carpeta / Ejemplo
00–01	Ingesta y limpieza	Lectura de datos crudos, depuración	notebooks/01_ingesta.ipynb
02	Feature engineering	Lags, rolling, estacionalidad, dataset model-ready	02_Features.ipynb
03	AutoML	PyCaret, H2O, FLAML, comparativa final	03D_Comparativa.ipynb
04	Evaluación y despliegue	Predicciones batch y tiles para app	04C_Despliegue_y_Predicción_Batch.ipynb

Scripts principales (en src/):
	•	preprocess.py → genera dataset model-ready.
	•	train_pycaret.py / train_h2o.py / train_flaml.py → entrena modelos.
	•	predict_batch.py → genera predictions/ y tiles/.
	•	app/main.py → dashboard (Streamlit).

⸻

3.3. Flujo resumido

1️⃣ Ingesta:         data/raw → data/curated
2️⃣ Features:        curated → model_ready.parquet
3️⃣ Entrenamiento:   AutoML (03*) → modelo final.pkl
4️⃣ Evaluación:      métricas en reports/
5️⃣ Predicción:      predict_batch.py → predictions/ y tiles/
6️⃣ App:             visualización interactiva (Streamlit)


⸻

🧠 4. Re-entrenar el modelo (camino dorado)
	1.	Confirmar data
data/curated/ecobici_model_ready.parquet
(si no existe, correr el notebook 02 o src/preprocess.py)
	2.	Entrenar (AutoML)
Abrir notebook 03* (PyCaret / H2O / FLAML).
Al finalizar, se guarda el mejor modelo en models/03D/best_model.pkl.
	3.	Generar predicciones batch

python src/predict_batch.py \
    --model "models/03D/best_model.pkl" \
    --input "data/curated/ecobici_model_ready.parquet" \
    --stations "data/curated/station_information.parquet" \
    --horizons 1 3 6 12 \
    --asof "2025-10-25 10:00:00"


	4.	Ejecutar la app

streamlit run app/main.py



⸻

🧩 5. Personalización
	•	Config: config/config.yaml
	•	Rutas de datos, horizontes, estación por defecto.
	•	App: app/main.py
	•	Umbrales de color (semáforo), textos, layout.
	•	Modelos: notebooks 03*
	•	Cambiar métrica (RMSE, MAE, R²), semilla o tiempo de entrenamiento.

⸻

🧾 6. Métricas y reportes generados

Archivo	Descripción
reports/backtest_metrics_by_horizon.csv	Resultados por horizonte (1–3–6–12 h).
reports/error_by_hour_and_horizon.csv	Análisis horario del error.
reports/holdout_month_metrics.csv	Validación realista (mes más reciente).
reports/spatial_generalization.csv	(Opcional) evaluación espacial.


⸻

🤝 7. Contribución (opcional)
	1.	Crear rama nueva:

git checkout -b feat/<descripcion>


	2.	Hacer cambios, agregar commits:

git add .
git commit -m "feat: descripción corta del cambio"


	3.	Subir la rama y abrir PR:

git push origin feat/<descripcion>


	4.	Checklist antes del merge:
	•	✅ Entorno reproducible (make verify-env OK)
	•	✅ Reports actualizados
	•	✅ App funcionando con nuevo modelo

⸻

🧭 8. Estructura general del repositorio

├── app/                     # Streamlit dashboard
├── config/                  # Archivos de configuración YAML
├── data/
│   ├── raw/                 # Datos crudos (bronze)
│   ├── curated/             # Datos curados (silver/gold)
│   └── predictions/         # Resultados de predicciones batch
├── docs/                    # Documentación del proyecto
├── models/                  # Modelos entrenados
├── notebooks/               # Jupyter Notebooks por etapa
├── reports/                 # Métricas y evaluaciones
├── src/                     # Scripts reproducibles
├── Makefile                 # Atajos de flujo (make verify-env, make app, etc.)
├── environment.yml          # Dependencias Conda
└── README.md


⸻

🧩 9. Notas técnicas y compatibilidad
	•	Python: 3.10
	•	Entorno: ecobici_automl
	•	Frameworks: PyCaret, FLAML, H2O
	•	Java: H2O requiere JDK 11 (ver setup.sh o README H2O).
	•	Puertos: Streamlit → 8501; MLflow → 5001.
	•	Sistema: probado en macOS y Linux.

⸻

🎯 10. Próximos pasos sugeridos
	•	Revisar docs/FAA_avanzado.doc para entender el flujo avanzado.
	•	Probar la app localmente.
	•	(Opcional) Re-entrenar modelo con un horizonte distinto.
	•	Abrir tu primera PR de prueba 🚀

⸻

Autor original: [Roberto Inza]
Repositorio académico: TP – Fundamentos de Aprendizaje Automático (Ecobici AutoML)
Última actualización: 25 Oct 2025

⸻
---

## 🧰 11. Comandos rápidos del Makefile

El proyecto incluye un **Makefile** con atajos que facilitan el flujo completo sin abrir notebooks.

| Comando | Descripción |
|----------|--------------|
| `make verify-env` | Verifica versión de Python, librerías clave y configuración YAML. |
| `make repo-tree` | Genera un árbol actualizado del repositorio en `docs/dev/repo_tree.txt`. |
| `make features` | Ejecuta el pipeline de features (equivalente a etapa 02). |
| `make train` | Lanza el entrenamiento automático con el framework por defecto (PyCaret). |
| `make predict` | Genera predicciones batch y tiles para la app (`src/predict_batch.py`). |
| `make app` | Inicia la app de Streamlit (`streamlit run app/main.py`). |
| `make mlflow-ui` | (Opcional) Abre el servidor local de MLflow para visualizar experimentos. |
| `make clean` | Limpia outputs temporales y logs. |
| `make freeze` | Exporta dependencias actuales (`requirements_freeze.txt`). |

> 💡 **Tip:** Podés encadenar comandos, por ejemplo:  
> `make features && make train && make predict && make app`

---

**Fin del documento — Guía de Onboarding Ecobici-AutoML**