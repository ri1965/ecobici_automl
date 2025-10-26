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

Perfecto 👌
Acá tenés el bloque listo para pegar directamente en tu archivo ONBOARDING.md, reemplazando la sección anterior de notebooks.
El formato mantiene el estilo limpio, con íconos y estructura clara en Markdown:

⸻

📘 Estructura y descripción de notebooks — Ecobici-AutoML

El flujo completo del proyecto se implementa mediante una secuencia de notebooks dentro de la carpeta notebooks/.
Cada notebook aborda una etapa específica del pipeline, desde la carga de datos hasta la generación de predicciones para el dashboard final.
Todos los pasos se apoyan en los datos procesados del sistema Ecobici (API GCBA) y pueden ejecutarse de forma independiente o encadenada.

Notebook	Etapa	Descripción
🟩 PASO_1_Cargar_chequear_y_preparar_features.ipynb	Ingesta y preparación de datos	Carga los datasets crudos provenientes de la API Ecobici, unifica las fuentes, verifica consistencia y genera variables iniciales (lags, medias móviles, estacionalidad).
🟨 PASO_2_Split_temporal_Train_Val_Test.ipynb	División temporal	Realiza la separación temporal de los datos para entrenamiento, validación y test, garantizando una correcta evaluación en escenarios futuros.
🟦 PASO_3_Modelo_base.ipynb	Modelo base	Construye un modelo de referencia (baseline) para medir el desempeño inicial frente a los modelos AutoML posteriores.
🟪 PASO_4_Predicción_batch_multi_horizonte.ipynb	Generación de predicciones	Ejecuta predicciones por lotes a múltiples horizontes (1 h, 3 h, 6 h, 12 h) y guarda los resultados en outputs/predictions/*.parquet.
🟥 PASO_5_Evaluación_Tracking.ipynb	Evaluación y seguimiento	Calcula métricas de error (RMSE, MAE, R², MAPE) por horizonte y hora del día. Produce los reportes en la carpeta reports/.
🟧 PASO_6A_AutoML_con_PyCaret.ipynb	AutoML – PyCaret	Entrena y compara modelos (LightGBM, CatBoost, RF, etc.) mediante PyCaret. Registra los experimentos en MLflow.
🟦 PASO_6B_AutoML_Flami.ipynb	AutoML – FLAML	Ejecuta un flujo alternativo de AutoML liviano con FLAML, optimizando recursos y tiempos de ejecución.
🟫 PASO_7_compare_models.ipynb	Comparativa final	Integra los resultados de PyCaret y FLAML, seleccionando el mejor modelo según desempeño y generalización temporal.
🟩 PASO_8_predict_dashboard.ipynb	Despliegue y dashboard	Genera los archivos finales de predicciones (tiles parquet) que alimentan el dashboard Streamlit con el mapa de calor de disponibilidad.


⸻

💡 Consejo: todos los notebooks registran sus resultados en carpetas versionadas (reports/, models/, outputs/), lo que permite reconstruir cualquier etapa del pipeline y comparar configuraciones sin reentrenar completamente el modelo.



⚙️ Scripts principales — Carpeta src/

Los scripts dentro de src/ implementan las funciones principales del pipeline de entrenamiento, evaluación y despliegue.
Cada uno puede ejecutarse desde CLI o ser importado por los notebooks, garantizando modularidad y reproducibilidad.

Script	Descripción
prepare_features.py	Genera el dataset model-ready: limpieza, cálculo de lags, rolling windows y variables estacionales.
split_time.py	Realiza la división temporal en conjuntos de entrenamiento, validación y test, respetando la secuencia cronológica.
train.py	Entrena el modelo base o realiza pruebas de referencia previas al AutoML.
automl_pycaret.py	Implementa el flujo AutoML usando PyCaret, entrenando múltiples modelos y registrando resultados en MLflow.
automl_flaml.py	Alternativa AutoML ligera basada en FLAML; busca hiperparámetros óptimos minimizando costo computacional.
backtest_mlflow.py	Ejecuta validaciones cruzadas temporales (rolling backtest) y almacena métricas en MLflow.
compare_and_register.py	Compara los modelos generados por diferentes frameworks (PyCaret / FLAML) y registra el mejor.
register_best.py	Registra en MLflow y/o en disco local el modelo final aprobado para producción.
predict_batch.py	Genera predicciones por lotes y exporta los resultados en outputs/predictions/*.parquet.
make_tiles.py	Convierte las predicciones en tiles espaciales listos para ser consumidos por el dashboard.


⸻

💡 Nota: todos los scripts utilizan rutas relativas y configuración centralizada, por lo que se recomienda ejecutar los comandos desde la raíz del proyecto o a través del Makefile.



Excelente, te dejo el texto actualizado y adaptado a la versión moderna del proyecto (post-reestructuración de notebooks y scripts), listo para pegar en tu ONBOARDING.md.
Mantiene tu estilo, emojis y formato limpio, pero con referencias actualizadas a src/ y a los nuevos nombres de notebooks.

⸻

⚡️ 3.3. Flujo resumido

1️⃣ Ingesta: data/raw → data/curated
2️⃣ Features: curated → model_ready.parquet
3️⃣ Entrenamiento: notebooks PASO_6* o scripts src/automl_pycaret.py / src/automl_flaml.py → modelo final .pkl
4️⃣ Evaluación: métricas generadas en reports/
5️⃣ Predicción: src/predict_batch.py → outputs/predictions/ y outputs/tiles/
6️⃣ App: visualización interactiva con Streamlit

⸻

🧠 4. Re-entrenar el modelo (Camino Dorado)

1️⃣ Confirmar data
Archivo base:
data/curated/ecobici_model_ready.parquet
(Si no existe, ejecutar PASO_1 y PASO_2 o el script src/prepare_features.py.)

2️⃣ Entrenar (AutoML)
Abrir el notebook PASO_6A_AutoML_con_PyCaret.ipynb o PASO_6B_AutoML_Flami.ipynb.
Al finalizar, el mejor modelo se guarda en:
models/best_model.pkl

3️⃣ Generar predicciones batch

python src/predict_batch.py \
  --model "models/best_model.pkl" \
  --input "data/curated/ecobici_model_ready.parquet" \
  --stations "data/curated/station_information.parquet" \
  --horizons 1 3 6 12 \
  --asof "2025-10-25 10:00:00"

4️⃣ Ejecutar la app

streamlit run app/main.py


⸻

🧩 5. Personalización
	•	Config: config/config.yaml
	•	Rutas: datos, horizontes, estación por defecto
	•	App: app/main.py
	•	Colores: umbrales del semáforo y layout
	•	Modelos: notebooks PASO_6* o scripts src/automl_*.py
	•	Métricas: cambiar RMSE, MAE, R² o semilla en YAML o Makefile

⸻

🧾 6. Métricas y reportes generados

Archivo	Descripción
reports/backtest_metrics_by_horizon.csv	Resultados por horizonte (1–3–6–12 h)
reports/error_by_hour_and_horizon.csv	Análisis horario del error
reports/holdout_month_metrics.csv	Validación realista (último mes disponible)
reports/spatial_generalization.csv	(Opcional) Evaluación espacial de estaciones


⸻

🤝 7. Contribución (opcional)

1️⃣ Crear rama nueva

git checkout -b feat/<descripcion>

2️⃣ Agregar commits

git add .
git commit -m "feat: descripción corta del cambio"

3️⃣ Subir rama y abrir PR

git push origin feat/<descripcion>

4️⃣ Checklist antes del merge
	•	✅ Entorno reproducible (make verify-env)
	•	✅ Reports actualizados
	•	✅ App funcionando con nuevo modelo

⸻

🧭 8. Estructura general del repositorio

├── app/                     # Dashboard Streamlit
├── config/                  # Archivos YAML de configuración
├── data/
│   ├── raw/                 # Datos crudos (bronze)
│   ├── curated/             # Datos curados (silver/gold)
│   └── predictions/         # Resultados de predicciones batch
├── docs/                    # Documentación del proyecto
├── models/                  # Modelos entrenados
├── notebooks/               # Jupyter Notebooks por etapa
├── outputs/                 # Predicciones y tiles para el dashboard
├── reports/                 # Métricas y evaluaciones
├── src/                     # Scripts reproducibles (entrenamiento, batch, tiles)
├── Makefile                 # Atajos de flujo (make verify-env, make app, etc.)
├── environment.yml          # Dependencias Conda
└── README.md


⸻

🧩 9. Notas técnicas y compatibilidad
	•	Python: 3.10
	•	Entorno: ecobici_automl
	•	Frameworks: PyCaret, FLAML, H2O
	•	Java: requerido JDK 11 para H2O (ver setup.sh)
	•	Puertos: Streamlit → 8501 · MLflow → 5001
	•	Sistema: probado en macOS y Linux

⸻

🎯 10. Próximos pasos sugeridos
	•	Revisar docs/FAA_avanzado.docx para entender el flujo avanzado
	•	Probar la app localmente
	•	(Opcional) Re-entrenar el modelo con otro horizonte
	•	Crear tu primera Pull Request 🚀

⸻

Autor: Roberto Inza
Repositorio académico: TP – Fundamentos de Aprendizaje Automático (Ecobici-AutoML)
Última actualización: 26 Oct 2025

⸻

🧰 11. Comandos rápidos del Makefile

Comando	Descripción
make verify-env	Verifica versión de Python, librerías clave y config YAML
make repo-tree	Genera docs/dev/repo_tree.txt
make features	Ejecuta pipeline de features (etapa 02)
make train	Entrena modelo AutoML (PyCaret por defecto)
make predict	Genera predicciones batch y tiles (src/predict_batch.py)
make app	Inicia app Streamlit (app/main.py)
make mlflow-ui	Abre interfaz local de MLflow (localhost:5001)
make clean	Limpia outputs temporales y logs
make freeze	Exporta dependencias (requirements_freeze.txt)

💡 Tip:
Podés encadenar comandos:

make features && make train && make predict && make app


