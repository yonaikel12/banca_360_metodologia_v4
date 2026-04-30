# Banca 360 Metodologia V4

Proyecto de portafolio orientado a banca retail que empaqueta un flujo MLOps end-to-end con ruteo dinamico de modelos por caso de negocio. La v4 hereda el rigor de la v3 y anade un catalogo extendido de modelos tabulares y temporales, gobernanza reforzada para deep learning y un contrato YAML capaz de cambiar entre clasificacion, prediccion de valor y forecasting.

## Propuesta de valor

- Caso BI + ML listo para demo con datos sinteticos, sin exponer informacion sensible.
- Pipeline modular reusable desde CLI y notebook, con la misma logica en ambos puntos de entrada.
- Capa metodologica adicional para parsimonia, incertidumbre, validacion temporal purgada, equidad y salud operativa del pipeline.
- Seleccion de benchmark guiada por `conf/settings.yaml`, con catalogos distintos para clasificacion, regresion de valor y forecasting temporal.
- Catalogo extendido con LightGBM, XGBoost, CatBoost, k-NN, GAM, MARS y modelos temporales como ARIMA y Prophet; si `py-earth` no esta disponible en Windows, MARS degrada a un surrogate spline-based mantenible sin romper el pipeline.
- Salidas accionables para retencion: scorecards, shortlist, brecha de valor contra consenso y politica bandit.

## Que incluye el proyecto

- `conf/settings.yaml`: configuracion central del caso, ruteo dinamico de modelos, umbrales y politica operativa.
- `data/raw/`: dataset sintetico y tablas base del marco metodologico.
- `notebooks/master_banca_360_mlops_orquestador.ipynb`: notebook maestro para narrativa, exploracion y demo.
- `src/banca_360_mlops/core/`: funciones de negocio, auditoria, modelado, validacion avanzada y visualizacion.
- `src/banca_360_mlops/services/`: servicio OO que encapsula el caso de negocio.
- `src/banca_360_mlops/pipeline/`: nodos y orquestacion del flujo end-to-end.

## Stack

Python 3.11, pandas, scikit-learn, matplotlib, seaborn, SHAP, statsmodels, MLflow y Jupyter, con extensiones opcionales para XGBoost, LightGBM, CatBoost, Prophet y GAM.

## Ejecucion local

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\banca360v4 run --project-root .
```

Cada corrida desde CLI deja ademas un log timestamped dentro de `data/processed/`, salvo que se indique una ruta explicita con `--log-path`.

El notebook reutiliza exactamente la misma configuracion y las mismas funciones del paquete, por lo que sirve como demo reproducible del pipeline. La v4 mantiene la parsimonia con AIC/BIC para familias lineales, obliga escalado en modelos sensibles a distancia como k-NN, extiende SHAP al benchmark no temporal y deja reglas de gobernanza especificas para MLP/LSTM cuando se activa deep learning.

## Notas para GitHub

- El repositorio conserva `data/raw/` para que el caso pueda ejecutarse desde cero.
- Los artefactos derivados (`data/interim/`, `data/processed/`, `mlruns/`) se regeneran localmente y se excluyen del control de versiones.
- No contiene credenciales ni datos reales de clientes.


## ajustes para caso real

Para utilizar la metodología de "banca_360_metodologia_v3" con un nuevo dataset, necesitarías ajustar principalmente los parámetros en el archivo de configuración central conf/settings.yaml, y potencialmente reemplazar o modificar cómo se carga el dataset. A continuación, detallo los parámetros clave que tendrías que cambiar, basándome en la estructura del proyecto:

1. Reemplazo del Dataset
El dataset actual se genera sintéticamente en el código (usando build_bank_client_case_dataset en src/banca_360_mlops/core/framework_bi_universal.py), pero hay un archivo CSV de respaldo en data/raw/bank360_dataset.csv.

Para usar un nuevo dataset real:
Reemplaza data/raw/bank360_dataset.csv con tu nuevo dataset (asegúrate de que tenga las mismas columnas o ajusta las listas de features).

Si el dataset no se genera sintéticamente, podrías necesitar modificar el código en src/banca_360_mlops/services/bank360_case.py (método build_dataset) para cargar desde CSV en lugar de generar sintéticamente. Por ejemplo, agregar una opción en settings.yaml para alternar entre dataset sintético o real.

2. Parámetros en conf/settings.yaml a Ajustar
Estos son los principales parámetros que dependen del dataset y el caso de negocio. Debes actualizarlos para que coincidan con las columnas y características de tu nuevo dataset:

case.dataset_rows: Cambia al número de filas de tu nuevo dataset (actualmente 1200).

case.target: La variable objetivo principal (actualmente "abandono"). Cámbiala si tu dataset 
tiene una variable de churn diferente.

case.ols_target: Variable objetivo para el modelo OLS de valor (actualmente "rentabilidad_mensual_estimada"). Ajusta si aplica a tu dataset.

case.feature_cols: Lista de columnas de features para el modelo de churn. Debes reemplazar con las columnas relevantes de tu dataset (ej. edad, ingresos, etc.).

case.outlier_cols: Columnas para detectar outliers. Actualiza con las numéricas de tu dataset que puedan tener outliers.

case.vif_cols: Columnas para análisis de multicolinealidad (VIF). Incluye las numéricas clave de tu dataset.

case.inference_features: Features para inferencia/predicción. Subconjunto de features esenciales.

case.value_model_features: Features para el modelo de valor/CLV. Lista completa de variables económicas y demográficas.

case.segment_features: Features para segmentación (ej. probabilidad de abandono, CLV). Ajusta si cambian las métricas.

case.test_size: Proporción de datos para test (actualmente 0.25). Ajusta si necesitas más/menos datos de entrenamiento.

case.clv_horizons: Horizontes para calcular CLV (actualmente [6, 12] meses). Cambia si tu horizonte de negocio es diferente.

case.financial_error_asymmetry: Asimetría en errores financieros (actualmente 1.8). Ajusta según el costo de falsos positivos/negativos en tu caso.

case.contact_cost: Costo de contacto por cliente (actualmente 22.0). Actualiza con tus costos reales.

case.retention_success_rate: Tasa de éxito de retención (actualmente 0.28). Basado en datos históricos de tu negocio.

case.max_contact_share: Máximo porcentaje de clientes a contactar (actualmente 0.30). Límite operativo.

fairness.sensitive_columns: Columnas sensibles para auditoría de equidad (actualmente ["edad", "genero", "region"]). Reemplaza con las de tu dataset (ej. si no hay "genero", quítalo).
fairness.age_bins: Bins para análisis de edad (actualmente [30, 45, 60]). Ajusta según la distribución de edad en tu dataset.

temporal_validation.n_splits: Número de splits para validación temporal (actualmente 4). Ajusta según la longitud de tu serie temporal.

temporal_validation.purge_gap_days y embargo_gap_days: Gaps para evitar data leakage (actualmente 7 y 14 días). Cambia según la frecuencia de tus datos.

seed: Semilla para reproducibilidad (actualmente 42). Cambia si quieres variabilidad.

3. ## Otros Ajustes Potenciales
Si tu dataset tiene columnas categóricas diferentes, asegúrate de que el código maneje encoding (el proyecto usa one-hot encoding automáticamente para categóricas).

Para deep learning o modelos avanzados, revisa deep_learning.enabled y subparámetros si no aplican.

Si el perfil operativo cambia (demo/qa/production), ajusta pipeline_health.profiles con thresholds apropiados para tu entorno.

Ejecuta el pipeline con python -m banca360v3 run --project-root . después de cambios, y valida con el notebook maestro para asegurar que todo funcione.

Si tu nuevo dataset difiere mucho en estructura (ej. columnas faltantes o tipos de datos), podrías necesitar modificar funciones en src/banca_360_mlops/core/ para adaptar la lógica de negocio. ¿Puedes describir más detalles sobre tu nuevo dataset (columnas, tamaño, dominio) para dar consejos más específicos?

4. ## NOTA IMPORTANTE:

El proyecto no detecta automáticamente las variables por tipo (como numéricas, categóricas, etc.) ni las asigna a categorías como churn, modelo, VIF, inferencia, multicolinealidad u outliers. Todas estas listas se definen manualmente en conf/settings.yaml basándose en conocimiento de dominio y un análisis exploratorio previo del dataset. El código simplemente lee estas configuraciones y las aplica directamente, sin inferencia automática.

Efectivamente, esto se debe a que la metodología del proyecto no está diseñada como una "caja negra" que adivina el propósito de los datos, sino como un sistema de gobernanza analítica que requiere el juicio experto del científico de datos antes de la ejecución técnica

Aquí te explico el porqué y el para qué de este enfoque, y cómo debes proceder:
1. El Porqué: Evitar la "Mecanización Ciega"
Confiar en que un software detecte automáticamente qué columnas usar para cada análisis (como multicolinealidad o inferencia) es peligroso. Las fuentes advierten que los procedimientos automatizados pueden seleccionar variables basándose solo en ruidos estadísticos o asociaciones azarosas (el "vast search effect"), ignorando la lógica funcional del negocio

Riesgo: Si el sistema eligiera automáticamente, podrías terminar con un modelo que parece preciso pero que es lógicamente incoherente o incapaz de generalizar ante nuevas realidades

2. El Para qué: El Contrato Metodológico
El proyecto utiliza un archivo de configuración llamado settings.yaml para establecer un contrato externo

Al definir tú mismo las columnas, aseguras que el pipeline respete la semántica del problema:
target: Identificas la variable que realmente representa el Churn (abandono)

vif_cols: Seleccionas qué variables compiten por estabilidad explicativa para evitar redundancias

outlier_cols: Decides en qué variables el negocio tolera o no valores extremos

inference_features: Defines qué drivers son éticamente aceptables o comercialmente lógicos para explicar el fenómeno

3. ¿Qué análisis previo debes realizar?
Antes de ejecutar el orquestador, debes asumir una postura detectivesca para completar el mapa de variables

Entender el Propósito: No te preguntes qué algoritmo usar, sino qué decisión quieres mejorar (ej. ¿a quién llamar para retención?)

Auditoría de Procedencia: Verificar si el dato tiene "basura" (principio Garbage In, Garbage Out). Si los datos de entrada son deficientes, ninguna "tortura estadística" salvará el modelo

Definición Semántica: Debes mapear tus columnas en los bloques de negocio (valor, vinculación, fricción, etc.) para que la explicabilidad posterior con SHAP tenga sentido para los directivos

Conclusión operativa
No necesitas realizar un análisis estadístico exhaustivo "a mano" antes de empezar, pero sí debes realizar una definición estratégica de roles.
Una vez que declares en el archivo settings.yaml qué columna es cada cual, el proyecto sí automatizará la validación técnica: detectará la Paradoja de Simpson, calculará el VIF, medirá el drift (PSI) y evaluará la salud del pipeline en estados allow, degraded o blocked

En resumen: tú pones el criterio humano, y el proyecto pone el rigor industrial

