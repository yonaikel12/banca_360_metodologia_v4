# Guia de archivos del proyecto Banca 360 Metodologia V4

<a id="inicio"></a>

## Objetivo del documento

Esta guia explica que hace cada archivo relevante del proyecto `banca_360_metodologia_v4`, como leerlo, como interpretarlo y que tipo de contenido guarda. La idea es que sirva como mapa operativo para tres perfiles distintos:

- quien quiere entender la metodologia de negocio;
- quien quiere modificar la configuracion o el codigo fuente;
- quien quiere auditar resultados, artefactos y trazabilidad.

## Indice interno

- [Objetivo del documento](#inicio)
- [Como recorrer el proyecto sin perderse](#recorrido)
- [Reglas de interpretacion rapida](#reglas)
- [Que no se detalla archivo por archivo](#exclusiones)
- [1. Archivos de la raiz del proyecto](#raiz)
- [2. Configuracion central](#configuracion-central)
- [3. Codigo fuente: paquete principal](#paquete-principal)
- [4. Codigo fuente: carpeta core](#core)
- [5. Codigo fuente: servicios, pipeline y utilidades](#servicios-pipeline-utils)
- [6. Notebook principal](#notebook-principal)
- [7. Datos base y evidencia persistida](#datos)
- [7.1 data/raw](#data-raw)
- [7.2 data/interim](#data-interim)
- [7.3 data/processed](#data-processed)
- [7.4 data/processed/figures](#data-processed-figures)
- [8. Tracking MLflow](#mlflow)
- [9. Metadatos de empaquetado generados](#packaging)
- [10. Como interpretar el proyecto como sistema](#sistema)
- [11. Atajos de lectura segun tu objetivo](#atajos)
- [12. Resumen ejecutivo de lectura](#resumen-ejecutivo)

<a id="recorrido"></a>

## Como recorrer el proyecto sin perderse

Orden de lectura recomendado:

1. [README.md](README.md): vision general, alcance y forma de ejecucion.
2. [conf/settings.yaml](conf/settings.yaml): contrato externo del caso.
3. [src/banca_360_mlops/cli.py](src/banca_360_mlops/cli.py): punto de entrada si ejecutas por consola.
4. [src/banca_360_mlops/config.py](src/banca_360_mlops/config.py): traduccion del YAML a objetos Python.
5. [src/banca_360_mlops/services/bank360_case.py](src/banca_360_mlops/services/bank360_case.py): capa de negocio que une dataset, benchmark, BI, metodologia y activacion.
6. [src/banca_360_mlops/pipeline/orchestrator.py](src/banca_360_mlops/pipeline/orchestrator.py): orden real de ejecucion del pipeline.
7. [src/banca_360_mlops/core/](src/banca_360_mlops/core/): funciones reutilizables de modelado, BI, limpieza, visualizacion y metodologia.
8. [notebooks/master_banca_360_mlops_orquestador.ipynb](notebooks/master_banca_360_mlops_orquestador.ipynb): vista narrativa y demostrativa del mismo flujo.
9. [data/raw](data/raw), [data/interim](data/interim), [data/processed](data/processed): evidencia persistida de lo que produce el pipeline.
10. [mlruns/](mlruns): tracking industrial en MLflow.

<a id="reglas"></a>

## Reglas de interpretacion rapida

- Archivos `.md`: explican decisiones, alcance o prompts de evolucion metodologica.
- Archivos `.yaml`: contrato declarativo; aqui no hay calculo, hay definicion de reglas.
- Archivos `.py`: implementacion ejecutable del framework.
- Archivos `.ipynb`: narrativa y ejecucion interactiva.
- Archivos `.csv`: evidencia tabular persistida para auditoria externa.
- Archivos `.json`: resumen estructurado o esquema de datos.
- Archivos `.png`: evidencia visual del pipeline.
- Archivos `.log`: traza textual de ejecucion.
- Archivos dentro de `mlruns/`: trazabilidad de parametros, metricas y artefactos por corrida.

<a id="exclusiones"></a>

## Que no se detalla archivo por archivo

No se documentan los `__pycache__/` ni los `.pyc` porque son caches generados por Python y no contienen logica editable. Si aparecen, no se leen como fuente de verdad.

---

<a id="raiz"></a>

## 1. Archivos de la raiz del proyecto

| Archivo | Para que sirve | Que contiene | Como leerlo e interpretarlo |
| --- | --- | --- | --- |
| [README.md](README.md) | Presentacion oficial del proyecto v4. | Objetivo, stack, propuesta de valor, forma de instalacion y notas de uso. | Leelo como resumen ejecutivo y guia de arranque. Si algo contradice al codigo, el codigo manda y el README debe actualizarse. |
| [pyproject.toml](pyproject.toml) | Metadatos de empaquetado del proyecto. | Nombre del paquete, version `0.4.0`, backend de build, busqueda de paquetes y script de consola `banca360v4`. | Interpretalo como el manifiesto para instalar el paquete. Si cambia la CLI, este archivo debe mantenerse alineado. |
| [requirements.txt](requirements.txt) | Dependencias de ejecucion. | Librerias base y opcionales: pandas, scikit-learn, shap, mlflow, xgboost, lightgbm, catboost, prophet, pygam, tensorflow, etc. | Leelo como contrato de entorno. Si una funcionalidad opcional falla por dependencia, empieza aqui. |
| [prompt.md](prompt.md) | Prompt historico para evolucionar la metodologia. | Instrucciones para construir un notebook autonomo de contrato externo y parametrizacion. | No es codigo de produccion. Interpretalo como especificacion de trabajo para agentes o desarrollo asistido. |
| [analisis_contrato_externo.md](analisis_contrato_externo.md) | Marco conceptual previo al `settings.yaml`. | Explica por que el contrato metodologico debe construirse con postura detectivesca, rigor estadistico y criterio de negocio. | Leelo antes de tocar [conf/settings.yaml](conf/settings.yaml). Es la justificacion metodologica de los parametros. |
| [diccionario de parametros.txt](diccionario%20de%20parametros.txt) | Guia narrativa de parametros. | Explicacion conceptual y operativa de variables de configuracion como `dataset_rows`, `target`, `vif_cols`, `financial_error_asymmetry`, etc. | Sirve para entender la semantica de cada bloque del YAML. Es una guia humana, no un archivo ejecutable. |
| [settings_parameter_guidelines.ipynb](settings_parameter_guidelines.ipynb) | Soporte interactivo para parametrizacion. | Notebook orientado a analizar o justificar parametros del contrato externo. | Leelo como complemento didactico del YAML; no sustituye al contrato ni al pipeline principal. |
| [GUIA_ARCHIVOS_PROYECTO_V4.md](GUIA_ARCHIVOS_PROYECTO_V4.md) | Este documento. | Mapa detallado del proyecto y de sus salidas. | Utilizalo como indice maestro para navegar el repo. |

---

<a id="configuracion-central"></a>

## 2. Configuracion central

### [conf/settings.yaml](conf/settings.yaml)

Archivo mas importante del proyecto desde la perspectiva operativa.

### Para que sirve

Define el contrato externo del caso de negocio. Controla:

- semilla del proyecto;
- tracking MLflow;
- target y variables;
- benchmark activo;
- ruteo por caso de negocio;
- parametros de forecasting;
- fairness;
- health gating;
- parsimonia, incertidumbre y deep learning governance.

### Que contiene

Bloques principales:

- `seed`
- `tracking`
- `case`
- `case.model_routing`
- `case.forecasting`
- `case.parsimony`
- `case.uncertainty`
- `case.temporal_validation`
- `case.fairness`
- `case.bandit`
- `case.deep_learning`
- `case.pipeline_health`

### Como leerlo

Leelo de arriba hacia abajo como si fuera una declaracion de negocio que luego el codigo convierte en ejecucion. Lo mas importante es no leerlo como "config tecnica suelta", sino como un contrato metodologico:

- `target` y `ols_target` dicen que decision se quiere mejorar;
- `feature_cols`, `inference_features` y `value_model_features` asignan roles semanticos a las columnas;
- `model_routing` decide que familia de benchmark se activara;
- `forecasting` solo cobra sentido si el `problem_type` es temporal;
- `pipeline_health` define hasta que punto una corrida se permite, se degrada o se bloquea.

### Como interpretarlo

Si el pipeline da resultados inesperados, este archivo es el primer lugar donde buscar. La mayoria de los cambios de comportamiento del proyecto no se hacen tocando el core, sino ajustando este contrato.

---

<a id="paquete-principal"></a>

## 3. Codigo fuente: paquete principal

Carpeta base: `src/banca_360_mlops/`

### [src/__init__.py](src/__init__.py)

Archivo vacio o minimo de paquete raiz bajo `src/`.

- Sirve para que `src` sea tratado correctamente en el empaquetado.
- No suele tener logica funcional.
- Se interpreta como soporte de layout del proyecto.

### [src/banca_360_mlops/__init__.py](src/banca_360_mlops/__init__.py)

### Para que sirve

Define la identidad publica minima del paquete.

### Que contiene

- forzado de backend `Agg` en matplotlib para ejecuciones headless;
- export de `ProjectConfig` y `load_project_config`.

### Como leerlo

Es un archivo pequeno, pero importante: deja claro que el paquete esta preparado para correr tanto en CLI como en entornos sin interfaz grafica.

### [src/banca_360_mlops/cli.py](src/banca_360_mlops/cli.py)

### Para que sirve

Es la puerta de entrada por consola del proyecto.

### Que contiene

- parser de argumentos;
- resolucion del `project_root`;
- resolucion del `log_path`;
- redireccion nativa de `stdout` y `stderr` hacia log persistido;
- llamada a `run_pipeline`;
- impresion del resumen ejecutivo de corrida.

### Como leerlo

Leelo como el borde externo del sistema. Si quieres saber como lanzar el pipeline sin notebook, este es el archivo. Si quieres saber por que se genero un `.log`, tambien.

### Como interpretarlo

No contiene la metodologia en si; contiene la operacion de la metodologia.

### [src/banca_360_mlops/config.py](src/banca_360_mlops/config.py)

### Para que sirve

Traduce `conf/settings.yaml` a dataclasses tipadas.

### Que contiene

- `TrackingConfig`
- `PipelineHealthConfig`
- `ParsimonyConfig`
- `UncertaintyConfig`
- `TemporalValidationConfig`
- `FairnessConfig`
- `BanditConfig`
- `DeepLearningConfig`
- `ModelRoutingConfig`
- `ForecastingConfig`
- `CaseConfig`
- `ProjectConfig`
- `load_project_config()`

### Como leerlo

Si dudas de como un parametro YAML se convierte en comportamiento real, este archivo responde la pregunta. Leelo comparando el YAML y la dataclass correspondiente.

### Como interpretarlo

Es el adaptador entre declaracion y ejecucion. Si el YAML es el contrato, este archivo es el traductor legal.

### [src/banca_360_mlops/io.py](src/banca_360_mlops/io.py)

### Para que sirve

Persistencia ligera de tablas, figuras y JSON.

### Que contiene

- creacion de carpetas runtime con `ensure_runtime_layout`;
- conversion segura a JSON con `_json_safe`;
- guardado de CSV, JSON y PNG.

### Como leerlo

Leelo cuando necesites entender por que una salida aparece en `data/raw`, `data/interim`, `data/processed` o `figures`.

### [src/banca_360_mlops/tracking.py](src/banca_360_mlops/tracking.py)

### Para que sirve

Envolver MLflow con degradacion segura.

### Que contiene

- clase `ExperimentTracker`;
- apertura de runs;
- logging de parametros, metricas y artefactos.

### Como leerlo

Leelo cuando quieras saber por que una corrida deja rastro en `mlruns/` y que pasa si MLflow no esta instalado.

---

<a id="core"></a>

## 4. Codigo fuente: carpeta `core`

Esta es la libreria metodologica reusable. Contiene la mayor parte del conocimiento tecnico del proyecto.

### [src/banca_360_mlops/core/configuracion.py](src/banca_360_mlops/core/configuracion.py)

- Sirve para aplicar tema visual y configuracion estetica transversal.
- Contiene helpers de estilo para tablas o figuras.
- Leelo cuando quieras entender decisiones visuales compartidas por notebooks y funciones.

### [src/banca_360_mlops/core/datasets_sinteticos.py](src/banca_360_mlops/core/datasets_sinteticos.py)

- Sirve para construir datasets sinteticos del caso cuando no se usan datos reales.
- Contiene generadores de tablas demo con columnas alineadas al contrato del proyecto.
- Interpretalo como el origen de los datos de ejemplo sobre los que se demuestra la metodologia.

### [src/banca_360_mlops/core/exploracion.py](src/banca_360_mlops/core/exploracion.py)

- Sirve para resumenes numericos y categoricos.
- Contiene funciones de EDA reusable.
- Leelo si quieres saber de donde salen tablas descriptivas y lecturas iniciales del dataset.

### [src/banca_360_mlops/core/framework_bi_universal.py](src/banca_360_mlops/core/framework_bi_universal.py)

### Para que sirve

Implementa la capa BI universal del caso.

### Que contiene

- manual metodologico BI;
- generacion del caso bancario sintetico;
- benchmark de valor;
- pipeline BI reusable;
- scorecard de retencion;
- dashboard;
- CLV probabilistico;
- umbral economico de activacion;
- explicabilidad SHAP.

### Como leerlo

Leelo si tu pregunta es: "como se traduce el modelo a una salida operativa entendible para negocio?".

### Como interpretarlo

Este archivo no es solo modelado; es la capa que convierte un score en decision comercial, dashboard y lectura ejecutiva.

### [src/banca_360_mlops/core/limpieza.py](src/banca_360_mlops/core/limpieza.py)

- Sirve para calidad de datos, normalizacion de columnas, nulos y outliers.
- Contiene funciones de sanitizacion y chequeos previos al modelado.
- Leelo antes de tocar transformaciones de entrada o reglas de calidad.

### [src/banca_360_mlops/core/metodologia.py](src/banca_360_mlops/core/metodologia.py)

### Para que sirve

Es el corazon metodologico de v4.

### Que contiene

- taxonomia metodologica del framework;
- catalogo universal de modelos;
- auditoria estadistica;
- benchmark supervisado;
- parsimonia logistica;
- fairness audit;
- validacion temporal con purga y embargo;
- bootstrap de incertidumbre;
- calibracion;
- analisis multiverse;
- soporte a algoritmos extendidos como `lightgbm`, `xgboost`, `catboost`, `gam`, `mars`, `mlp`, `arima`, `prophet`, `lstm`;
- manejo de dependencias opcionales;
- mitigacion de warnings de LightGBM mediante salida pandas en el preprocesamiento;
- fallback de MARS con `SplineTransformer` cuando `py-earth` no esta disponible.

### Como leerlo

No intentes leerlo linealmente de principio a fin la primera vez. Hazlo por capas:

1. constantes y catalogos;
2. helpers internos;
3. construccion de preprocesamiento;
4. construccion y entrenamiento de modelos;
5. validaciones y reportes.

### Como interpretarlo

Si el proyecto tiene "cerebro metodologico", esta aqui.

### [src/banca_360_mlops/core/plantilla_pipeline_ciencia_datos.py](src/banca_360_mlops/core/plantilla_pipeline_ciencia_datos.py)

- Sirve para ejecutar una plantilla metodologica reusable de ciencia de datos.
- Contiene una orquestacion mas general del pipeline metodologico universal.
- Leelo cuando quieras reutilizar la metodologia fuera del caso bancario exacto.

### [src/banca_360_mlops/core/segmentacion_nba.py](src/banca_360_mlops/core/segmentacion_nba.py)

- Sirve para segmentacion KMeans, perfilado y next-best-action.
- Contiene funciones de evaluacion de k, construccion de perfiles y dashboard de segmentacion.
- Interpretalo como la capa de activacion comercial post-modelo.

### [src/banca_360_mlops/core/visualizacion.py](src/banca_360_mlops/core/visualizacion.py)

- Sirve para graficos y salidas visuales comunes.
- Contiene funciones auxiliares de graficacion usadas por otros modulos.
- Leelo si quieres cambiar estilo o tipo de figura persistida.

### [src/banca_360_mlops/core/__init__.py](src/banca_360_mlops/core/__init__.py)

- Marca la carpeta como subpaquete Python.
- Puede ser minimo o vacio.
- No es fuente de logica principal.

---

<a id="servicios-pipeline-utils"></a>

## 5. Codigo fuente: servicios, pipeline y utilidades

### [src/banca_360_mlops/services/bank360_case.py](src/banca_360_mlops/services/bank360_case.py)

### Para que sirve

Es la capa de negocio del caso Banca 360.

### Que contiene

- construccion del contrato del caso;
- glosario de columnas;
- referencias metodologicas heredadas;
- generacion y auditoria del dataset;
- benchmark de modelos;
- capa BI;
- validacion metodologica;
- SHAP;
- activacion CLV y segmentacion;
- resumen final de ejecucion.

### Como leerlo

Leelo como la "historia" del caso. Si `metodologia.py` aporta piezas, este archivo decide como se ensamblan para el dominio bancario concreto.

### Como interpretarlo

Es el mejor archivo para entender el negocio implementado, no solo la tecnica.

### [src/banca_360_mlops/services/__init__.py](src/banca_360_mlops/services/__init__.py)

- Archivo de paquete.
- Soporte de importacion.
- Sin lectura metodologica profunda.

### [src/banca_360_mlops/pipeline/nodes.py](src/banca_360_mlops/pipeline/nodes.py)

### Para que sirve

Define nodos asincronos para ejecutar cada etapa en threads separados.

### Que contiene

- `build_context_node`
- `build_dataset_node`
- `benchmark_node`
- `bi_layer_node`
- `methodology_node`
- `shap_node`
- `activation_node`

### Como leerlo

Leelo como un mapa de etapas, no como lugar de calculo profundo. Cada nodo delega en el servicio.

### [src/banca_360_mlops/pipeline/orchestrator.py](src/banca_360_mlops/pipeline/orchestrator.py)

### Para que sirve

Es el orden maestro de ejecucion del pipeline.

### Que contiene

- clase `Bank360PipelineOrchestrator`;
- `run()` asincrono;
- persistencia de artefactos;
- tracking de parametros y metricas;
- punto de entrada sincronico `run_pipeline()`.

### Como leerlo

Si quieres entender el flujo exacto de extremo a extremo y en que momento se persiste cada archivo de `data/`, este es el archivo correcto.

### Como interpretarlo

Es el director de orquesta. No genera toda la logica, pero decide la secuencia y la persistencia.

### [src/banca_360_mlops/pipeline/__init__.py](src/banca_360_mlops/pipeline/__init__.py)

- Archivo de paquete para la capa pipeline.
- Soporte de importaciones.

### [src/banca_360_mlops/utils/reproducibility.py](src/banca_360_mlops/utils/reproducibility.py)

- Sirve para fijar semillas de Python y NumPy.
- Contiene `set_global_seed`.
- Se interpreta como garantia de reproducibilidad basica.

### [src/banca_360_mlops/utils/__init__.py](src/banca_360_mlops/utils/__init__.py)

- Archivo de paquete para utilidades.
- Soporte estructural.

---

<a id="notebook-principal"></a>

## 6. Notebook principal

### [notebooks/master_banca_360_mlops_orquestador.ipynb](notebooks/master_banca_360_mlops_orquestador.ipynb)

### Para que sirve

Es la vista narrativa e interactiva del pipeline v4.

### Que contiene

Segun el resumen actual del notebook, alterna celdas Markdown y celdas Python para:

- presentar el caso;
- cargar configuracion y objetos del paquete;
- ejecutar el contexto metodologico;
- inspeccionar dataset y auditorias;
- revisar benchmark, BI, metodologia, SHAP y activacion.

### Como leerlo

Leelo como demo guiada. Primero las celdas Markdown para contexto y luego las celdas de codigo para ver como se consumen los modulos del paquete.

### Como interpretarlo

No es la fuente primaria de logica. Es la mejor superficie para explicar y demostrar el pipeline ante negocio, aula o comite.

---

<a id="datos"></a>

## 7. Datos base y evidencia persistida

<a id="data-raw"></a>

### 7.1 [data/raw/](data/raw)

Estos archivos contienen datos base o referencias regeneradas por el pipeline. Se leen como evidencia de entrada y contexto, no como salida final para decision.

| Archivo | Para que sirve | Que contiene | Como interpretarlo |
| --- | --- | --- | --- |
| [data/raw/bank360_dataset.csv](data/raw/bank360_dataset.csv) | Dataset sintetico principal del caso. | Filas de clientes y variables del modelo. | Es la materia prima del pipeline. Si cambia su estructura, el contrato puede romperse. |
| [data/raw/bank360_data_dictionary.csv](data/raw/bank360_data_dictionary.csv) | Diccionario de datos procesable. | Variables, tipos, roles semanticos y descripciones. | Sirve para auditar gobernanza semantica. |
| [data/raw/bank360_data_dictionary_metadata.csv](data/raw/bank360_data_dictionary_metadata.csv) | Metadata auxiliar del diccionario. | Informacion de ownership, fuente, reglas o esquema resumido. | Complementa el diccionario principal. |
| [data/raw/bank360_data_dictionary_schema.json](data/raw/bank360_data_dictionary_schema.json) | Esquema serializado del dataset. | Estructura legible por maquina del contrato tabular. | Si quieres validar campos de forma programatica, empieza aqui. |
| [data/raw/manual_bi_resumen.csv](data/raw/manual_bi_resumen.csv) | Resumen del manual BI. | Tabla sintetica de fases, objetivos y estandares. | Sirve para auditoria documental de la capa BI. |
| [data/raw/guia_metricas.csv](data/raw/guia_metricas.csv) | Guia de traduccion de metricas. | Equivalencias entre metricas tecnicas y lectura de negocio. | Util para comites no tecnicos. |
| [data/raw/metodologia_v3_resumen.csv](data/raw/metodologia_v3_resumen.csv) | Resumen de herencia metodologica v3. | Pilares heredados: parsimonia, incertidumbre, temporalidad, consenso, fairness y DL. | Permite ver que v4 no nace desde cero sino como evolucion. |
| [data/raw/alineacion_metodologica_pdf.csv](data/raw/alineacion_metodologica_pdf.csv) | Alineacion con principios metodologicos externos. | Traduccion de principios teoricos a componentes del proyecto. | Es evidencia de coherencia entre framework y implementacion. |

<a id="data-interim"></a>

### 7.2 [data/interim/](data/interim)

Aqui viven artefactos intermedios del pipeline. Son auditorias y reportes tecnicos previos a la activacion de negocio.

| Archivo | Para que sirve | Que contiene | Como interpretarlo |
| --- | --- | --- | --- |
| [data/interim/benchmark_modelos.csv](data/interim/benchmark_modelos.csv) | Comparativa de modelos benchmark. | Algoritmo, metricas, complejidad, ranking y elegibilidad. | Es la tabla para justificar el champion model. |
| [data/interim/benchmark_parsimonia.csv](data/interim/benchmark_parsimonia.csv) | Estudio de parsimonia. | Comparacion de especificaciones simples vs complejas. | Si el mejor modelo no es el mas complejo, aqui deberia verse. |
| [data/interim/bi_conclusiones.csv](data/interim/bi_conclusiones.csv) | Resumen BI intermedio. | Lecturas ejecutivas, hallazgos y conclusiones del pipeline BI. | Es una traduccion de tecnicismo a accion. |
| [data/interim/column_name_audit.csv](data/interim/column_name_audit.csv) | Auditoria de nombres de columnas. | Cumplimiento de estandar tabular y convenciones. | Sirve para detectar deuda de gobernanza basica. |
| [data/interim/deep_learning_governance.csv](data/interim/deep_learning_governance.csv) | Checklist de gobernanza DL. | Requisitos de dropout, batch norm, early stopping, etc. | Si se activa MLP o LSTM, este archivo gana peso. |
| [data/interim/fairness_group_metrics.csv](data/interim/fairness_group_metrics.csv) | Metricas por grupo sensible. | Desempeno y brechas por edad, genero o region. | Se usa para diagnostico detallado de fairness. |
| [data/interim/fairness_summary.csv](data/interim/fairness_summary.csv) | Resumen del fairness audit. | Brechas agregadas y semaforos eticos. | Es la vista ejecutiva de equidad. |
| [data/interim/methodology_tabular_summary.csv](data/interim/methodology_tabular_summary.csv) | Resumen tabular metodologico. | Controles tabulares y checks del framework. | Sirve como evidencia de gobierno del dato. |
| [data/interim/riesgos_metodologicos.csv](data/interim/riesgos_metodologicos.csv) | Semaforo de riesgos. | Drift, estabilidad, fairness, salud y otros flags. | Es la tabla para leer el estado metodologico del pipeline. |
| [data/interim/sampling_plan.csv](data/interim/sampling_plan.csv) | Plan de muestreo. | Recomendaciones o diagnosticos de representatividad. | Importante si el caso se quiere defender estadisticamente. |
| [data/interim/segment_distribution_audit.csv](data/interim/segment_distribution_audit.csv) | Distribucion por segmentos. | Cobertura y equilibrio segmental. | Complementa muestreo y representatividad. |
| [data/interim/tabular_contract_summary.csv](data/interim/tabular_contract_summary.csv) | Resumen del contrato tabular. | Reglas y cumplimiento de estructura de dataset. | Util para QA de ingestion. |
| [data/interim/temporal_validation_folds.csv](data/interim/temporal_validation_folds.csv) | Detalle de folds temporales. | Fechas, ventanas y bloques de purga/embargo. | Si dudas de leakage temporal, revisa este archivo. |
| [data/interim/temporal_validation_summary.csv](data/interim/temporal_validation_summary.csv) | Resumen de validacion temporal. | Metricas agregadas por esquema temporal. | Sirve para evaluar estabilidad fuera de muestra. |
| [data/interim/uncertainty_portfolio_summary.csv](data/interim/uncertainty_portfolio_summary.csv) | Resumen de incertidumbre. | Dispersion, intervalos o amplitud de pronostico agregada. | Importante para no sobrerreaccionar a scores puntuales. |

<a id="data-processed"></a>

### 7.3 [data/processed/](data/processed)

Aqui estan las salidas listas para consumo operativo o ejecutivo.

| Archivo | Para que sirve | Que contiene | Como interpretarlo |
| --- | --- | --- | --- |
| [data/processed/execution_summary.json](data/processed/execution_summary.json) | Resumen maestro de la corrida. | Champion model, metricas clave, decision de salud, ROI, segmentos y lectura ejecutiva. | Es el archivo mas util para saber "que paso" en una corrida sin abrir 20 tablas. |
| [data/processed/scorecard_retencion.csv](data/processed/scorecard_retencion.csv) | Scorecard operativo. | Priorizacion de clientes con score, semaforo y lectura operativa. | Es el puente directo entre modelo y accion comercial. |
| [data/processed/shortlist_retencion.csv](data/processed/shortlist_retencion.csv) | Lista priorizada para accion. | Subconjunto accionable de clientes a contactar. | Se interpreta como salida lista para campaña. |
| [data/processed/perfil_segmentos.csv](data/processed/perfil_segmentos.csv) | Perfil de segmentos. | Rasgos descriptivos por cluster o segmento. | Sirve para entender "quien es quien". |
| [data/processed/resumen_segmentos.csv](data/processed/resumen_segmentos.csv) | Resumen agregado de segmentos. | KPIs resumidos por segmento. | Util para lectura gerencial de segmentacion. |
| [data/processed/playbook_segmentos.csv](data/processed/playbook_segmentos.csv) | Recomendaciones por segmento. | Next-best-action o accion sugerida segun perfil. | Es una salida de activacion comercial. |
| [data/processed/brecha_valor_consenso.csv](data/processed/brecha_valor_consenso.csv) | Brecha contra consenso. | Diferencia entre valor individual y valor esperado por grupo. | Ayuda a detectar donde la accion individual supera la media del segmento. |
| [data/processed/bandit_policy.csv](data/processed/bandit_policy.csv) | Politica bandit operativa. | Recomendacion de brazo o accion bajo logica de exploracion-explotacion. | Sirve para operacion adaptativa, no solo scoring estatico. |
| [data/processed/shap_summary.csv](data/processed/shap_summary.csv) | Resumen global SHAP. | Importancia y contribucion agregada de features. | Se interpreta como transparencia global del modelo. |
| [data/processed/shap_lectura_local.csv](data/processed/shap_lectura_local.csv) | Lectura local SHAP. | Explicaciones por registro o muestra local. | Sirve para justificar casos individuales. |
| [data/processed/pipeline_run_20260405_v4.log](data/processed/pipeline_run_20260405_v4.log) | Log historico manual de una corrida. | Trazas textuales del pipeline. | Evidencia de una ejecucion validada. |
| [data/processed/pipeline_run_native_cli_smoke.log](data/processed/pipeline_run_native_cli_smoke.log) | Log de smoke test de CLI. | Prueba controlada del mecanismo de logging nativo. | No es el log principal de negocio; es evidencia de validacion tecnica. |
| [data/processed/pipeline_run_native_cli_v4.log](data/processed/pipeline_run_native_cli_v4.log) | Log nativo de corrida real. | Salida persistida de la CLI v4. | Es la referencia correcta para auditoria de ejecucion por consola. |

<a id="data-processed-figures"></a>

### 7.4 [data/processed/figures/](data/processed/figures)

Graficos persistidos del pipeline.

| Archivo | Para que sirve | Que contiene | Como interpretarlo |
| --- | --- | --- | --- |
| [data/processed/figures/calibracion.png](data/processed/figures/calibracion.png) | Visual de calibracion probabilistica. | Curva o diagrama de calibracion. | Sirve para ver si la probabilidad predicha es creible. |
| [data/processed/figures/dashboard_retencion.png](data/processed/figures/dashboard_retencion.png) | Dashboard de retencion. | Visual ejecutivo del caso churn. | Es presentable a negocio. |
| [data/processed/figures/dashboard_segmentacion.png](data/processed/figures/dashboard_segmentacion.png) | Dashboard de segmentos. | Resumen visual de clustering y perfiles. | Util para activacion comercial y storytelling. |
| [data/processed/figures/diagnostico_ols.png](data/processed/figures/diagnostico_ols.png) | Diagnostico OLS. | Residuos, influencia u otras pruebas visuales. | Importante para lectura econometrica. |
| [data/processed/figures/shap_dependence.png](data/processed/figures/shap_dependence.png) | Dependencia SHAP. | Efecto de una variable sobre prediccion. | Sirve para interpretar no linealidades o interacciones. |
| [data/processed/figures/shap_summary.png](data/processed/figures/shap_summary.png) | Beeswarm o resumen SHAP global. | Distribucion de impactos de features. | Es la vista estandar de interpretabilidad global. |
| [data/processed/figures/tradeoff_umbral.png](data/processed/figures/tradeoff_umbral.png) | Tradeoff de umbral. | Relacion entre captura, costo, valor o precision. | Se usa para justificar el threshold operativo. |

---

<a id="mlflow"></a>

## 8. Tracking MLflow: carpeta [mlruns/](mlruns)

La estructura observada incluye un experimento con id `412889489156128449` y varias corridas. Cada corrida repite una estructura casi identica.

### Archivos de nivel experimento

| Ruta patron | Para que sirve | Que contiene | Como interpretarlo |
| --- | --- | --- | --- |
| [mlruns/412889489156128449/meta.yaml](mlruns/412889489156128449/meta.yaml) | Metadata del experimento MLflow. | Nombre, ubicacion y atributos del experimento. | Sirve para saber que conjunto de corridas se esta agrupando. |

### Archivos de nivel corrida

Ejemplos de `run_id` observados: `3402bc49dd874ec39876f9a9b48be9da`, `382da0d580d64ba693e9869d01441424`, `3e9734b8b9bf41d79760be2f43c86070`.

| Ruta patron | Para que sirve | Que contiene | Como interpretarlo |
| --- | --- | --- | --- |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/meta.yaml](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/meta.yaml) | Metadata de una corrida. | Tiempos, estado y datos de la run. | Es el encabezado tecnico de la ejecucion. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/dataset_rows](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/dataset_rows) | Parametro de dataset. | Valor usado en esa corrida. | Sirve para reproducibilidad. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/seed](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/seed) | Parametro de semilla. | Semilla usada. | Ayuda a replicar resultados. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/selected_feature_count](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/selected_feature_count) | Parametro derivado. | Numero de variables seleccionadas. | Resume parsimonia observada. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/target](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/target) | Parametro de negocio. | Target configurado. | Confirma identidad del problema modelado. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/test_size](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/params/test_size) | Parametro de particion. | Fraccion de test. | Afecta validacion y lectura del benchmark. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/benchmark_primary_metric](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/benchmark_primary_metric) | Metrica principal benchmark. | Valor de desempeno principal. | Sirve para comparar corridas. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/benchmark_secondary_metric](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/benchmark_secondary_metric) | Metrica secundaria benchmark. | Segunda metrica clave. | Complementa la principal. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/bi_log_loss](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/bi_log_loss) | Metrica BI. | Log loss del modelo BI. | Penaliza mala calibracion/confianza excesiva. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/bi_roc_auc](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/bi_roc_auc) | Metrica BI. | ROC AUC del pipeline BI. | Mide ranking global. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/bootstrap_interval_width_mean](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/bootstrap_interval_width_mean) | Incertidumbre. | Anchura media de intervalos. | Cuanto mas grande, mas incertidumbre. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/brier_score](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/brier_score) | Calibracion. | Error cuadratico probabilistico. | Menor suele ser mejor. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/fairness_max_gap](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/fairness_max_gap) | Equidad. | Mayor brecha detectada entre grupos. | Cuanto mas alta, mayor alerta etica. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/roi_estimado](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/roi_estimado) | Negocio. | ROI modelado de la estrategia. | Metrica clave para activacion. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/temporal_cv_roc_auc](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/temporal_cv_roc_auc) | Robustez temporal. | ROC AUC temporal purgado. | Si cae mucho, el modelo es menos estable que en holdout simple. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/threshold](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/threshold) | Decision operativa. | Umbral seleccionado. | Determina priorizacion comercial. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/valor_esperado_neto](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/metrics/valor_esperado_neto) | Negocio. | Valor neto esperado. | Resume impacto economico esperado. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.runName](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.runName) | Tag interno MLflow. | Nombre de la corrida. | Facilita localizar corridas. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.source.name](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.source.name) | Tag interno MLflow. | Origen del proceso. | Ayuda a auditar fuente de ejecucion. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.source.type](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.source.type) | Tag interno MLflow. | Tipo de fuente o launcher. | Trazabilidad tecnica. |
| [mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.user](mlruns/412889489156128449/3402bc49dd874ec39876f9a9b48be9da/tags/mlflow.user) | Tag interno MLflow. | Usuario de la corrida. | Auditoria basica. |

### Carpeta `artifacts/` de cada corrida

Dentro de `mlruns/<experiment_id>/<run_id>/artifacts/` se replican muchos archivos de `data/processed/` y, segun la corrida, algunos logs. Interpretacion:

- no son una nueva fuente de verdad distinta;
- son una copia versionada por corrida para trazabilidad en MLflow;
- si quieres el artefacto mas reciente y directo, suele ser mas comodo leer `data/processed/`;
- si quieres saber exactamente que vio una corrida pasada, mira `mlruns/`.

---

<a id="packaging"></a>

## 9. Metadatos de empaquetado generados

Carpeta: [src/banca_360_mlops.egg-info/](src/banca_360_mlops.egg-info)

Estos archivos aparecen tras instalar el paquete en modo editable o construir metadata local.

| Archivo | Para que sirve | Que contiene | Como interpretarlo |
| --- | --- | --- | --- |
| [src/banca_360_mlops.egg-info/dependency_links.txt](src/banca_360_mlops.egg-info/dependency_links.txt) | Metadata setuptools. | Enlaces especiales de dependencias si existieran. | Normalmente es informativo y pequeno. |
| [src/banca_360_mlops.egg-info/entry_points.txt](src/banca_360_mlops.egg-info/entry_points.txt) | Registro de scripts. | Declaracion del entry point `banca360v4 = banca_360_mlops.cli:main`. | Si la CLI no se expone bien, revisa este archivo junto con [pyproject.toml](pyproject.toml). |
| [src/banca_360_mlops.egg-info/PKG-INFO](src/banca_360_mlops.egg-info/PKG-INFO) | Metadata del paquete. | Nombre, version, descripcion y otros campos del proyecto. | Es la ficha tecnica instalada del paquete. |
| [src/banca_360_mlops.egg-info/SOURCES.txt](src/banca_360_mlops.egg-info/SOURCES.txt) | Lista de fuentes incluidas. | Archivos contemplados por el empaquetado. | Util para diagnosticar packaging. |
| [src/banca_360_mlops.egg-info/top_level.txt](src/banca_360_mlops.egg-info/top_level.txt) | Paquetes top-level. | Nombre raiz importable del proyecto. | Ayuda a confirmar el nombre del modulo instalado. |

---

<a id="sistema"></a>

## 10. Como interpretar el proyecto como sistema

Piensa el repo en 5 capas:

1. **Contrato**: [conf/settings.yaml](conf/settings.yaml)
   Aqui se decide que problema se resuelve y bajo que reglas.

2. **Motor metodologico**: [src/banca_360_mlops/core/](src/banca_360_mlops/core/)
   Aqui viven las funciones que auditan, modelan, validan y explican.

3. **Traduccion al negocio**: [src/banca_360_mlops/services/bank360_case.py](src/banca_360_mlops/services/bank360_case.py)
   Aqui se convierte el framework general en un caso bancario concreto.

4. **Orquestacion y operacion**: [src/banca_360_mlops/cli.py](src/banca_360_mlops/cli.py), [src/banca_360_mlops/pipeline/nodes.py](src/banca_360_mlops/pipeline/nodes.py), [src/banca_360_mlops/pipeline/orchestrator.py](src/banca_360_mlops/pipeline/orchestrator.py), [src/banca_360_mlops/tracking.py](src/banca_360_mlops/tracking.py), [src/banca_360_mlops/io.py](src/banca_360_mlops/io.py)
   Aqui se ejecuta, persiste y traza todo.

5. **Evidencia**: [data/](data), [notebooks/](notebooks), [mlruns/](mlruns)
   Aqui queda la prueba observable de que el sistema corrio y que produjo.

---

<a id="atajos"></a>

## 11. Atajos de lectura segun tu objetivo

### Si quieres modificar el caso de negocio

Lee primero:

1. [conf/settings.yaml](conf/settings.yaml)
2. [src/banca_360_mlops/services/bank360_case.py](src/banca_360_mlops/services/bank360_case.py)
3. [src/banca_360_mlops/core/framework_bi_universal.py](src/banca_360_mlops/core/framework_bi_universal.py)
4. [src/banca_360_mlops/core/metodologia.py](src/banca_360_mlops/core/metodologia.py)

### Si quieres cambiar el orden del pipeline

Lee primero:

1. [src/banca_360_mlops/cli.py](src/banca_360_mlops/cli.py)
2. [src/banca_360_mlops/pipeline/nodes.py](src/banca_360_mlops/pipeline/nodes.py)
3. [src/banca_360_mlops/pipeline/orchestrator.py](src/banca_360_mlops/pipeline/orchestrator.py)

### Si quieres auditar resultados

Lee primero:

1. [data/processed/execution_summary.json](data/processed/execution_summary.json)
2. [data/interim/benchmark_modelos.csv](data/interim/benchmark_modelos.csv)
3. [data/interim/riesgos_metodologicos.csv](data/interim/riesgos_metodologicos.csv)
4. [data/processed/scorecard_retencion.csv](data/processed/scorecard_retencion.csv)
5. [data/processed/pipeline_run_native_cli_v4.log](data/processed/pipeline_run_native_cli_v4.log)
6. [mlruns/](mlruns)

### Si quieres explicar el proyecto a un tercero

Empieza por:

1. [README.md](README.md)
2. [notebooks/master_banca_360_mlops_orquestador.ipynb](notebooks/master_banca_360_mlops_orquestador.ipynb)
3. [data/processed/execution_summary.json](data/processed/execution_summary.json)
4. [GUIA_ARCHIVOS_PROYECTO_V4.md](GUIA_ARCHIVOS_PROYECTO_V4.md)

---

<a id="resumen-ejecutivo"></a>

## 12. Resumen ejecutivo de lectura

Si solo tuvieras 10 minutos para entender el proyecto, la lectura minima seria esta:

1. [README.md](README.md) para saber que problema resuelve.
2. [conf/settings.yaml](conf/settings.yaml) para saber con que contrato corre.
3. [src/banca_360_mlops/services/bank360_case.py](src/banca_360_mlops/services/bank360_case.py) para saber como se traduce a negocio.
4. [src/banca_360_mlops/pipeline/orchestrator.py](src/banca_360_mlops/pipeline/orchestrator.py) para saber el orden de ejecucion.
5. [data/processed/execution_summary.json](data/processed/execution_summary.json) para saber que produjo.

Con eso ya puedes ubicar el resto de archivos sin perder el hilo.