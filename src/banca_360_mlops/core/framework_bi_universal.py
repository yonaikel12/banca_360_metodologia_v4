"""Framework universal para proyectos de BI y Data Analytics con rigor estadistico.

Este modulo actua como boilerplate de referencia para proyectos de BI de extremo a
extremo. Su foco esta en gobernanza del dato, validacion de supuestos y narrativa
tecnica accionable, siguiendo el espiritu de CRISP-DM y DAMA-DMBOK.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError:  # pragma: no cover - dependencia opcional
    shap = None

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
except ImportError:  # pragma: no cover - dependencia opcional
    BetaGeoFitter = None
    GammaGammaFitter = None

from .datasets_sinteticos import generar_dataset_clientes_sintetico
from .exploracion import resumen_categorico, resumen_numerico
from .limpieza import normalizar_nombres_columnas
from .metodologia import (
    analyze_correlation,
    audit_dataset,
    build_preprocessing_pipeline,
    calculate_vif,
    check_normality,
    check_variance_homogeneity,
    compare_groups,
    compare_power_transformations,
    fit_ols_inference,
    handle_outliers,
    plot_feature_importance,
    plot_group_distributions,
    plot_missingness_heatmap,
    plot_model_diagnostics,
    plot_power_transformations,
    plot_qq_diagnostic,
    train_supervised_model,
)
from .visualizacion import grafico_mapa_correlacion


FRAMEWORK_PHASES: list[dict[str, Any]] = [
    {
        "fase": "Ingesta y auditoria",
        "objetivo": "Validar integridad, trazabilidad y representatividad antes de transformar o modelar.",
        "preguntas_auditoria_critica": [
            "La unidad de analisis esta definida sin ambiguedad o hay riesgo de mezclar cliente, transaccion y periodo?",
            "Existen duplicados, llaves inconsistentes o columnas que se parecen demasiado al objetivo y anticipan leakage?",
            "La cobertura temporal, geografica o por segmento representa el negocio real o hay huecos operativos?",
        ],
        "metodos_resolucion": [
            "Perfilado de calidad, control de duplicados, cardinalidad y trazabilidad de identificadores.",
            "Revision de faltantes y clasificacion conceptual MCAR, MAR o MNAR para evitar imputaciones ingenuas.",
            "Deteccion de leakage por correlaciones extremas, coincidencias exactas con el objetivo y validacion temporal.",
        ],
        "justificacion_estrategica": "Un reporte de BI es tan fiable como su dato de entrada. Si la fuente tiene leakage, duplicados o una ventana temporal no representativa, el modelo o dashboard puede verse brillante en papel y fallar en produccion.",
        "guia_interpretacion_stakeholders": [
            "Muchos nulos o duplicados implican que el KPI puede estar sobrestimado o subestimado.",
            "Una variable casi identica al objetivo no es insight; suele ser fuga de informacion y rompe la credibilidad del analisis.",
            "Cobertura sesgada significa que la decision solo es valida para una parte del negocio, no para toda la operacion.",
        ],
        "visual_analytics": [
            "Heatmap de faltantes para detectar patrones estructurales.",
            "Serie temporal de cobertura para validar estabilidad de la fuente.",
            "Barras de distribucion por segmento para revisar representatividad.",
        ],
        "estandares": ["CRISP-DM - Business Understanding y Data Understanding", "DAMA-DMBOK - Data Quality y Metadata"],
    },
    {
        "fase": "ETL y saneamiento",
        "objetivo": "Estandarizar el dato, controlar outliers y dejar un preprocesado reproducible sin data leakage.",
        "preguntas_auditoria_critica": [
            "Los outliers son errores operativos, eventos raros validos o senales de alto valor de negocio?",
            "La imputacion respeta el mecanismo de ausencia o solo es una comodidad tecnica?",
            "La transformacion de variables se ajusta unicamente sobre train para evitar fuga de datos?",
        ],
        "metodos_resolucion": [
            "Reglas IQR o winsorizacion para controlar extremos sin borrar senal valiosa sin justificacion.",
            "Imputacion numerica y categorica encapsulada en sklearn.pipeline y ColumnTransformer.",
            "PowerTransformer con Yeo-Johnson o Box-Cox para estabilizar varianza y reducir asimetria.",
        ],
        "justificacion_estrategica": "Una ETL robusta protege la trazabilidad y la repetibilidad. Si el escalado o la imputacion se hacen fuera de pipeline, la evaluacion se contamina y los resultados dejan de ser defendibles.",
        "guia_interpretacion_stakeholders": [
            "El objetivo no es maquillar el dato, sino reducir ruido para que el indicador refleje mejor la realidad operativa.",
            "Una transformacion de potencia no cambia el negocio; cambia la forma estadistica para compararlo con menos sesgo.",
            "Un pipeline reproducible reduce dependencia de trabajo manual y baja riesgo operacional en despliegue.",
        ],
        "visual_analytics": [
            "Boxplots antes y despues del tratamiento de outliers.",
            "Histogramas comparativos original vs Yeo-Johnson vs Box-Cox.",
            "Tablas de auditoria ETL con volumen afectado por cada regla aplicada.",
        ],
        "estandares": ["CRISP-DM - Data Preparation", "DAMA-DMBOK - Data Integration y Data Quality"],
    },
    {
        "fase": "EDA y validacion de supuestos",
        "objetivo": "Entender distribuciones, relaciones y riesgos de mala especificacion antes de modelar o inferir.",
        "preguntas_auditoria_critica": [
            "La variable clave se parece a una normal o requiere enfoque robusto/no parametrico?",
            "Existen grupos con varianzas muy distintas que puedan invalidar comparaciones directas?",
            "Hay multicolinealidad suficiente para volver inestables los coeficientes e inflar conclusiones?",
        ],
        "metodos_resolucion": [
            "Shapiro-Wilk y Q-Q plot para normalidad aproximada.",
            "Levene o Brown-Forsythe para homocedasticidad entre grupos.",
            "VIF, numero de condicion escalado y analisis de Belsley para multicolinealidad; correlaciones Pearson o Spearman segun supuestos.",
        ],
        "justificacion_estrategica": "Los supuestos no son burocracia academica. Son el mecanismo que evita convertir una coincidencia estadistica en una accion de negocio equivocada. Ignorar heterocedasticidad o colinealidad puede deformar proyecciones y priorizaciones.",
        "guia_interpretacion_stakeholders": [
            "P-valores bajos indican evidencia, no causalidad. Siempre deben leerse junto a tamano del efecto y contexto.",
            "VIF alto significa que dos o mas variables cuentan casi la misma historia y los coeficientes pueden cambiar demasiado ante pequenas variaciones.",
            "Un numero de condicion alto o componentes criticos de Belsley indican que la inestabilidad es vectorial, no solo bivariada.",
            "Normalidad y homocedasticidad guian que tipo de contraste es tecnicamente defendible.",
        ],
        "visual_analytics": [
            "Q-Q plots para validar desviaciones de normalidad.",
            "Heatmap de correlacion para bloques de redundancia.",
            "Violin y boxplots por grupo para ver dispersion, sesgo y asimetria.",
        ],
        "estandares": ["CRISP-DM - Data Understanding", "DAMA-DMBOK - Data Quality Assessment"],
    },
    {
        "fase": "Modelado",
        "objetivo": "Construir baselines reproducibles e interpretables antes de perseguir complejidad algoritmica.",
        "preguntas_auditoria_critica": [
            "El baseline es explicable y esta alineado con el coste real del error?",
            "Las metricas elegidas reflejan el problema del negocio o solo la comodidad tecnica del analista?",
            "El pipeline separa train y test antes de imputar, transformar y escalar?",
        ],
        "metodos_resolucion": [
            "sklearn.pipeline con imputacion, transformacion y estimador encadenados sin leakage.",
            "Metricas de clasificacion o regresion segun el tipo de objetivo: ROC AUC, F1, MAE, RMSE, R2 y errores escalados cuando el contexto lo justifique, evitando MAPE sobre objetivos con ceros o intermitencia.",
            "Permutation importance como baseline global y SHAP avanzado con beeswarm, dependence plot y auditoria aditiva para justificar impacto de variables.",
        ],
        "justificacion_estrategica": "Modelar en BI no es optimizar una competencia de Kaggle. Es crear una herramienta fiable para priorizar acciones. Un baseline interpretable y bien diagnosticado suele generar mas valor que un modelo opaco sin gobierno.",
        "guia_interpretacion_stakeholders": [
            "ROC AUC alto implica buen ranking relativo de riesgo, no certeza absoluta caso a caso.",
            "MAE y RMSE traducen cuanto se equivoca el modelo en la unidad de negocio original; si divergen, la eleccion del champion debe depender del coste de errores extremos.",
            "R2 indica cuanta variabilidad explica el modelo; un valor moderado puede seguir siendo util si el coste del error es manejable.",
        ],
        "visual_analytics": [
            "Importancia de variables para lectura global.",
            "Matriz de confusion y curva de calibracion en clasificacion.",
            "Observado vs predicho y residuos en regresion.",
        ],
        "estandares": ["CRISP-DM - Modeling", "DAMA-DMBOK - Analytics Governance"],
    },
    {
        "fase": "Inferencia",
        "objetivo": "Cuantificar diferencias y relaciones de forma estadisticamente defendible para soportar decisiones.",
        "preguntas_auditoria_critica": [
            "La comparacion entre grupos cumple supuestos para t-test o ANOVA, o exige Welch o Kruskal-Wallis?",
            "Los coeficientes lineales resisten heterocedasticidad y colinealidad?",
            "El tamano del efecto es material para negocio o solo estadisticamente detectable por tamano muestral?",
        ],
        "metodos_resolucion": [
            "Student, Welch, Mann-Whitney, ANOVA, Welch ANOVA o Kruskal-Wallis segun supuestos.",
            "OLS con errores robustos HC3 para no depender de homocedasticidad perfecta.",
            "VIF, numero de condicion escalado, Belsley e intervalos de confianza para robustecer interpretacion.",
        ],
        "justificacion_estrategica": "La inferencia es la barrera entre intuicion y evidencia. Sin errores robustos, chequeos de supuestos y tamano del efecto, una conclusion puede ser estadisticamente fragil y financieramente costosa.",
        "guia_interpretacion_stakeholders": [
            "Un p-valor pequeno dice que la diferencia es dificil de atribuir al azar bajo el modelo asumido; no mide impacto economico por si solo.",
            "Intervalos estrechos aumentan confianza operativa; intervalos amplios sugieren prudencia y posible necesidad de mas datos.",
            "Coeficientes con VIF alto son menos estables y deben evitarse para decisiones sensibles o causalidad fuerte.",
        ],
        "visual_analytics": [
            "Boxplots y violin plots por grupo.",
            "Grafico de coeficientes con intervalos de confianza.",
            "Scatter de residuos para validar especificacion lineal.",
        ],
        "estandares": ["CRISP-DM - Evaluation", "DAMA-DMBOK - Data Science Risk Control"],
    },
    {
        "fase": "Conclusiones y accion",
        "objetivo": "Traducir hallazgos tecnicos a decisiones, riesgos y plan de accion priorizado.",
        "preguntas_auditoria_critica": [
            "Que decision concreta cambia si el hallazgo es verdadero?",
            "Que riesgo residual permanece aunque el modelo o contraste sea estadisticamente valido?",
            "Que controles de seguimiento deben entrar al dashboard o al gobierno del dato?",
        ],
        "metodos_resolucion": [
            "Sintesis ejecutiva con hallazgo, impacto esperado, riesgo residual y accion recomendada.",
            "Separacion explicita entre evidencia, interpretacion y recomendacion operativa.",
            "Checklist de monitoreo para drift, recalibracion y calidad de dato post-despliegue.",
        ],
        "justificacion_estrategica": "La ultima milla del BI es convertir estadistica en priorizacion. Sin traducir el resultado a impacto, riesgo y accion, el pipeline termina en un informe bonito pero no en una decision mejor.",
        "guia_interpretacion_stakeholders": [
            "Cada hallazgo debe cerrar con una accion: mantener, corregir, escalar o investigar.",
            "Toda conclusion relevante debe declarar que tan robusta es y bajo que condiciones deja de ser valida.",
            "Un buen informe ejecutivo no oculta incertidumbre; la hace util para decidir mejor.",
        ],
        "visual_analytics": [
            "Tabla semaforizada de riesgos y acciones.",
            "Dashboard de KPIs con umbrales de calidad y performance.",
            "Grafico de impacto vs esfuerzo para priorizar roadmap analitico.",
        ],
        "estandares": ["CRISP-DM - Deployment", "DAMA-DMBOK - Data Governance"],
    },
]


METRIC_TRANSLATION_GUIDE: list[dict[str, str]] = [
    {
        "metrica": "p_value",
        "lectura_ejecutiva": "Cuanta evidencia hay contra una explicacion por azar bajo el supuesto estadistico elegido.",
        "accion_bi": "Si es bajo y el efecto es material, prioriza la accion. Si es alto, evita vender certeza donde no la hay.",
    },
    {
        "metrica": "r2",
        "lectura_ejecutiva": "Porcentaje de variabilidad del negocio explicado por el modelo.",
        "accion_bi": "Si es alto, el modelo sirve para planificacion. Si es moderado, usalo como apoyo y no como automatizacion ciega.",
    },
    {
        "metrica": "mae",
        "lectura_ejecutiva": "Error medio absoluto en la misma unidad del KPI estimado.",
        "accion_bi": "Compuralo contra la tolerancia del negocio para saber si el modelo es util operativamente.",
    },
    {
        "metrica": "rmse",
        "lectura_ejecutiva": "Error que penaliza mas los fallos grandes y anticipa riesgo de proyecciones extremas.",
        "accion_bi": "Si supera la tolerancia financiera, ajusta features o limita el uso del modelo a apoyo tactico.",
    },
    {
        "metrica": "smape",
        "lectura_ejecutiva": "Error porcentual simetrico util para comparar magnitudes relativas sin el sesgo explosivo de MAPE.",
        "accion_bi": "Usalo como apoyo explicativo cuando necesitas porcentajes, pero manteniendo MAE y RMSE como anclas principales de decision.",
    },
    {
        "metrica": "mae_relativo_baseline",
        "lectura_ejecutiva": "Cuanto mejora o empeora el error medio frente a un baseline ingenuo del negocio.",
        "accion_bi": "Si es menor que 1, el modelo supera al baseline; si no, todavia no justifica complejidad operativa adicional.",
    },
    {
        "metrica": "roc_auc",
        "lectura_ejecutiva": "Capacidad de ordenar correctamente casos de mayor y menor riesgo.",
        "accion_bi": "Usalo para priorizacion comercial, antifraude o churn cuando importa ordenar antes que acertar umbral fijo.",
    },
    {
        "metrica": "f1",
        "lectura_ejecutiva": "Balance entre precision y recall, util cuando hay clases desbalanceadas.",
        "accion_bi": "Si es bajo, revisa coste de falsos positivos y falsos negativos antes de activar campañas o alertas.",
    },
    {
        "metrica": "vif",
        "lectura_ejecutiva": "Nivel de redundancia entre predictores; valores altos degradan estabilidad explicativa.",
        "accion_bi": "Reduce variables redundantes o regulariza si necesitas coeficientes fiables para decisiones sensibles; si el numero de condicion tambien es alto, revisa la estructura completa del espacio de variables.",
    },
]


def _emit_interpretation(message: str, verbose: bool) -> None:
    if verbose:
        print(f"Interpretacion: {message}")


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"No se encontraron las columnas requeridas: {missing}")


def _safe_smape(y_true: Sequence[float], y_pred: Sequence[float], epsilon: float = 1e-6) -> float:
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    denominator = np.maximum((np.abs(true_values) + np.abs(predicted_values)) / 2.0, epsilon)
    return float(np.mean(np.abs(true_values - predicted_values) / denominator) * 100)


def _resolve_error_policy_weights(catastrophic_error_weight: float) -> tuple[str, float, float]:
    if catastrophic_error_weight <= 0:
        raise ValueError("catastrophic_error_weight debe ser mayor que cero.")

    if catastrophic_error_weight >= 1.35:
        return (
            "La politica financiera prioriza blindarse contra errores extremos; RMSE recibe mas peso que MAE en la seleccion del champion.",
            1.0,
            float(catastrophic_error_weight),
        )

    if catastrophic_error_weight <= 0.85:
        return (
            "La politica financiera tolera outliers aislados y prioriza consistencia media; MAE recibe mas peso que RMSE en la seleccion del champion.",
            float(1.0 / catastrophic_error_weight),
            1.0,
        )

    return (
        "La politica financiera trata de equilibrar errores medios y errores extremos; MAE y RMSE pesan de forma casi pareja.",
        1.0,
        1.0,
    )


def _translate_metric(metric_name: str, value: float) -> str:
    metric = metric_name.lower()

    if metric == "p_value":
        if value < 0.01:
            return "La evidencia estadistica es muy fuerte; pasa a validar relevancia de negocio y coste de implementacion."
        if value < 0.05:
            return "Hay evidencia suficiente para considerar accion, siempre que el tamano del efecto sea material."
        return "La evidencia es insuficiente para defender una accion firme solo con este contraste."

    if metric == "r2":
        if value >= 0.7:
            return "El modelo explica gran parte de la variabilidad y puede apoyar planificacion y forecast con confianza razonable."
        if value >= 0.4:
            return "La explicacion es util como apoyo, pero no basta para automatizar decisiones sin supervision."
        return "La capacidad explicativa es limitada; conviene enriquecer datos o acotar el alcance operativo."

    if metric == "mae":
        return "Compara este error medio con la tolerancia economica del proceso para decidir si el modelo es util."

    if metric == "rmse":
        return "Si este error penalizado supera el umbral financiero aceptable, las proyecciones extremas necesitan contencion."

    if metric == "smape":
        return "Lee este porcentaje como apoyo comparativo y evita usarlo como unico criterio cuando existan ceros, intermitencia o colas pesadas."

    if metric == "mae_relativo_baseline":
        if value < 1:
            return "El modelo mejora al baseline ingenuo; hay evidencia de valor adicional mas alla de una regla trivial."
        return "El modelo no mejora al baseline ingenuo; conviene revisar variables, algoritmo o alcance antes de industrializarlo."

    if metric == "roc_auc":
        if value >= 0.8:
            return "El modelo ordena bien el riesgo y sirve para priorizar casos o segmentos."
        if value >= 0.7:
            return "Hay senal util para priorizacion, aunque todavia puede haber mezcla relevante entre clases."
        return "La capacidad de ranking es debil y no conviene usarla para decisiones automaticas."

    if metric == "f1":
        if value >= 0.75:
            return "El equilibrio entre precision y recall es fuerte para un baseline interpretable."
        if value >= 0.6:
            return "El baseline es util, pero requiere revisar umbral, desbalance o nuevas variables."
        return "El rendimiento aun es fragil para activar operaciones sensibles."

    if metric == "vif":
        if value >= 10:
            return "La colinealidad es severa y los coeficientes pueden ser inestables."
        if value >= 5:
            return "Existe redundancia relevante; conviene simplificar variables, revisar numero de condicion o regularizar."
        return "La redundancia es manejable para interpretacion y estimacion."

    return "Interpreta la metrica junto a coste del error, tamano del efecto y contexto operativo."


def _safe_logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def _ensure_feature_named_matrix(
    values: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None,
) -> pd.DataFrame | np.ndarray:
    if isinstance(values, pd.DataFrame) or feature_names is None:
        return values
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return pd.DataFrame(matrix, columns=list(feature_names))


class _FeatureNameAwareEstimatorProxy:
    """Adapta estimadores tabulares para preservar nombres de columnas en inferencia."""

    def __init__(self, estimator: Any, feature_names: Sequence[str] | None = None) -> None:
        self._estimator = estimator
        self._feature_names = tuple(feature_names) if feature_names is not None else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._estimator, name)

    def predict(self, values: pd.DataFrame | np.ndarray) -> Any:
        return self._estimator.predict(_ensure_feature_named_matrix(values, self._feature_names))

    def predict_proba(self, values: pd.DataFrame | np.ndarray) -> Any:
        return self._estimator.predict_proba(_ensure_feature_named_matrix(values, self._feature_names))

    def decision_function(self, values: pd.DataFrame | np.ndarray) -> Any:
        return self._estimator.decision_function(_ensure_feature_named_matrix(values, self._feature_names))


def _is_tree_shap_candidate(estimator: Any) -> bool:
    module_name = str(getattr(estimator.__class__, "__module__", "")).lower()
    class_name = str(getattr(estimator.__class__, "__name__", "")).lower()
    tree_tokens = ("lightgbm", "xgboost", "catboost", "forest", "gradientboosting", "tree")
    return any(token in module_name or token in class_name for token in tree_tokens)


def _resolve_shap_matrix(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 3:
        matrix = matrix[:, :, -1]
    if matrix.ndim != 2:
        raise ValueError("No fue posible convertir la salida SHAP a una matriz bidimensional interpretable.")
    return matrix


def _resolve_shap_base_values(base_values: Any, n_samples: int) -> np.ndarray:
    resolved = np.asarray(base_values, dtype=float)
    if resolved.ndim == 0:
        return np.repeat(float(resolved), n_samples)
    if resolved.ndim == 2:
        return resolved[:, -1]
    if resolved.ndim > 2:
        return resolved.reshape(n_samples, -1)[:, -1]
    if resolved.shape[0] != n_samples:
        return np.repeat(float(np.ravel(resolved)[-1]), n_samples)
    return resolved


def _audit_shap_consistency(
    estimator: Any,
    transformed_sample: pd.DataFrame | np.ndarray,
    problem_type: str,
    reconstructed_output: np.ndarray,
) -> pd.DataFrame:
    if problem_type == "classification" and hasattr(estimator, "predict_proba"):
        predicted_probability = estimator.predict_proba(transformed_sample)
        if predicted_probability.ndim == 2:
            predicted_probability = predicted_probability[:, -1]
        probability_error = np.abs(reconstructed_output - predicted_probability)
        candidate_errors: list[tuple[str, np.ndarray]] = [("probability", probability_error)]

        if hasattr(estimator, "decision_function"):
            decision_output = np.asarray(estimator.decision_function(transformed_sample), dtype=float)
            if decision_output.ndim == 2:
                decision_output = decision_output[:, -1]
            candidate_errors.append(("decision_function", np.abs(reconstructed_output - decision_output)))
        else:
            candidate_errors.append(("log_odds", np.abs(reconstructed_output - _safe_logit(predicted_probability))))
    else:
        prediction = np.asarray(estimator.predict(transformed_sample), dtype=float)
        if prediction.ndim == 2:
            prediction = prediction[:, -1]
        candidate_errors = [("prediction", np.abs(reconstructed_output - prediction))]

    best_space, best_error = min(candidate_errors, key=lambda item: float(np.nanmean(item[1])))
    error_mean = float(np.nanmean(best_error))
    error_max = float(np.nanmax(best_error))
    is_consistent = bool(error_mean <= 1e-3 and error_max <= 1e-2)
    return pd.DataFrame(
        [
            {
                "espacio_salida": best_space,
                "error_abs_medio": error_mean,
                "error_abs_max": error_max,
                "reconstruccion_consistente": is_consistent,
                "n_observaciones_auditadas": int(len(reconstructed_output)),
            }
        ]
    )


def _audit_feature_dependence(sample: pd.DataFrame, threshold: float = 0.65) -> pd.DataFrame:
    numeric_sample = sample.select_dtypes(include=[np.number]).copy()
    if numeric_sample.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_spearman", "riesgo_interpretacion"])

    correlation = numeric_sample.corr(method="spearman").abs()
    pairs: list[dict[str, Any]] = []
    for row_position, feature_a in enumerate(correlation.columns):
        for feature_b in correlation.columns[row_position + 1 :]:
            strength = float(correlation.loc[feature_a, feature_b])
            if np.isnan(strength) or strength < threshold:
                continue
            pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "abs_spearman": round(strength, 4),
                    "riesgo_interpretacion": (
                        "Dependencia alta: conviene leer SHAP como distribucion de contribuciones compartidas y no como reparto causal limpio."
                    ),
                }
            )

    return pd.DataFrame(pairs).sort_values("abs_spearman", ascending=False).reset_index(drop=True) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "abs_spearman", "riesgo_interpretacion"]
    )


def get_bi_framework_manual(as_dataframe: bool = False) -> dict[str, Any] | pd.DataFrame:
    """Devuelve el manual metodologico universal para un pipeline BI de extremo a extremo.

    Parametros:
        as_dataframe: Si es True, devuelve una tabla resumida por fase. Si es False,
            devuelve la estructura completa con listas por categoria.

    Retorna:
        Un diccionario estructurado o un DataFrame listo para consulta ejecutiva.

    Logica estadistica:
        No ejecuta pruebas; consolida el marco de decision recomendado para auditar,
        transformar, explorar, modelar, inferir y concluir sobre datos estructurados.
    """
    if not as_dataframe:
        return {
            "framework": FRAMEWORK_PHASES,
            "metric_translation_guide": METRIC_TRANSLATION_GUIDE,
            "estandares": ["CRISP-DM", "DAMA-DMBOK"],
        }

    rows: list[dict[str, str]] = []
    for phase in FRAMEWORK_PHASES:
        rows.append(
            {
                "fase": phase["fase"],
                "objetivo": phase["objetivo"],
                "preguntas_auditoria_critica": " | ".join(phase["preguntas_auditoria_critica"]),
                "metodos_resolucion": " | ".join(phase["metodos_resolucion"]),
                "justificacion_estrategica": phase["justificacion_estrategica"],
                "guia_interpretacion_stakeholders": " | ".join(phase["guia_interpretacion_stakeholders"]),
                "visual_analytics": " | ".join(phase["visual_analytics"]),
                "estandares": " | ".join(phase["estandares"]),
            }
        )
    return pd.DataFrame(rows)


def get_metric_translation_guide() -> pd.DataFrame:
    """Devuelve una guia ejecutiva para traducir metricas tecnicas a decisiones BI.

    Parametros:
        No recibe parametros.

    Retorna:
        DataFrame con metrica, lectura ejecutiva y accion sugerida.

    Logica estadistica:
        No calcula metricas nuevas; actua como tabla puente entre salida analitica y
        decision de negocio para evitar conclusiones tecnicamente correctas pero poco utiles.
    """
    return pd.DataFrame(METRIC_TRANSLATION_GUIDE)


def build_demo_bi_dataset(n_registros: int = 1400, semilla: int = 42) -> pd.DataFrame:
    """Construye un dataset demo complejo y agnostico para validar el framework.

    Parametros:
        n_registros: Numero de observaciones a generar.
        semilla: Semilla de reproducibilidad.

    Retorna:
        DataFrame con variables mixtas, nulos, outliers, fecha de corte y objetivos
        compatibles con analisis descriptivo, inferencial y supervisado.

    Logica estadistica:
        Parte de un generador sintetico reusable y anade columnas con senal, ruido y
        complejidad estructural para estresar auditoria, ETL, EDA, modelado e inferencia.
    """
    rng = np.random.default_rng(semilla)
    df = normalizar_nombres_columnas(generar_dataset_clientes_sintetico(n_registros=n_registros, semilla=semilla))

    df["registro_id"] = np.arange(1, len(df) + 1)
    df["fecha_corte"] = pd.Timestamp("2026-03-31") - pd.to_timedelta(
        rng.integers(0, 540, size=len(df)),
        unit="D",
    )
    df["antiguedad_meses"] = rng.integers(1, 121, size=len(df))
    df["promociones_12m"] = rng.poisson(lam=3.2, size=len(df))
    df["ticket_medio"] = (df["gasto_mensual"] / np.clip(df["compras_12m"], 1, None)).round(2)
    df["margen_estimado"] = (
        df["gasto_mensual"] * rng.uniform(0.18, 0.42, size=len(df)) - df["reclamaciones"] * rng.uniform(2, 9, size=len(df))
    ).round(2)
    df["indice_engagement"] = np.clip(
        18
        + 1.7 * df["visitas_web_30d"].fillna(df["visitas_web_30d"].median())
        + 4.8 * df["compras_12m"]
        + 2.5 * df["satisfaccion"].fillna(df["satisfaccion"].median())
        - 3.2 * df["reclamaciones"],
        0,
        100,
    ).round(2)
    df["canal_servicio"] = rng.choice(["Digital", "Sucursal", "Mixto"], size=len(df), p=[0.48, 0.22, 0.30])
    df["macro_segmento"] = rng.choice(["Masivo", "Valor", "Premium", "Recuperacion"], size=len(df), p=[0.4, 0.3, 0.15, 0.15])

    raw_risk_score = (
        0.90 * df["reclamaciones"].fillna(0)
        - 0.55 * df["satisfaccion"].fillna(df["satisfaccion"].median())
        - 0.015 * df["antiguedad_meses"]
        + 0.00035 * np.maximum(3200 - df["ingreso_mensual"].fillna(df["ingreso_mensual"].median()), 0)
        + 0.75 * (df["usa_app"].astype(str).str.lower() == "no").astype(int)
        + 0.55 * (df["macro_segmento"] == "Recuperacion").astype(int)
        + 0.20 * (df["canal_servicio"] == "Sucursal").astype(int)
        - 0.030 * df["indice_engagement"]
    )
    standardized_risk = (raw_risk_score - raw_risk_score.mean()) / raw_risk_score.std(ddof=0)
    churn_probability = 1 / (1 + np.exp(-(1.4 * standardized_risk - 0.9)))
    df["abandono"] = rng.binomial(1, np.clip(churn_probability, 0.08, 0.82)).astype(int)

    missing_idx = rng.choice(df.index, size=max(12, len(df) // 30), replace=False)
    outlier_idx = rng.choice(df.index, size=max(10, len(df) // 90), replace=False)
    df.loc[missing_idx, "ticket_medio"] = np.nan
    df.loc[outlier_idx, "margen_estimado"] = df.loc[outlier_idx, "margen_estimado"] * 2.8

    ordered_columns = [
        "registro_id",
        "cliente_id",
        "fecha_registro",
        "fecha_corte",
        "region",
        "canal_captacion",
        "canal_servicio",
        "segmento",
        "macro_segmento",
        "edad",
        "antiguedad_meses",
        "ingreso_mensual",
        "gasto_mensual",
        "ticket_medio",
        "margen_estimado",
        "visitas_web_30d",
        "compras_12m",
        "promociones_12m",
        "indice_engagement",
        "satisfaccion",
        "reclamaciones",
        "usa_app",
        "producto_premium",
        "abandono",
    ]
    return df[ordered_columns].copy()


def build_bank_client_case_dataset(n_registros: int = 1800, semilla: int = 17) -> pd.DataFrame:
    """Construye un caso sintetico de clientes bancarios con foco en fuga y rentabilidad.

    Parametros:
        n_registros: Numero de clientes a simular.
        semilla: Semilla de reproducibilidad.

    Retorna:
        DataFrame con variables de relacion bancaria, vinculo digital, riesgo operativo,
        valor economico y objetivo de abandono listo para un caso de retencion.

    Logica estadistica:
        Parte del dataset demo universal, anade senales tipicas de banca retail y vuelve
        a generar la probabilidad de abandono para que el caso combine riesgo, vinculo,
        uso de productos y valor economico de cliente.
    """
    rng = np.random.default_rng(semilla + 101)
    df = build_demo_bi_dataset(n_registros=n_registros, semilla=semilla).copy()

    segmento_factor = df["macro_segmento"].map(
        {
            "Masivo": 0.95,
            "Valor": 1.15,
            "Premium": 1.45,
            "Recuperacion": 0.72,
        }
    ).fillna(1.0)
    premium_flag = (df["producto_premium"].astype(str).str.lower() == "si").astype(int)
    app_flag = (df["usa_app"].astype(str).str.lower() == "si").astype(int)
    ingreso_proxy = df["ingreso_mensual"].fillna(df["ingreso_mensual"].median())

    df["saldo_promedio_3m"] = np.clip(
        ingreso_proxy * rng.uniform(1.7, 6.6, size=len(df)) * segmento_factor
        + df["antiguedad_meses"] * rng.uniform(2.5, 7.5, size=len(df))
        - df["reclamaciones"] * rng.uniform(90, 260, size=len(df)),
        180,
        None,
    ).round(2)
    df["saldo_variacion_pct_3m"] = np.clip(
        rng.normal(0.01, 0.11, size=len(df))
        + 0.03 * app_flag
        + 0.01 * premium_flag
        - 0.012 * df["reclamaciones"]
        + 0.004 * df["satisfaccion"].fillna(df["satisfaccion"].median()),
        -0.58,
        0.42,
    ).round(4)
    df["productos_activos"] = np.clip(
        rng.poisson(lam=2.4, size=len(df)) + premium_flag + (segmento_factor > 1.1).astype(int),
        1,
        7,
    )
    df["ratio_uso_credito"] = np.clip(
        rng.beta(2.4, 3.2, size=len(df))
        + 0.05 * (df["reclamaciones"] >= 2).astype(int)
        - 0.03 * premium_flag,
        0.03,
        0.99,
    ).round(3)
    visitas_web_proxy = df["visitas_web_30d"].fillna(df["visitas_web_30d"].median())
    df["transacciones_app_30d"] = np.clip(
        rng.poisson(lam=9.0, size=len(df)) + 6 * app_flag + (visitas_web_proxy * 0.35).round().astype(int),
        0,
        None,
    )
    df["interacciones_sucursal_90d"] = np.clip(
        rng.poisson(lam=1.8, size=len(df))
        + (df["canal_servicio"] == "Sucursal").astype(int)
        + (df["canal_servicio"] == "Mixto").astype(int),
        0,
        None,
    )
    df["nomina_domiciliada"] = np.where(
        rng.random(len(df))
        < np.clip(0.28 + 0.06 * premium_flag + 0.08 * (df["antiguedad_meses"] >= 36).astype(int), 0.12, 0.88),
        "Si",
        "No",
    )
    df["hipoteca_activa"] = np.where(
        rng.random(len(df))
        < np.clip(
            0.12
            + 0.07 * (df["edad"].between(30, 55)).astype(int)
            + 0.05 * (ingreso_proxy >= ingreso_proxy.median()).astype(int),
            0.03,
            0.48,
        ),
        "Si",
        "No",
    )
    atraso_prob = np.clip(
        0.025
        + 0.16 * (df["ratio_uso_credito"] >= 0.78).astype(int)
        + 0.06 * (df["nomina_domiciliada"] == "No").astype(int)
        + 0.03 * (df["saldo_variacion_pct_3m"] <= -0.12).astype(int),
        0.01,
        0.55,
    )
    df["atraso_30d"] = rng.binomial(1, atraso_prob, size=len(df)).astype(int)
    df["rentabilidad_mensual_estimada"] = np.clip(
        18
        + 0.018 * df["saldo_promedio_3m"]
        + 7.5 * df["productos_activos"]
        + 0.42 * df["compras_12m"]
        - 14.0 * df["reclamaciones"]
        - 22.0 * df["atraso_30d"]
        + 10.0 * premium_flag,
        6,
        None,
    ).round(2)
    df["valor_cliente_12m"] = (
        df["rentabilidad_mensual_estimada"]
        * np.clip(rng.normal(11.2, 1.4, size=len(df)), 7.5, 14.0)
    ).round(2)
    df["clv_t_dias"] = np.clip(
        (df["antiguedad_meses"] * 30 + rng.integers(-18, 19, size=len(df))).astype(int),
        120,
        1080,
    )
    event_count = np.clip(
        (
            df["compras_12m"].fillna(0).astype(float)
            + np.rint(df["transacciones_app_30d"] / 8.0)
            + df["productos_activos"].astype(float)
            + premium_flag
            - 1.0
        ),
        1,
        None,
    ).astype(int)
    df["clv_frecuencia"] = np.maximum(event_count - 1, 0).astype(int)
    inactivity_days = np.clip(
        20
        + 125 / (1 + np.log1p(df["clv_frecuencia"] + 1))
        + 24 * df["reclamaciones"].fillna(0)
        + 38 * (df["saldo_variacion_pct_3m"] <= -0.08).astype(int)
        + 24 * df["atraso_30d"]
        - 1.3 * df["transacciones_app_30d"]
        - 10 * app_flag,
        1,
        df["clv_t_dias"] - 1,
    ).round().astype(int)
    df["clv_t_ultima_compra_dias"] = np.where(
        df["clv_frecuencia"] > 0,
        df["clv_t_dias"] - inactivity_days,
        0,
    ).astype(int)
    df["clv_dias_desde_ultima_transaccion"] = np.where(
        df["clv_frecuencia"] > 0,
        inactivity_days,
        df["clv_t_dias"],
    ).astype(int)
    df["clv_monetario_promedio"] = np.clip(
        (
            df["valor_cliente_12m"] / np.maximum(event_count, 1)
        ) * rng.uniform(0.86, 1.14, size=len(df))
        + 9 * premium_flag
        + 4 * (df["nomina_domiciliada"] == "Si").astype(int),
        18,
        None,
    ).round(2)
    df["clv_valor_observado_12m"] = (event_count * df["clv_monetario_promedio"]).round(2)

    raw_risk_score = (
        1.25 * df["atraso_30d"]
        + 0.95 * df["reclamaciones"].fillna(0)
        - 0.62 * df["satisfaccion"].fillna(df["satisfaccion"].median())
        - 0.022 * df["antiguedad_meses"]
        - 0.32 * df["productos_activos"]
        + 1.35 * np.maximum(-df["saldo_variacion_pct_3m"], 0)
        + 0.85 * (df["nomina_domiciliada"] == "No").astype(int)
        + 0.55 * (df["hipoteca_activa"] == "No").astype(int)
        + 0.90 * (1 - app_flag)
        + 0.75 * (df["ratio_uso_credito"] - 0.5)
        - 0.018 * df["transacciones_app_30d"]
        - 0.00045 * df["saldo_promedio_3m"]
    )
    standardized_risk = (raw_risk_score - raw_risk_score.mean()) / raw_risk_score.std(ddof=0)
    churn_probability = 1 / (1 + np.exp(-(1.55 * standardized_risk - 0.95)))
    df["abandono"] = rng.binomial(1, np.clip(churn_probability, 0.05, 0.78), size=len(df)).astype(int)

    missing_idx = rng.choice(df.index, size=max(14, len(df) // 34), replace=False)
    outlier_idx = rng.choice(df.index, size=max(12, len(df) // 85), replace=False)
    df.loc[missing_idx, "saldo_variacion_pct_3m"] = np.nan
    df.loc[outlier_idx, "valor_cliente_12m"] = df.loc[outlier_idx, "valor_cliente_12m"] * 1.9

    ordered_columns = [
        "registro_id",
        "cliente_id",
        "fecha_registro",
        "fecha_corte",
        "region",
        "canal_captacion",
        "canal_servicio",
        "segmento",
        "macro_segmento",
        "edad",
        "antiguedad_meses",
        "ingreso_mensual",
        "saldo_promedio_3m",
        "saldo_variacion_pct_3m",
        "gasto_mensual",
        "ticket_medio",
        "productos_activos",
        "ratio_uso_credito",
        "transacciones_app_30d",
        "interacciones_sucursal_90d",
        "compras_12m",
        "promociones_12m",
        "indice_engagement",
        "satisfaccion",
        "reclamaciones",
        "usa_app",
        "nomina_domiciliada",
        "hipoteca_activa",
        "producto_premium",
        "atraso_30d",
        "rentabilidad_mensual_estimada",
        "valor_cliente_12m",
        "clv_t_dias",
        "clv_frecuencia",
        "clv_t_ultima_compra_dias",
        "clv_dias_desde_ultima_transaccion",
        "clv_monetario_promedio",
        "clv_valor_observado_12m",
        "abandono",
    ]
    return df[ordered_columns].copy()


def audit_bank_probabilistic_clv_inputs(
    df: pd.DataFrame,
    frequency_column: str = "clv_frecuencia",
    recency_column: str = "clv_t_ultima_compra_dias",
    age_column: str = "clv_t_dias",
    monetary_column: str = "clv_monetario_promedio",
    verbose: bool = True,
) -> dict[str, Any]:
    """Audita la estructura RFM-T necesaria para un CLV probabilistico defendible.

    Parametros:
        df: Dataset cliente-nivel con estadisticos de comportamiento.
        frequency_column: Frecuencia repetida compatible con BTYD.
        recency_column: Antiguedad de la ultima compra dentro de la ventana observada.
        age_column: Edad total de la relacion en la ventana de observacion.
        monetary_column: Valor monetario medio por transaccion repetida.
        verbose: Si es True, imprime una lectura automatizada.

    Retorna:
        Diccionario con chequeos estructurales, independencia frecuencia-monetario e interpretacion.

    Logica estadistica:
        Revisa las restricciones RFM-T del marco BG/NBD y comprueba si la correlacion entre
        frecuencia y monetario en repetidores es lo bastante baja como para no contradecir
        el supuesto operativo habitual del modelo Gamma-Gamma.
    """
    _ensure_columns(df, [frequency_column, recency_column, age_column, monetary_column])

    frequency = df[frequency_column].fillna(0).astype(float)
    recency = df[recency_column].fillna(0).astype(float)
    age = df[age_column].fillna(0).astype(float)
    monetary = df[monetary_column].fillna(0).astype(float)

    invalid_age = age <= 0
    invalid_frequency = frequency < 0
    invalid_recency = (recency < 0) | (recency > age)
    invalid_monetary = (frequency > 0) & (monetary <= 0)
    repeated_mask = (frequency > 0) & (~invalid_recency) & (~invalid_age) & (monetary > 0)

    structural_checks = pd.DataFrame(
        [
            {
                "chequeo": "edad_observacion_valida",
                "pct_invalidos": round(float(invalid_age.mean() * 100), 4),
                "lectura": "La ventana de observacion debe ser positiva para todo cliente.",
            },
            {
                "chequeo": "frecuencia_no_negativa",
                "pct_invalidos": round(float(invalid_frequency.mean() * 100), 4),
                "lectura": "La frecuencia repetida no puede ser negativa.",
            },
            {
                "chequeo": "recencia_dentro_de_T",
                "pct_invalidos": round(float(invalid_recency.mean() * 100), 4),
                "lectura": "La ultima compra observada debe caer dentro de la vida del cliente.",
            },
            {
                "chequeo": "monetario_positivo_en_repetidores",
                "pct_invalidos": round(float(invalid_monetary.mean() * 100), 4),
                "lectura": "Gamma-Gamma requiere ticket positivo en clientes con compras repetidas.",
            },
        ]
    )

    corr_value = np.nan
    corr_pvalue = np.nan
    if repeated_mask.sum() >= 3 and monetary[repeated_mask].nunique() > 1 and frequency[repeated_mask].nunique() > 1:
        corr_value, corr_pvalue = pearsonr(frequency[repeated_mask], monetary[repeated_mask])

    if np.isnan(corr_value):
        independence_read = "No hay suficiente variacion en repetidores para auditar independencia frecuencia-monetario con Pearson."
    elif abs(corr_value) <= 0.30:
        independence_read = "La correlacion frecuencia-monetario es baja; el supuesto operativo Gamma-Gamma es razonable para este caso."
    else:
        independence_read = "La correlacion frecuencia-monetario es material; conviene tratar el CLV monetario con prudencia y mantener shrinkage conservador."

    independence_summary = pd.DataFrame(
        [
            {
                "clientes_repetidores": int(repeated_mask.sum()),
                "pearson_r": round(float(corr_value), 4) if not np.isnan(corr_value) else np.nan,
                "p_value": round(float(corr_pvalue), 4) if not np.isnan(corr_pvalue) else np.nan,
                "lectura": independence_read,
            }
        ]
    )

    invalid_rows = int((invalid_age | invalid_frequency | invalid_recency | invalid_monetary).sum())
    interpretation = (
        f"La auditoria RFM-T detecta {invalid_rows} filas estructuralmente no aptas para un ajuste probabilistico exacto. "
        f"{independence_read}"
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "structural_checks": structural_checks,
        "independence_summary": independence_summary,
        "valid_mask": ~(invalid_age | invalid_frequency | invalid_recency | invalid_monetary),
        "repeated_mask": repeated_mask,
        "interpretation": interpretation,
    }


def estimate_bank_probabilistic_clv(
    df: pd.DataFrame,
    frequency_column: str = "clv_frecuencia",
    recency_column: str = "clv_t_ultima_compra_dias",
    age_column: str = "clv_t_dias",
    monetary_column: str = "clv_monetario_promedio",
    horizons_months: Sequence[int] = (6, 12),
    monthly_discount_rate: float = 0.01,
    penalizer_coef: float = 0.01,
    verbose: bool = True,
) -> dict[str, Any]:
    """Estima una capa CLV probabilistica para el caso bancario.

    Parametros:
        df: Dataset cliente-nivel con columnas RFM-T.
        frequency_column: Frecuencia repetida compatible con BTYD.
        recency_column: Recencia BTYD medida en la misma unidad que T.
        age_column: Edad total del cliente en la ventana de observacion.
        monetary_column: Ticket monetario medio para repetidores.
        horizons_months: Horizontes de prediccion a generar.
        monthly_discount_rate: Tasa mensual de descuento para valor presente.
        penalizer_coef: Penalizacion suave si existe ajuste exacto con `lifetimes`.
        verbose: Si es True, imprime lectura automatizada.

    Retorna:
        Diccionario con dataset enriquecido, auditoria, resumen del motor usado e interpretacion.

    Logica estadistica:
        Si `lifetimes` esta disponible, intenta un ajuste BG/NBD para frecuencia-recencia y un
        Gamma-Gamma para monetario. Si el entorno o la muestra no lo permiten, cae a un
        estimador conservador basado en shrinkage de tasa de compra y ticket medio, manteniendo
        trazabilidad sobre el motor utilizado.
    """
    _ensure_columns(df, [frequency_column, recency_column, age_column, monetary_column])
    horizons = sorted({int(horizon) for horizon in horizons_months if int(horizon) > 0})
    if not horizons:
        raise ValueError("Se requiere al menos un horizonte positivo para estimar CLV probabilistico.")

    enriched = df.copy()
    audit = audit_bank_probabilistic_clv_inputs(
        enriched,
        frequency_column=frequency_column,
        recency_column=recency_column,
        age_column=age_column,
        monetary_column=monetary_column,
        verbose=False,
    )

    valid_mask = audit["valid_mask"]
    repeated_mask = audit["repeated_mask"]
    frequency = enriched[frequency_column].fillna(0).astype(float)
    recency = enriched[recency_column].fillna(0).astype(float)
    age = enriched[age_column].fillna(0).astype(float)
    monetary = enriched[monetary_column].fillna(0).astype(float)
    global_monetary = float(monetary[repeated_mask].median()) if repeated_mask.any() else float(monetary.median())
    global_monetary = max(global_monetary, 18.0)

    def _fallback_estimation() -> tuple[str, pd.Series, pd.Series, dict[int, pd.Series]]:
        recency_ratio = np.divide(recency, age, out=np.zeros(len(enriched)), where=age > 0)
        alive_probability = 1.0 / (1.0 + np.exp(-(-1.35 + 2.85 * recency_ratio + 0.55 * np.log1p(frequency))))
        alive_probability = pd.Series(np.clip(alive_probability, 0.03, 0.995), index=enriched.index, dtype=float)
        expected_profit = pd.Series(
            np.where(
                frequency > 0,
                (frequency * monetary + 3.0 * global_monetary) / (frequency + 3.0),
                global_monetary * 0.85,
            ),
            index=enriched.index,
            dtype=float,
        )
        expected_transactions: dict[int, pd.Series] = {}
        purchase_rate = np.divide(frequency + 1.0, age, out=np.zeros(len(enriched)), where=age > 0)
        shrinkage = (frequency + 1.0) / (frequency + 3.5)
        for horizon in horizons:
            horizon_days = horizon * 30.0
            expected_transactions[horizon] = pd.Series(
                np.clip(alive_probability * purchase_rate * horizon_days * shrinkage, 0, None),
                index=enriched.index,
                dtype=float,
            )
        return "fallback_shrinkage", alive_probability, expected_profit, expected_transactions

    engine = "fallback_shrinkage"
    probability_alive = pd.Series(np.nan, index=enriched.index, dtype=float)
    expected_average_profit = pd.Series(np.nan, index=enriched.index, dtype=float)
    expected_transactions_by_horizon: dict[int, pd.Series] = {}

    can_fit_exact = BetaGeoFitter is not None and valid_mask.sum() >= 40
    if can_fit_exact:
        try:
            # lifetimes puede emitir RuntimeWarning internos al calcular la raiz
            # del hessiano durante el ajuste, incluso cuando el modelo converge.
            with np.errstate(invalid="ignore"):
                bgnbd = BetaGeoFitter(penalizer_coef=penalizer_coef)
                bgnbd.fit(frequency[valid_mask], recency[valid_mask], age[valid_mask])
                probability_alive = pd.Series(
                    bgnbd.conditional_probability_alive(frequency[valid_mask], recency[valid_mask], age[valid_mask]),
                    index=enriched.index[valid_mask],
                    dtype=float,
                ).reindex(enriched.index)

                for horizon in horizons:
                    horizon_days = horizon * 30.0
                    expected_transactions_by_horizon[horizon] = pd.Series(
                        bgnbd.conditional_expected_number_of_purchases_up_to_time(
                            horizon_days,
                            frequency[valid_mask],
                            recency[valid_mask],
                            age[valid_mask],
                        ),
                        index=enriched.index[valid_mask],
                        dtype=float,
                    ).reindex(enriched.index)

            if GammaGammaFitter is not None and repeated_mask.sum() >= 30:
                ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
                ggf.fit(frequency[repeated_mask], monetary[repeated_mask])
                expected_average_profit = pd.Series(
                    ggf.conditional_expected_average_profit(frequency[repeated_mask], monetary[repeated_mask]),
                    index=enriched.index[repeated_mask],
                    dtype=float,
                ).reindex(enriched.index)
                engine = "lifetimes_bgnbd_gammagamma"
            else:
                engine = "lifetimes_bgnbd_fallback_monetary"
        except Exception:
            engine, probability_alive, expected_average_profit, expected_transactions_by_horizon = _fallback_estimation()
    else:
        engine, probability_alive, expected_average_profit, expected_transactions_by_horizon = _fallback_estimation()

    if probability_alive.isna().all() or expected_average_profit.isna().all() or not expected_transactions_by_horizon:
        engine, probability_alive, expected_average_profit, expected_transactions_by_horizon = _fallback_estimation()

    expected_average_profit = expected_average_profit.fillna(
        pd.Series(
            np.where(
                frequency > 0,
                (frequency * monetary + 3.0 * global_monetary) / (frequency + 3.0),
                global_monetary * 0.85,
            ),
            index=enriched.index,
            dtype=float,
        )
    )
    probability_alive = probability_alive.fillna(pd.Series(0.03, index=enriched.index, dtype=float))

    for horizon in horizons:
        horizon_transactions = expected_transactions_by_horizon[horizon].fillna(0.0).clip(lower=0)
        discount_factor = (1.0 + monthly_discount_rate) ** horizon
        clv_column = f"clv_probabilistico_{horizon}m"
        tx_column = f"transacciones_esperadas_{horizon}m"
        enriched[tx_column] = horizon_transactions.round(4)
        enriched[clv_column] = (horizon_transactions * expected_average_profit / discount_factor).round(2)

    enriched["probabilidad_activo_clv"] = probability_alive.clip(lower=0.03, upper=0.995).round(4)
    enriched["valor_monetario_esperado_clv"] = expected_average_profit.round(2)
    enriched["valor_futuro_probabilistico"] = enriched[f"clv_probabilistico_{max(horizons)}m"]
    enriched["motor_clv_probabilistico"] = engine

    model_summary = pd.DataFrame(
        [
            {
                "motor": engine,
                "clientes_validos": int(valid_mask.sum()),
                "clientes_repetidores": int(repeated_mask.sum()),
                "probabilidad_activo_media": round(float(enriched["probabilidad_activo_clv"].mean()), 4),
                f"clv_medio_{max(horizons)}m": round(float(enriched[f"clv_probabilistico_{max(horizons)}m"].mean()), 2),
                "ticket_esperado_medio": round(float(enriched["valor_monetario_esperado_clv"].mean()), 2),
            }
        ]
    )

    interpretation = (
        f"La capa CLV probabilistica utiliza el motor {engine} y deja una proyeccion de valor futuro para {int(valid_mask.sum())} clientes validos. "
        f"La probabilidad media de actividad estimada es {float(enriched['probabilidad_activo_clv'].mean()):.2f}."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "data": enriched,
        "audit": audit,
        "model_summary": model_summary,
        "interpretation": interpretation,
    }


def benchmark_bank_value_models(
    df: pd.DataFrame,
    target_column: str = "clv_valor_observado_12m",
    feature_columns: Sequence[str] | None = None,
    algorithms: Sequence[str] = ("random_forest", "neural_network"),
    catastrophic_error_weight: float = 1.8,
    test_size: float = 0.25,
    random_state: int = 42,
    id_column: str = "cliente_id",
    verbose: bool = True,
) -> dict[str, Any]:
    """Compara modelos continuos para valor/CLV con criterio explicito de negocio.

    Parametros:
        df: Dataset cliente-nivel con target continuo observable.
        target_column: Variable continua que actuara como realidad empirica del benchmark.
        feature_columns: Predictores a usar. Si es None, se infieren evitando ids, fechas y salidas post-modelado.
        algorithms: Algoritmos de regresion a comparar sobre el mismo split holdout.
        catastrophic_error_weight: Cuanto mas costoso es un error extremo frente a un error medio.
            Valores > 1 favorecen RMSE; valores < 1 favorecen MAE.
        test_size: Proporcion del holdout.
        random_state: Semilla para reproducibilidad.
        id_column: Identificador a preservar en la salida enriquecida si existe.
        verbose: Si es True, imprime una lectura ejecutiva.

    Retorna:
        Diccionario con tabla comparativa, champion model, baseline ingenuo, figura y dataset enriquecido.

    Logica estadistica:
        El benchmark usa MAE y RMSE de forma simultanea para evitar seleccionar un modelo solo por
        error medio o solo por castigo a extremos. MAPE se excluye deliberadamente porque en CLV y
        valor bancario puede volverse inestable ante valores bajos, ceros o intermitencia. Para no
        fingir un MASE canonico sin una secuencia temporal homogea, la escala relativa se construye
        contra un baseline ingenuo de mediana sobre train.
    """
    if not algorithms:
        raise ValueError("Se requiere al menos un algoritmo para ejecutar el benchmark continuo.")

    inferred_exclusions = {
        target_column,
        "registro_id",
        "cliente_id",
        "fecha_registro",
        "fecha_corte",
        "abandono",
        "abandono_predicho",
        "probabilidad_abandono",
        "valor_futuro_probabilistico",
        "motor_clv_probabilistico",
        "clv_probabilistico_6m",
        "clv_probabilistico_12m",
        "clv_probabilistico_18m",
        "clv_probabilistico_24m",
        "transacciones_esperadas_6m",
        "transacciones_esperadas_12m",
        "transacciones_esperadas_18m",
        "transacciones_esperadas_24m",
        "valor_monetario_esperado_clv",
        "probabilidad_activo_clv",
        "valor_esperado_contacto",
        "prioridad_integrada",
        "modelo_valor_champion",
        "valor_ml_champion_12m",
        "gap_relativo_valor_ml_vs_clv",
    }
    candidate_features = (
        list(feature_columns)
        if feature_columns is not None
        else [column for column in df.columns if column not in inferred_exclusions]
    )
    _ensure_columns(df, [target_column, *candidate_features])

    working = df[candidate_features + [target_column]].dropna(subset=[target_column]).copy()
    if working.empty:
        raise ValueError("No hay observaciones validas para ejecutar el benchmark continuo.")

    _, _, y_train, y_test = train_test_split(
        working[candidate_features],
        working[target_column],
        test_size=test_size,
        random_state=random_state,
    )
    naive_prediction = np.repeat(float(y_train.median()), len(y_test))
    naive_baseline = pd.DataFrame(
        [
            {
                "modelo": "Baseline ingenuo (mediana train)",
                "mae": round(float(mean_absolute_error(y_test, naive_prediction)), 4),
                "rmse": round(float(mean_squared_error(y_test, naive_prediction) ** 0.5), 4),
                "smape": round(float(_safe_smape(y_test, naive_prediction)), 4),
            }
        ]
    )
    naive_mae = float(naive_baseline.iloc[0]["mae"])

    policy_message, mae_weight, rmse_weight = _resolve_error_policy_weights(catastrophic_error_weight)
    display_names = {
        "random_forest": "Random Forest",
        "neural_network": "Neural Network",
        "gradient_boosting": "Gradient Boosting",
        "linear": "Linear Regression",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elasticnet": "Elastic Net",
    }

    comparison_rows: list[dict[str, Any]] = []
    model_results: dict[str, dict[str, Any]] = {}
    interpretations_by_model: dict[str, str] = {}
    scored_data = df.copy()

    for algorithm in algorithms:
        benchmark_result = train_supervised_model(
            working,
            target=target_column,
            problem_type="regression",
            algorithm=algorithm,
            features=candidate_features,
            test_size=test_size,
            random_state=random_state,
            verbose=False,
        )
        prediction_frame = benchmark_result["predictions"].copy()
        metrics_row = benchmark_result["metrics"].iloc[0].to_dict()
        algorithm_name = display_names.get(algorithm, algorithm)
        smape_value = _safe_smape(prediction_frame["actual"], prediction_frame["predicted"])
        mae_value = float(metrics_row["mae"])
        rmse_value = float(metrics_row["rmse"])
        comparison_rows.append(
            {
                "algoritmo": algorithm,
                "modelo": algorithm_name,
                "mae": round(mae_value, 4),
                "rmse": round(rmse_value, 4),
                "smape": round(smape_value, 4),
                "mae_relativo_baseline": round(float(mae_value / naive_mae), 4) if naive_mae > 0 else np.nan,
                "ratio_rmse_mae": round(float(rmse_value / max(mae_value, 1e-8)), 4),
            }
        )
        model_results[algorithm] = benchmark_result
        interpretations_by_model[algorithm_name] = benchmark_result["interpretation"]
        scored_data[f"prediccion_valor_{algorithm}"] = benchmark_result["pipeline"].predict(df[candidate_features]).round(2)

    comparison = pd.DataFrame(comparison_rows)
    for metric in ["mae", "rmse"]:
        spread = float(comparison[metric].max() - comparison[metric].min())
        comparison[f"{metric}_normalizado"] = 0.0 if spread == 0 else (
            (comparison[metric] - comparison[metric].min()) / spread
        )

    comparison["score_champion"] = (
        comparison["mae_normalizado"] * mae_weight + comparison["rmse_normalizado"] * rmse_weight
    ).round(4)
    comparison = comparison.sort_values(
        ["score_champion", "rmse", "mae", "mae_relativo_baseline"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    champion_row = comparison.iloc[0]
    champion_algorithm = str(champion_row["algoritmo"])
    champion_name = str(champion_row["modelo"])
    comparison["es_champion"] = comparison["modelo"].eq(champion_name)
    scored_data["modelo_valor_champion"] = champion_name
    scored_data["prediccion_valor_champion"] = scored_data[f"prediccion_valor_{champion_algorithm}"]

    best_mae_model = str(comparison.sort_values("mae").iloc[0]["modelo"])
    best_rmse_model = str(comparison.sort_values("rmse").iloc[0]["modelo"])
    if best_mae_model != best_rmse_model:
        divergence_message = (
            f"MAE favorece a {best_mae_model} mientras RMSE favorece a {best_rmse_model}; "
            f"la politica financiera deja como champion a {champion_name}."
        )
    else:
        divergence_message = f"MAE y RMSE convergen sobre {champion_name}, por lo que la seleccion es estable."

    figure_order = comparison.copy().sort_values("score_champion", ascending=False)
    colors = ["#0F766E" if is_champion else "#94A3B8" for is_champion in figure_order["es_champion"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].barh(figure_order["modelo"], figure_order["mae"], color=colors)
    axes[0].set_title("Tablero MAE del benchmark continuo")
    axes[0].set_xlabel("MAE")
    axes[1].barh(figure_order["modelo"], figure_order["rmse"], color=colors)
    axes[1].set_title("Tablero RMSE del benchmark continuo")
    axes[1].set_xlabel("RMSE")
    fig.tight_layout()

    interpretation = (
        f"Se compararon {len(comparison)} modelos continuos sobre {target_column} usando MAE y RMSE de forma simultanea. "
        f"{policy_message} MAPE se excluye deliberadamente del tablero para no contaminar la seleccion con denominadores inestables. "
        f"El champion recomendado es {champion_name} con MAE {float(champion_row['mae']):.4f}, RMSE {float(champion_row['rmse']):.4f} y mejora relativa vs baseline de {float(champion_row['mae_relativo_baseline']):.4f}. "
        f"{divergence_message}"
    )
    _emit_interpretation(interpretation, verbose)

    return {
        "summary": comparison,
        "champion": comparison.head(1).copy(),
        "naive_baseline": naive_baseline,
        "figure": fig,
        "model_results": model_results,
        "interpretations": interpretations_by_model,
        "selection_policy": policy_message,
        "scored_data": scored_data,
        "feature_columns": candidate_features,
        "interpretation": interpretation,
    }


def audit_bi_source(
    df: pd.DataFrame,
    target: str | None = None,
    id_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta la fase de ingesta y auditoria con foco en calidad y representatividad.

    Parametros:
        df: Dataset estructurado a auditar.
        target: Variable objetivo opcional para buscar leakage y balance.
        id_columns: Identificadores de negocio que deben controlarse por unicidad.
        date_columns: Columnas temporales a revisar por cobertura y granularidad.
        verbose: Si es True, imprime una conclusion humana.

    Retorna:
        Diccionario con auditoria base, perfiles de representatividad, cobertura temporal,
        figura de faltantes e interpretacion ejecutiva.

    Logica estadistica:
        Reutiliza reglas de auditoria de calidad y leakage, y anade controles de balance
        y cobertura para evitar conclusiones sesgadas por subrepresentacion operacional.
    """
    audit = audit_dataset(df, target=target, id_columns=id_columns, verbose=False)
    missingness_figure, _ = plot_missingness_heatmap(df)

    representativity = pd.DataFrame()
    target_message = "No se proporciono objetivo; la representatividad se evalua solo sobre estructura y cobertura."
    if target is not None:
        _ensure_columns(df, [target])
        target_series = df[target].dropna()
        if target_series.nunique(dropna=True) <= 10:
            representativity = (
                target_series.value_counts(dropna=False, normalize=True)
                .mul(100)
                .round(2)
                .rename_axis("valor_objetivo")
                .reset_index(name="pct_registros")
            )
            dominant_share = float(representativity["pct_registros"].max()) if not representativity.empty else 0.0
            if dominant_share >= 80:
                target_message = (
                    "La variable objetivo esta fuertemente desbalanceada. Las metricas deben priorizar recall, F1 o calibracion antes que accuracy bruta."
                )
            else:
                target_message = "La distribucion del objetivo no presenta un desbalance extremo para el baseline inicial."
        else:
            representativity = target_series.describe().to_frame(name="valor").reset_index().rename(columns={"index": "metrica"})
            target_message = "El objetivo es continuo; valida estabilidad temporal y cola de errores antes de proyectar a negocio."

    date_rows: list[dict[str, Any]] = []
    for column in date_columns or []:
        _ensure_columns(df, [column])
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().sum() == 0:
            date_rows.append(
                {
                    "columna": column,
                    "fecha_min": None,
                    "fecha_max": None,
                    "dias_cobertura": 0,
                    "pct_fechas_validas": 0.0,
                }
            )
            continue
        date_rows.append(
            {
                "columna": column,
                "fecha_min": parsed.min(),
                "fecha_max": parsed.max(),
                "dias_cobertura": int((parsed.max() - parsed.min()).days),
                "pct_fechas_validas": round(float(parsed.notna().mean() * 100), 2),
            }
        )
    date_audit = pd.DataFrame(date_rows)

    messages = [audit["interpretation"], target_message]
    if not date_audit.empty:
        min_days = int(date_audit["dias_cobertura"].min())
        if min_days < 90:
            messages.append("La cobertura temporal es corta en al menos una columna fecha; las conclusiones pueden ser poco estables.")
        else:
            messages.append("La cobertura temporal observada es razonable para una lectura inicial del negocio.")

    interpretation = " ".join(messages)
    _emit_interpretation(interpretation, verbose)
    return {
        "data": df.copy(),
        "audit": audit,
        "representativity": representativity,
        "date_audit": date_audit,
        "critical_questions": FRAMEWORK_PHASES[0]["preguntas_auditoria_critica"],
        "figures": {"missingness": missingness_figure},
        "interpretation": interpretation,
    }


def execute_bi_etl(
    df: pd.DataFrame,
    features: Sequence[str],
    outlier_columns: Sequence[str] | None = None,
    scaler: str = "robust",
    apply_power_transform: bool = True,
    power_method: str = "yeo-johnson",
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta la fase ETL con limpieza reproducible y preprocesado sin leakage.

    Parametros:
        df: Dataset estructurado a sanear.
        features: Variables que alimentaran el pipeline de modelado.
        outlier_columns: Columnas numericas a tratar por IQR.
        scaler: Tipo de escalado numerico del pipeline.
        apply_power_transform: Si aplica transformaciones de potencia a numericas.
        power_method: Metodo solicitado para PowerTransformer.
        verbose: Si es True, imprime conclusion automatizada.

    Retorna:
        Diccionario con dataset depurado, filas duplicadas eliminadas, tratamiento de outliers,
        preprocesador y narrativa ejecutiva.

    Logica estadistica:
        Controla extremos con reglas robustas y encapsula imputacion, transformacion y escalado
        dentro de sklearn.pipeline para mantener reproducibilidad y evitar fuga de datos.
    """
    _ensure_columns(df, features)
    duplicates_removed = int(df.duplicated().sum())
    deduplicated = df.drop_duplicates().copy()

    outlier_result = handle_outliers(
        deduplicated,
        columns=outlier_columns,
        method="clip_iqr",
        verbose=False,
    )
    preprocessor = build_preprocessing_pipeline(
        outlier_result["data"][list(features)],
        scaler=scaler,
        apply_power_transform=apply_power_transform,
        power_method=power_method,
        verbose=False,
    )

    if duplicates_removed > 0:
        duplicate_message = f"Se eliminaron {duplicates_removed} filas duplicadas antes del saneamiento."
    else:
        duplicate_message = "No se detectaron filas duplicadas para esta fase ETL."

    interpretation = (
        f"{duplicate_message} {outlier_result['interpretation']} {preprocessor['interpretation']} "
        "El preprocesado queda encapsulado en un pipeline reproducible para evitar data leakage."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "data": outlier_result["data"],
        "duplicates_removed": duplicates_removed,
        "outliers": outlier_result,
        "preprocessing": preprocessor,
        "critical_questions": FRAMEWORK_PHASES[1]["preguntas_auditoria_critica"],
        "interpretation": interpretation,
    }


def execute_bi_eda(
    df: pd.DataFrame,
    normality_column: str,
    group_column: str,
    group_value_column: str,
    vif_columns: Sequence[str],
    correlation_pair: Sequence[str] | None = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta la fase EDA con validacion de supuestos y visual analytics critico.

    Parametros:
        df: Dataset estructurado ya saneado.
        normality_column: Variable numerica a someter a diagnostico de normalidad.
        group_column: Variable de agrupacion para contrastes de dispersion y grupos.
        group_value_column: Variable numerica comparada entre grupos.
        vif_columns: Predictores numericos candidatos a revision de multicolinealidad.
        correlation_pair: Par opcional de columnas para correlacion; si no se pasa, se usa
            la primera columna de vif_columns contra normality_column cuando sea posible.
        alpha: Nivel de significacion para pruebas de supuestos.
        verbose: Si es True, imprime conclusion automatizada.

    Retorna:
        Diccionario con estadistica descriptiva, diagnosticos, figuras e interpretacion.

    Logica estadistica:
        Combina Shapiro-Wilk, Levene/Brown-Forsythe, VIF y correlacion para decidir si la
        via parametrica es defendible y si el espacio de variables es estable para inferencia.
    """
    _ensure_columns(df, [normality_column, group_column, group_value_column, *vif_columns])
    selected_pair = list(correlation_pair or [])
    if not selected_pair:
        # Si el usuario no define el par, se toma una combinacion razonable para no romper el flujo exploratorio.
        fallback_x = next((column for column in vif_columns if column != normality_column), vif_columns[0])
        selected_pair = [fallback_x, normality_column]
    _ensure_columns(df, selected_pair)

    normality = check_normality(df[normality_column], alpha=alpha, verbose=False)
    qq_figure, _ = plot_qq_diagnostic(df[normality_column], title=f"Q-Q plot de {normality_column}")
    power_comparison = compare_power_transformations(df[normality_column], verbose=False)
    power_figure, _ = plot_power_transformations(df[normality_column])
    variance = check_variance_homogeneity(
        df,
        value_column=group_value_column,
        group_column=group_column,
        alpha=alpha,
        center="median",
        verbose=False,
    )
    vif = calculate_vif(df, columns=vif_columns, verbose=False)
    correlation = analyze_correlation(
        df,
        x_column=selected_pair[0],
        y_column=selected_pair[1],
        method="auto",
        alpha=alpha,
        verbose=False,
    )
    correlation_figure, _ = grafico_mapa_correlacion(df[list(dict.fromkeys(vif_columns + [group_value_column]))])

    # La interpretacion ejecutiva se compone a partir de cada chequeo de supuestos para dejar rastro de por que se recomienda o no una via parametrica.
    messages = [normality["interpretation"], variance["interpretation"], vif["interpretation"], correlation["interpretation"]]
    recommended_power = power_comparison["recommended"]
    messages.append(
        f"La transformacion candidata mas estable para {normality_column} es '{recommended_power}'."
    )
    interpretation = " ".join(messages)
    _emit_interpretation(interpretation, verbose)
    return {
        "numeric_summary": resumen_numerico(df),
        "categorical_summary": resumen_categorico(df),
        "normality": normality,
        "power_comparison": power_comparison,
        "variance_homogeneity": variance,
        "vif": vif,
        "correlation": correlation,
        "critical_questions": FRAMEWORK_PHASES[2]["preguntas_auditoria_critica"],
        "figures": {
            "qq_plot": qq_figure,
            "power_transformations": power_figure,
            "correlation_heatmap": correlation_figure,
        },
        "interpretation": interpretation,
    }


def explain_bi_model(
    model_result: dict[str, Any],
    top_n: int = 15,
    verbose: bool = True,
) -> dict[str, Any]:
    """Extrae una lectura explicable del modelo usando SHAP si esta disponible.

    Parametros:
        model_result: Salida de train_supervised_model.
        top_n: Numero maximo de variables a devolver en el resumen.
        verbose: Si es True, imprime interpretacion automatizada.

    Retorna:
        Diccionario con metodo usado, tabla resumen e interpretacion ejecutiva.

    Logica estadistica:
        Prioriza SHAP como lectura aditiva local-global cuando la dependencia esta disponible;
        si no, usa permutation importance ya calculada como alternativa model-agnostic.
    """
    feature_importance = model_result["feature_importance"].copy().head(top_n)
    if feature_importance.empty:
        raise ValueError("No hay importancia de variables disponible para explicar el modelo.")

    explanation_method = "permutation_importance"
    summary = feature_importance.rename(columns={"importance_mean": "score", "importance_std": "dispersion"})
    interpretation = (
        "Se usa permutation importance como lectura global del modelo. Es suficiente para priorizacion de variables, "
        "aunque no describe contribuciones locales observacion a observacion."
    )
    consistency_report = pd.DataFrame(
        [
            {
                "espacio_salida": "no_aplica",
                "error_abs_medio": np.nan,
                "error_abs_max": np.nan,
                "reconstruccion_consistente": False,
                "n_observaciones_auditadas": 0,
            }
        ]
    )
    dependence_audit = pd.DataFrame(columns=["feature_a", "feature_b", "abs_spearman", "riesgo_interpretacion"])
    figures = {"summary_plot": None, "dependence_plot": None}
    shap_payload: dict[str, Any] | None = None

    if shap is not None:
        try:
            # SHAP se ejecuta sobre una muestra controlada para mantener trazabilidad y coste computacional razonables en notebooks y CI manual.
            pipeline = model_result["pipeline"]
            preprocessor = pipeline.named_steps["preprocessor"]
            estimator = pipeline.named_steps["model"]
            X_test = model_result["X_test"]

            sample_size = min(250, len(X_test))
            X_sample = X_test.head(sample_size).copy()
            transformed = preprocessor.transform(X_sample)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            feature_names = list(preprocessor.get_feature_names_out())
            transformed_sample = np.asarray(transformed, dtype=float)
            transformed_sample_df = pd.DataFrame(transformed_sample, columns=feature_names, index=X_sample.index)
            explainable_estimator = _FeatureNameAwareEstimatorProxy(estimator, feature_names=feature_names)
            if _is_tree_shap_candidate(estimator):
                explainer = shap.TreeExplainer(estimator)
                shap_values_raw = explainer.shap_values(transformed_sample_df)
                base_values_raw = getattr(explainer, "expected_value", 0.0)
                consistency_estimator = estimator
            else:
                explainer = shap.Explainer(explainable_estimator, transformed_sample_df, feature_names=feature_names)
                shap_values = explainer(transformed_sample_df)
                shap_values_raw = shap_values.values
                base_values_raw = getattr(shap_values, "base_values", getattr(explainer, "expected_value", 0.0))
                consistency_estimator = explainable_estimator
            shap_array = _resolve_shap_matrix(shap_values_raw)
            base_values = _resolve_shap_base_values(base_values_raw, len(transformed_sample))
            reconstructed_output = base_values + shap_array.sum(axis=1)
            consistency_report = _audit_shap_consistency(
                estimator=consistency_estimator,
                transformed_sample=transformed_sample_df,
                problem_type=str(model_result.get("problem_type", "classification")),
                reconstructed_output=reconstructed_output,
            )
            dependence_audit = _audit_feature_dependence(X_sample)

            summary = pd.DataFrame(
                {
                    "feature": feature_names,
                    "score": np.abs(shap_array).mean(axis=0),
                    "dispersion": np.abs(shap_array).std(axis=0),
                    "mean_contribution": shap_array.mean(axis=0),
                }
            ).sort_values("score", ascending=False).head(top_n)
            explanation_method = "shap"
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_array, transformed_sample_df, show=False, max_display=min(top_n, 12))
            figures["summary_plot"] = plt.gcf()
            if figures["summary_plot"].axes:
                figures["summary_plot"].axes[0].set_title("SHAP beeswarm: drivers globales del score")

            top_feature = str(summary.iloc[0]["feature"])
            if transformed_sample_df[top_feature].nunique(dropna=True) > 1:
                plt.figure(figsize=(9, 6))
                shap.dependence_plot(top_feature, shap_array, transformed_sample_df, interaction_index="auto", show=False)
                figures["dependence_plot"] = plt.gcf()
                if figures["dependence_plot"].axes:
                    figures["dependence_plot"].axes[0].set_title(f"SHAP dependence: {top_feature}")

            consistency_row = consistency_report.iloc[0]
            consistency_message = (
                "La suma base value + SHAP reproduce la salida del modelo con error numerico despreciable."
                if bool(consistency_row["reconstruccion_consistente"])
                else "La reconstruccion aditiva no cierra con precision suficiente; conviene revisar la escala explicada o ampliar la muestra auditada."
            )
            dependence_message = (
                "No se detectaron pares numericos con dependencia severa en la muestra auditada."
                if dependence_audit.empty
                else "Aparecen variables numericas correlacionadas en la muestra auditada; evita leer cada contribucion SHAP como evidencia causal aislada."
            )
            interpretation = (
                "La explicabilidad se apoya en SHAP con lectura global y visual de transparencia. "
                f"{consistency_message} {dependence_message}"
            )
            shap_payload = {
                "sample": X_sample,
                "transformed_sample": transformed_sample_df,
                "values": shap_array,
                "base_values": base_values,
                "reconstructed_output": reconstructed_output,
                "top_feature": top_feature,
            }
        except Exception:
            pass

    # Aunque SHAP falle o no exista, la funcion siempre devuelve una lectura valida basada en permutation importance.
    top_features = ", ".join(summary["feature"].head(5).tolist())
    interpretation = f"{interpretation} Variables mas influyentes: {top_features}."
    _emit_interpretation(interpretation, verbose)
    return {
        "method": explanation_method,
        "summary": summary.reset_index(drop=True),
        "consistency_report": consistency_report.reset_index(drop=True),
        "dependence_audit": dependence_audit.reset_index(drop=True),
        "figures": figures,
        "shap_payload": shap_payload,
        "interpretation": interpretation,
    }


def model_bi_baseline(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    problem_type: str = "auto",
    algorithm: str = "auto",
    apply_power_transform: bool = True,
    power_method: str = "yeo-johnson",
    verbose: bool = True,
) -> dict[str, Any]:
    """Entrena un baseline BI reproducible y produce diagnosticos listos para negocio.

    Parametros:
        df: Dataset estructurado ya saneado.
        target: Variable objetivo del modelo.
        features: Predictores a incluir.
        problem_type: Tipo de problema o deteccion automatica.
        algorithm: Algoritmo baseline a emplear.
        apply_power_transform: Si aplica PowerTransformer a numericas.
        power_method: Metodo solicitado para la transformacion de potencia.
        verbose: Si es True, imprime conclusion automatizada.

    Retorna:
        Diccionario con modelo, metricas, explicabilidad, figuras y narrativa ejecutiva.

    Logica estadistica:
        Encadena preprocesado y estimador en sklearn.pipeline, evalua sobre holdout y
        genera una explicacion global del comportamiento del modelo.
    """
    _ensure_columns(df, [target, *features])
    model_result = train_supervised_model(
        df,
        target=target,
        problem_type=problem_type,
        algorithm=algorithm,
        features=features,
        scaler="robust",
        apply_power_transform=apply_power_transform,
        power_method=power_method,
        verbose=False,
    )
    # La capa BI traduce el baseline tecnico a una lectura util para negocio: explicabilidad, figuras y glosario de metricas.
    explanation = explain_bi_model(model_result, verbose=False)
    feature_figure, _ = plot_feature_importance(model_result["feature_importance"])
    diagnostics_figure, _ = plot_model_diagnostics(model_result)

    metrics_row = model_result["metrics"].iloc[0].to_dict()
    metric_translation = pd.DataFrame(
        [
            {
                "metrica": metric_name,
                "valor": metric_value,
                "lectura_ejecutiva": _translate_metric(metric_name, float(metric_value)),
            }
            for metric_name, metric_value in metrics_row.items()
        ]
    )

    interpretation = f"{model_result['interpretation']} {explanation['interpretation']}"
    _emit_interpretation(interpretation, verbose)
    return {
        "model": model_result,
        "explainability": explanation,
        "metric_translation": metric_translation,
        "critical_questions": FRAMEWORK_PHASES[3]["preguntas_auditoria_critica"],
        "figures": {
            "feature_importance": feature_figure,
            "model_diagnostics": diagnostics_figure,
            "shap_summary": explanation["figures"].get("summary_plot"),
            "shap_dependence": explanation["figures"].get("dependence_plot"),
        },
        "interpretation": interpretation,
    }


def evaluate_retention_thresholds(
    model_result: dict[str, Any],
    reference_df: pd.DataFrame | None = None,
    value_column: str | None = None,
    thresholds: Sequence[float] | None = None,
    contact_cost: float = 22.0,
    retention_success_rate: float = 0.28,
    retention_horizon_months: int = 12,
    value_is_period_total: bool | None = None,
    max_contact_share: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua umbrales de retencion con criterio economico y operativo.

    Parametros:
        model_result: Salida del baseline supervisado con probabilidades de clase.
        reference_df: Dataset original para recuperar valor economico por cliente.
        value_column: Columna monetaria a usar como proxy de valor recuperable.
        thresholds: Umbrales a evaluar sobre la probabilidad de abandono.
        contact_cost: Coste unitario de activar una accion de retencion.
        retention_success_rate: Probabilidad esperada de retener un cliente bien priorizado.
        retention_horizon_months: Horizonte de captura de valor esperado.
        value_is_period_total: Indica si la columna monetaria ya representa valor acumulado del
            periodo completo y no debe multiplicarse otra vez por el horizonte.
        max_contact_share: Fraccion maxima de la base que la operacion puede contactar.
        verbose: Si es True, imprime una recomendacion ejecutiva.

    Retorna:
        Diccionario con tabla de escenarios, recomendacion de umbral, figura e interpretacion.

    Logica estadistica:
        Recorre umbrales sobre la probabilidad predicha y combina precision, recall y valor
        economico esperado para escoger un punto operativo defendible en campanas de retencion.
    """
    predictions = model_result["predictions"].copy()
    if "predicted_probability" not in predictions.columns:
        raise ValueError("El modelo no expone probabilidades y no permite evaluar umbrales de retencion.")

    actual = predictions["actual"].astype(int).reset_index(drop=True)
    probabilities = predictions["predicted_probability"].astype(float).reset_index(drop=True)
    evaluated_thresholds = np.array(thresholds if thresholds is not None else np.arange(0.15, 0.86, 0.05), dtype=float)

    if reference_df is not None and value_column is not None:
        _ensure_columns(reference_df, [value_column])
        value_series = (
            reference_df.loc[model_result["X_test"].index, value_column]
            .reset_index(drop=True)
            .fillna(reference_df[value_column].median())
            .astype(float)
        )
    else:
        value_series = pd.Series(np.repeat(1.0, len(predictions)), dtype=float)

    if value_is_period_total is None:
        normalized_name = (value_column or "").lower()
        value_is_period_total = any(token in normalized_name for token in ["_12m", "_6m", "anual", "clv_probabilistico"])
    value_multiplier = 1.0 if value_is_period_total else float(retention_horizon_months)

    scenario_rows: list[dict[str, float]] = []
    total_population = len(actual)
    total_positives = max(int((actual == 1).sum()), 1)

    for threshold in evaluated_thresholds:
        targeted = probabilities >= threshold
        contacted = int(targeted.sum())
        true_positives = int(((actual == 1) & targeted).sum())
        predicted = targeted.astype(int)

        precision = float(precision_score(actual, predicted, zero_division=0)) if contacted > 0 else 0.0
        recall = float(recall_score(actual, predicted, zero_division=0)) if contacted > 0 else 0.0
        f1_value = float(f1_score(actual, predicted, zero_division=0)) if contacted > 0 else 0.0
        recovered_value = float(
            value_series[(actual == 1) & targeted].sum() * retention_success_rate * value_multiplier
        )
        campaign_cost = float(contacted * contact_cost)
        net_value = recovered_value - campaign_cost
        roi = net_value / campaign_cost if campaign_cost > 0 else np.nan

        scenario_rows.append(
            {
                "umbral": round(float(threshold), 2),
                "clientes_contactados": contacted,
                "pct_base_contactada": round(float(contacted / total_population * 100), 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1_value, 4),
                "churners_detectados": true_positives,
                "pct_churners_capturados": round(float(true_positives / total_positives * 100), 2),
                "valor_recuperado_estimado": round(recovered_value, 2),
                "coste_campana": round(campaign_cost, 2),
                "valor_esperado_neto": round(net_value, 2),
                "roi_estimado": round(float(roi), 4) if not np.isnan(roi) else np.nan,
            }
        )

    scenario_report = pd.DataFrame(scenario_rows).sort_values("umbral").reset_index(drop=True)
    eligible_report = scenario_report
    if max_contact_share is not None:
        eligible_report = scenario_report[scenario_report["pct_base_contactada"] <= max_contact_share * 100]
        if eligible_report.empty:
            eligible_report = scenario_report

    best_idx = eligible_report["valor_esperado_neto"].idxmax()
    recommended = scenario_report.loc[[best_idx]].reset_index(drop=True)

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_left.plot(scenario_report["umbral"], scenario_report["valor_esperado_neto"], color="#0F766E", linewidth=2.4)
    ax_left.axvline(float(recommended.loc[0, "umbral"]), color="#B45309", linestyle="--", linewidth=1.6)
    ax_left.set_title("Trade-off entre valor esperado y cobertura de retencion")
    ax_left.set_xlabel("Umbral de probabilidad de abandono")
    ax_left.set_ylabel("Valor esperado neto", color="#0F766E")
    ax_left.tick_params(axis="y", labelcolor="#0F766E")

    ax_right = ax_left.twinx()
    ax_right.plot(scenario_report["umbral"], scenario_report["precision"], color="#1D4ED8", linewidth=2, label="Precision")
    ax_right.plot(scenario_report["umbral"], scenario_report["recall"], color="#DC2626", linewidth=2, label="Recall")
    ax_right.set_ylabel("Metricas operativas", color="#1F2937")
    ax_right.tick_params(axis="y", labelcolor="#1F2937")

    fig.tight_layout()

    best_threshold = float(recommended.loc[0, "umbral"])
    best_net_value = float(recommended.loc[0, "valor_esperado_neto"])
    best_contact_share = float(recommended.loc[0, "pct_base_contactada"])
    best_recall = float(recommended.loc[0, "recall"])
    best_precision = float(recommended.loc[0, "precision"])
    capacity_message = ""
    if max_contact_share is not None:
        capacity_message = f" La recomendacion respeta una capacidad maxima de contacto del {max_contact_share * 100:.2f}% de la base."

    if best_net_value > 0:
        interpretation = (
            f"El umbral recomendado es {best_threshold:.2f} porque maximiza el valor esperado neto bajo los supuestos de campana. "
            f"Implica contactar aproximadamente el {best_contact_share:.2f}% de la base de prueba, con precision {best_precision:.2f} y recall {best_recall:.2f}."
            f"{capacity_message}"
        )
    else:
        interpretation = (
            f"Ningun umbral genera valor esperado neto positivo con un coste unitario de {contact_cost:.2f}. "
            "Conviene renegociar el coste de contacto, focalizar mejor segmentos o revisar la propuesta comercial antes de lanzar la campana."
            f"{capacity_message}"
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "scenario_report": scenario_report,
        "recommended_threshold": recommended,
        "figure": fig,
        "critical_questions": [
            "Que volumen de clientes puede contactar la operacion sin degradar experiencia?",
            "El valor retenido esperado supera el coste comercial y de servicio?",
            "El umbral equilibra captura de churners con saturacion de campanas?",
        ],
        "interpretation": interpretation,
    }


def build_bank_retention_scorecard(
    model_result: dict[str, Any],
    reference_df: pd.DataFrame,
    threshold_result: dict[str, Any],
    segment_column: str = "macro_segmento",
    value_column: str = "rentabilidad_mensual_estimada",
    client_id_column: str = "cliente_id",
    channel_column: str = "canal_servicio",
    contact_cost: float = 22.0,
    retention_success_rate: float = 0.28,
    top_n: int = 25,
    verbose: bool = True,
) -> dict[str, Any]:
    """Construye un scorecard accionable para campanas de retencion bancaria.

    Parametros:
        model_result: Salida del baseline supervisado con probabilidades de clase.
        reference_df: Dataset base para recuperar atributos de negocio del cliente.
        threshold_result: Salida de evaluate_retention_thresholds con umbral recomendado.
        segment_column: Segmento principal para resumir cartera priorizada.
        value_column: Proxy monetaria para estimar prioridad economica.
        client_id_column: Identificador de cliente visible para operaciones.
        channel_column: Canal de servicio para recomendar via de contacto.
        contact_cost: Coste unitario de activar una accion de retencion.
        retention_success_rate: Probabilidad esperada de retener un cliente contactado.
        top_n: Numero de clientes a mostrar en el shortlist ejecutivo.
        verbose: Si es True, imprime una conclusion automatizada.

    Retorna:
        Diccionario con scorecard completo, shortlist priorizado, resumen ejecutivo e interpretacion.

    Logica estadistica:
        Toma el score de abandono del modelo, aplica el umbral recomendado y lo traduce a
        una cartera operativa con semaforos, valor esperado y accion comercial sugerida.
    """
    predictions = model_result["predictions"].copy()
    if "predicted_probability" not in predictions.columns:
        raise ValueError("El modelo no dispone de probabilidades para construir el scorecard de retencion.")

    required_columns = [
        client_id_column,
        segment_column,
        value_column,
        channel_column,
        "nomina_domiciliada",
        "producto_premium",
        "hipoteca_activa",
        "usa_app",
    ]
    optional_clv_columns = [
        "clv_probabilistico_6m",
        "clv_probabilistico_12m",
        "probabilidad_activo_clv",
        "valor_monetario_esperado_clv",
        "transacciones_esperadas_6m",
        "transacciones_esperadas_12m",
        "valor_ml_champion_12m",
        "gap_relativo_valor_ml_vs_clv",
    ]
    required_columns.extend([column for column in optional_clv_columns if column in reference_df.columns])
    required_columns = list(dict.fromkeys(required_columns))
    _ensure_columns(reference_df, required_columns)

    recommended_threshold = float(threshold_result["recommended_threshold"].iloc[0]["umbral"])
    aligned_index = model_result["X_test"].index
    base_frame = reference_df.loc[aligned_index, required_columns].reset_index(drop=True).copy()
    base_frame["abandono_real"] = predictions["actual"].astype(int).reset_index(drop=True)
    base_frame["probabilidad_abandono"] = predictions["predicted_probability"].astype(float).reset_index(drop=True)
    base_frame["abandono_predicho"] = (base_frame["probabilidad_abandono"] >= recommended_threshold).astype(int)

    probability_rank = base_frame["probabilidad_abandono"].rank(pct=True, method="average")
    base_frame["percentil_riesgo"] = (probability_rank * 100).round(2)

    if "clv_probabilistico_12m" in base_frame.columns:
        future_value = base_frame["clv_probabilistico_12m"].fillna(base_frame["clv_probabilistico_12m"].median()).astype(float)
    else:
        future_value = base_frame[value_column].fillna(base_frame[value_column].median()).astype(float)
    if "clv_probabilistico_6m" in base_frame.columns:
        short_term_value = base_frame["clv_probabilistico_6m"].fillna(base_frame["clv_probabilistico_6m"].median()).astype(float)
    else:
        short_term_value = (future_value * 0.48).astype(float)
    alive_probability = (
        base_frame["probabilidad_activo_clv"].fillna(base_frame["probabilidad_activo_clv"].median()).astype(float)
        if "probabilidad_activo_clv" in base_frame.columns
        else pd.Series(np.repeat(0.5, len(base_frame)), index=base_frame.index, dtype=float)
    )
    value_rank = future_value.rank(pct=True, method="average")
    base_frame["percentil_valor_futuro"] = (value_rank * 100).round(2)
    base_frame["prioridad_integrada"] = (
        0.58 * base_frame["percentil_riesgo"] + 0.42 * base_frame["percentil_valor_futuro"]
    ).round(2)
    base_frame["riesgo_recomendado"] = np.select(
        [
            base_frame["probabilidad_abandono"] >= max(recommended_threshold + 0.15, 0.70),
            base_frame["probabilidad_abandono"] >= recommended_threshold,
            base_frame["probabilidad_abandono"] >= max(recommended_threshold - 0.10, 0.20),
        ],
        ["Critico", "Alto", "Medio"],
        default="Bajo",
    )
    base_frame["prioridad_retencion"] = np.select(
        [
            base_frame["riesgo_recomendado"].eq("Critico") & (base_frame["percentil_valor_futuro"] >= 70),
            base_frame["riesgo_recomendado"].isin(["Critico", "Alto"]) & (base_frame["percentil_valor_futuro"] >= 50),
            base_frame["riesgo_recomendado"].isin(["Alto", "Medio"]) & (base_frame["percentil_valor_futuro"] >= 35),
        ],
        ["Maxima", "Alta", "Media"],
        default="Monitoreo",
    )

    contact_channel = np.where(
        base_frame[channel_column].eq("Sucursal")
        | base_frame["producto_premium"].astype(str).str.lower().eq("si")
        | base_frame["prioridad_retencion"].eq("Maxima"),
        "Gestor especializado",
        np.where(base_frame[channel_column].eq("Mixto"), "Llamada consultiva", "Campana digital"),
    )
    action = np.select(
        [
            base_frame["prioridad_retencion"].eq("Maxima") & base_frame["producto_premium"].astype(str).str.lower().eq("si"),
            base_frame["prioridad_retencion"].eq("Maxima"),
            base_frame["prioridad_retencion"].eq("Alta") & base_frame["nomina_domiciliada"].astype(str).str.lower().eq("no"),
            base_frame["riesgo_recomendado"].eq("Alto"),
            base_frame["prioridad_retencion"].eq("Media"),
        ],
        [
            "Escalar a gestor senior con proteccion de CLV y revision integral de relacion",
            "Activar retencion inmediata con foco en valor futuro y llamada prioritaria",
            "Ofrecer vinculacion de nomina y paquete de fidelizacion con horizonte de CLV",
            "Activar campana de retencion estandar con incentivo tactico y seguimiento semanal",
            "Nutrir con comunicaciones preventivas y seguimiento de satisfaccion",
        ],
        default="Mantener monitoreo sin contacto intensivo",
    )
    next_best_offer = np.select(
        [
            base_frame["prioridad_retencion"].eq("Maxima") & base_frame["producto_premium"].astype(str).str.lower().eq("si"),
            base_frame["prioridad_retencion"].eq("Maxima") & base_frame["nomina_domiciliada"].astype(str).str.lower().eq("no"),
            base_frame["prioridad_retencion"].eq("Maxima") & base_frame["hipoteca_activa"].astype(str).str.lower().eq("no"),
            base_frame["riesgo_recomendado"].eq("Alto") & base_frame["usa_app"].astype(str).str.lower().eq("no"),
            base_frame["riesgo_recomendado"].eq("Alto") & base_frame["nomina_domiciliada"].astype(str).str.lower().eq("no"),
            base_frame["riesgo_recomendado"].eq("Alto"),
            base_frame["prioridad_retencion"].eq("Media") & base_frame["producto_premium"].astype(str).str.lower().eq("no"),
            base_frame["prioridad_retencion"].eq("Media"),
        ],
        [
            "Upgrade premium con blindaje de comisiones, gestor dedicado y plan CLV",
            "Bonificacion por domiciliar nomina durante 6 meses",
            "Bundle hipoteca-tarjeta con revision de condiciones",
            "Incentivo de migracion digital y activacion de app",
            "Paquete nomina + cashback de captacion inmediata",
            "Oferta tactica de permanencia con cashback o puntos",
            "Cross-sell de producto premium ligero sin coste inicial",
            "Oferta preventiva de fidelizacion sin descuento agresivo",
        ],
        default="Monitoreo sin oferta economica inmediata",
    )
    base_frame["valor_futuro_probabilistico"] = future_value.round(2)
    base_frame["probabilidad_activo_clv"] = alive_probability.round(4)
    base_frame["valor_esperado_contacto"] = np.where(
        base_frame["abandono_predicho"] == 1,
        (short_term_value * base_frame["probabilidad_abandono"] * retention_success_rate - contact_cost).round(2),
        0.0,
    )
    base_frame["canal_recomendado"] = contact_channel
    base_frame["accion_recomendada"] = action
    base_frame["next_best_offer"] = next_best_offer

    scorecard = base_frame.sort_values(
        ["abandono_predicho", "valor_esperado_contacto", "probabilidad_abandono"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    shortlist = scorecard[scorecard["abandono_predicho"] == 1].head(top_n).copy()
    summary = (
        scorecard.groupby(["riesgo_recomendado", segment_column], observed=False)
        .agg(
            clientes=(client_id_column, "size"),
            score_medio=("probabilidad_abandono", "mean"),
            valor_medio=(value_column, "mean"),
            valor_futuro_medio=("valor_futuro_probabilistico", "mean"),
            valor_esperado_medio=("valor_esperado_contacto", "mean"),
        )
        .reset_index()
        .sort_values(["riesgo_recomendado", "score_medio"], ascending=[True, False])
    )
    for column in ["score_medio", "valor_medio", "valor_futuro_medio", "valor_esperado_medio"]:
        summary[column] = summary[column].round(4 if column == "score_medio" else 2)

    prioritized_clients = int((scorecard["abandono_predicho"] == 1).sum())
    critical_clients = int(scorecard["riesgo_recomendado"].eq("Critico").sum())
    if "clv_probabilistico_12m" in scorecard.columns:
        interpretation = (
            f"El scorecard traduce el modelo a una cartera operativa con {prioritized_clients} clientes priorizados y {critical_clients} en riesgo critico. "
            "La recomendacion combina probabilidad de fuga, CLV probabilistico y canal sugerido para que la operacion pueda defender impacto futuro y no solo valor historico."
        )
    else:
        interpretation = (
            f"El scorecard traduce el modelo a una cartera operativa con {prioritized_clients} clientes priorizados y {critical_clients} en riesgo critico. "
            "La recomendacion combina probabilidad de fuga, valor economico y canal sugerido para que la operacion pueda actuar sin rehacer el analisis manualmente."
        )
    _emit_interpretation(interpretation, verbose)

    return {
        "scorecard": scorecard,
        "shortlist": shortlist,
        "summary": summary,
        "recommended_threshold": recommended_threshold,
        "interpretation": interpretation,
    }


def build_bank_retention_dashboard(
    scorecard_result: dict[str, Any],
    threshold_result: dict[str, Any],
    segment_column: str = "macro_segmento",
    top_offers: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Construye un mini dashboard ejecutivo para la campana de retencion.

    Parametros:
        scorecard_result: Salida de build_bank_retention_scorecard.
        threshold_result: Salida de evaluate_retention_thresholds.
        segment_column: Segmento a usar en la vista agregada.
        top_offers: Numero de ofertas a mostrar en el mix principal.
        verbose: Si es True, imprime una lectura ejecutiva resumida.

    Retorna:
        Diccionario con figura, tablas de soporte e interpretacion.

    Logica estadistica:
        Resume cartera priorizada, mezcla de riesgo, mix de ofertas y valor esperado por
        segmento para facilitar una lectura ejecutiva sin abrir tablas fila a fila.
    """
    scorecard = scorecard_result["scorecard"].copy()
    prioritized = scorecard[scorecard["abandono_predicho"] == 1].copy()
    if prioritized.empty:
        raise ValueError("No hay clientes priorizados para construir el dashboard de retencion.")

    recommended = threshold_result["recommended_threshold"].iloc[0]
    risk_order = ["Critico", "Alto", "Medio", "Bajo"]
    risk_counts = (
        prioritized["riesgo_recomendado"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .reset_index()
        .rename(columns={"index": "riesgo_recomendado", "count": "clientes"})
    )
    offer_mix = (
        prioritized["next_best_offer"]
        .value_counts()
        .head(top_offers)
        .reset_index()
        .rename(columns={"index": "next_best_offer", "count": "clientes"})
    )
    segment_value = (
        prioritized.groupby(segment_column, observed=False)
        .agg(
            clientes=(segment_column, "size"),
            valor_esperado_total=("valor_esperado_contacto", "sum"),
            valor_futuro_total=("valor_futuro_probabilistico", "sum"),
            score_medio=("probabilidad_abandono", "mean"),
        )
        .reset_index()
        .sort_values("valor_esperado_total", ascending=False)
    )
    segment_value["valor_esperado_total"] = segment_value["valor_esperado_total"].round(2)
    segment_value["valor_futuro_total"] = segment_value["valor_futuro_total"].round(2)
    segment_value["score_medio"] = segment_value["score_medio"].round(4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_kpi, ax_risk, ax_offer, ax_segment = axes.flatten()

    ax_kpi.axis("off")
    kpi_lines = [
        "Campana de retencion bancaria",
        f"Umbral recomendado: {float(recommended['umbral']):.2f}",
        f"Clientes priorizados: {int(recommended['clientes_contactados'])}",
        f"Cobertura: {float(recommended['pct_base_contactada']):.2f}%",
        f"Valor esperado neto: {float(recommended['valor_esperado_neto']):.2f}",
        f"CLV futuro priorizado: {float(prioritized['valor_futuro_probabilistico'].sum()):.2f}",
        f"ROI estimado: {float(recommended['roi_estimado']):.2f}",
    ]
    ax_kpi.text(
        0.02,
        0.95,
        "\n".join(kpi_lines),
        va="top",
        ha="left",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    risk_palette = {"Critico": "#B91C1C", "Alto": "#EA580C", "Medio": "#D97706", "Bajo": "#2563EB"}
    ax_risk.bar(
        risk_counts["riesgo_recomendado"],
        risk_counts["clientes"],
        color=[risk_palette.get(label, "#64748B") for label in risk_counts["riesgo_recomendado"]],
    )
    ax_risk.set_title("Clientes priorizados por semaforo")
    ax_risk.set_xlabel("Riesgo")
    ax_risk.set_ylabel("Clientes")

    ax_offer.barh(offer_mix["next_best_offer"], offer_mix["clientes"], color="#0F766E")
    ax_offer.set_title("Mix principal de next best offers")
    ax_offer.set_xlabel("Clientes")
    ax_offer.invert_yaxis()

    ax_segment.bar(segment_value[segment_column], segment_value["valor_esperado_total"], color="#1D4ED8")
    ax_segment.set_title("Valor esperado por segmento priorizado")
    ax_segment.set_xlabel("Segmento")
    ax_segment.set_ylabel("Valor esperado total")

    fig.tight_layout()

    top_segment = str(segment_value.iloc[0][segment_column]) if not segment_value.empty else "N/D"
    interpretation = (
        f"El mini dashboard resume la campana con foco en cartera priorizada, mix de ofertas y valor esperado. "
        f"El segmento con mayor valor esperado agregado es {top_segment}, incorporando tambien el valor futuro probabilistico de la cartera."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "figure": fig,
        "risk_counts": risk_counts,
        "offer_mix": offer_mix,
        "segment_value": segment_value,
        "interpretation": interpretation,
    }


def infer_bi_relationships(
    df: pd.DataFrame,
    group_column: str,
    group_value_column: str,
    ols_features: Sequence[str],
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta contrastes de grupos y OLS robusto para la capa inferencial.

    Parametros:
        df: Dataset estructurado a analizar.
        group_column: Variable de agrupacion para contrastes.
        group_value_column: Variable numerica a comparar y modelar en OLS.
        ols_features: Predictores del modelo lineal inferencial.
        alpha: Nivel de significacion de referencia.
        verbose: Si es True, imprime conclusion automatizada.

    Retorna:
        Diccionario con contraste de grupos, OLS robusto, figura e interpretacion.

    Logica estadistica:
        Selecciona automaticamente el contraste entre grupos segun supuestos y ajusta OLS
        con errores robustos HC3 para proteger la inferencia frente a heterocedasticidad.
    """
    _ensure_columns(df, [group_column, group_value_column, *ols_features])
    group_test = compare_groups(
        df,
        value_column=group_value_column,
        group_column=group_column,
        alpha=alpha,
        verbose=False,
    )
    ols_result = fit_ols_inference(
        df,
        target=group_value_column,
        features=ols_features,
        robust_cov="HC3",
        verbose=False,
    )
    group_figure, _ = plot_group_distributions(df, value_column=group_value_column, group_column=group_column)

    inference_summary = pd.DataFrame(
        [
            {
                "componente": "contraste_grupos",
                "metrica_clave": "p_value",
                "valor": group_test["p_value"],
                "lectura_ejecutiva": _translate_metric("p_value", float(group_test["p_value"])),
            },
            {
                "componente": "modelo_ols",
                "metrica_clave": "r2",
                "valor": ols_result["r_squared"],
                "lectura_ejecutiva": _translate_metric("r2", float(ols_result["r_squared"])),
            },
        ]
    )

    interpretation = (
        f"{group_test['interpretation']} {ols_result['interpretation']} "
        "La lectura inferencial debe cerrarse con tamano del efecto, robustez del modelo y accion operativa asociada."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "group_test": group_test,
        "ols": ols_result,
        "summary": inference_summary,
        "critical_questions": FRAMEWORK_PHASES[4]["preguntas_auditoria_critica"],
        "figures": {"group_distributions": group_figure},
        "interpretation": interpretation,
    }


def build_bi_conclusions(results: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Convierte los resultados tecnicos del pipeline en una sintesis accionable.

    Parametros:
        results: Diccionario consolidado del pipeline universal.
        verbose: Si es True, imprime un cierre ejecutivo.

    Retorna:
        Diccionario con tabla de hallazgos, resumen ejecutivo y preguntas de gobierno.

    Logica estadistica:
        No ejecuta pruebas nuevas; interpreta umbrales de calidad, supuestos, metricas de
        modelo e inferencia para derivar acciones de negocio priorizadas.
    """
    rows: list[dict[str, str]] = []

    audit = results["ingesta_auditoria"]["audit"]
    if not audit["high_missing_columns"].empty:
        rows.append(
            {
                "fase": "Ingesta",
                "hallazgo_clave": "Columnas con nulos relevantes",
                "impacto_bi": "Los KPIs pueden estar sesgados si se imputan sin entender el mecanismo de ausencia.",
                "accion_recomendada": "Priorizar analisis MCAR/MAR/MNAR y documentar reglas de imputacion por dominio.",
            }
        )
    if not audit["leakage_candidates"].empty:
        rows.append(
            {
                "fase": "Ingesta",
                "hallazgo_clave": "Variables con riesgo de leakage",
                "impacto_bi": "El rendimiento observado puede ser artificial y no replicarse fuera de muestra.",
                "accion_recomendada": "Revisar origen temporal y excluir variables que anticipen el objetivo por definicion.",
            }
        )

    vif_report = results["eda"]["vif"]["report"]
    if not vif_report.empty and float(vif_report["vif"].max()) >= 5:
        rows.append(
            {
                "fase": "EDA",
                "hallazgo_clave": "Multicolinealidad relevante",
                "impacto_bi": "Los coeficientes pueden volverse inestables y exagerar hallazgos explicativos.",
                "accion_recomendada": "Reducir redundancias, agrupar features o usar regularizacion antes de escalar decisiones.",
            }
        )

    metrics_row = results["modelado"]["model"]["metrics"].iloc[0].to_dict()
    if "roc_auc" in metrics_row:
        rows.append(
            {
                "fase": "Modelado",
                "hallazgo_clave": f"ROC AUC = {metrics_row['roc_auc']:.4f}",
                "impacto_bi": _translate_metric("roc_auc", float(metrics_row["roc_auc"])),
                "accion_recomendada": "Usar el score para priorizar intervenciones, no como sustituto de criterio de negocio aislado.",
            }
        )
    elif "f1" in metrics_row:
        rows.append(
            {
                "fase": "Modelado",
                "hallazgo_clave": f"F1 = {metrics_row['f1']:.4f}",
                "impacto_bi": _translate_metric("f1", float(metrics_row["f1"])),
                "accion_recomendada": "Revisar umbral operativo y coste de error antes de activarlo en procesos criticos.",
            }
        )
    else:
        rows.append(
            {
                "fase": "Modelado",
                "hallazgo_clave": f"R2 = {metrics_row.get('r2', np.nan):.4f}",
                "impacto_bi": _translate_metric("r2", float(metrics_row.get("r2", np.nan))),
                "accion_recomendada": "Comparar error esperado frente a tolerancia de negocio antes de automatizar forecast.",
            }
        )

    group_test = results["inferencia"]["group_test"]
    rows.append(
        {
            "fase": "Inferencia",
            "hallazgo_clave": f"{group_test['test']} con p-value = {group_test['p_value']:.4f}",
            "impacto_bi": _translate_metric("p_value", float(group_test["p_value"])),
            "accion_recomendada": "Combinar con tamano del efecto y visualizacion por grupo antes de cambiar politica o presupuesto.",
        }
    )

    rows.append(
        {
            "fase": "Gobierno continuo",
            "hallazgo_clave": "El pipeline ya separa ETL, EDA, modelado e inferencia con trazabilidad.",
            "impacto_bi": "La analitica se vuelve repetible, auditable y transferible entre dominios.",
            "accion_recomendada": "Monitorear drift, recalibracion, calidad de dato y versionado de reglas de negocio tras despliegue.",
        }
    )

    summary = pd.DataFrame(rows)
    executive_summary = (
        "El pipeline BI universal prioriza calidad de dato, validacion de supuestos y explicabilidad. "
        "La recomendacion es operar con baselines reproducibles, traducir cada metrica a impacto de negocio "
        "y mantener gobierno continuo sobre fuente, features y rendimiento."
    )
    _emit_interpretation(executive_summary, verbose)
    return {
        "summary": summary,
        "critical_questions": FRAMEWORK_PHASES[5]["preguntas_auditoria_critica"],
        "interpretation": executive_summary,
    }


def run_bi_pipeline_universal(
    df: pd.DataFrame,
    target: str,
    model_features: Sequence[str],
    normality_column: str,
    group_column: str,
    group_value_column: str,
    id_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
    outlier_columns: Sequence[str] | None = None,
    vif_columns: Sequence[str] | None = None,
    inference_features: Sequence[str] | None = None,
    correlation_pair: Sequence[str] | None = None,
    problem_type: str = "auto",
    algorithm: str = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Orquesta el pipeline BI universal desde auditoria hasta conclusiones.

    Parametros:
        df: Dataset estructurado de entrada.
        target: Variable objetivo para modelado.
        model_features: Predictores del modelo supervisado.
        normality_column: Variable numerica a diagnosticar con pruebas de normalidad.
        group_column: Variable categorica para comparacion de grupos.
        group_value_column: Variable numerica usada en comparaciones e inferencia.
        id_columns: Identificadores de negocio a auditar.
        date_columns: Columnas temporales para revisar cobertura.
        outlier_columns: Variables numericas donde aplicar control de extremos.
        vif_columns: Variables numericas para diagnosticar multicolinealidad.
        inference_features: Variables explicativas del modelo OLS robusto.
        correlation_pair: Par de variables para prueba de correlacion.
        problem_type: Tipo de problema supervisado o deteccion automatica.
        algorithm: Algoritmo base para modelado.
        verbose: Si es True, imprime narrativa fase a fase.

    Retorna:
        Diccionario consolidado con resultados tabulares, figuras e interpretaciones por fase.

    Logica estadistica:
        Ejecuta un flujo gobernado por pruebas de supuestos, pipelines reproducibles y una
        capa final de traduccion ejecutiva para convertir salidas analiticas en decisiones.
    """
    _ensure_columns(df, [target, normality_column, group_column, group_value_column, *model_features])

    resolved_vif_columns = list(vif_columns or [column for column in model_features if pd.api.types.is_numeric_dtype(df[column])])
    resolved_inference_features = list(
        inference_features
        or [column for column in model_features if column not in {target, group_value_column}]
    )

    ingestion = audit_bi_source(
        df,
        target=target,
        id_columns=id_columns,
        date_columns=date_columns,
        verbose=verbose,
    )
    etl = execute_bi_etl(
        ingestion["data"],
        features=model_features,
        outlier_columns=outlier_columns,
        scaler="robust",
        apply_power_transform=True,
        power_method="yeo-johnson",
        verbose=verbose,
    )
    eda = execute_bi_eda(
        etl["data"],
        normality_column=normality_column,
        group_column=group_column,
        group_value_column=group_value_column,
        vif_columns=resolved_vif_columns,
        correlation_pair=correlation_pair,
        verbose=verbose,
    )
    modeling = model_bi_baseline(
        etl["data"],
        target=target,
        features=model_features,
        problem_type=problem_type,
        algorithm=algorithm,
        apply_power_transform=True,
        power_method="yeo-johnson",
        verbose=verbose,
    )
    inference = infer_bi_relationships(
        etl["data"],
        group_column=group_column,
        group_value_column=group_value_column,
        ols_features=resolved_inference_features,
        verbose=verbose,
    )

    results = {
        "manual": get_bi_framework_manual(as_dataframe=False),
        "ingesta_auditoria": ingestion,
        "etl": etl,
        "eda": eda,
        "modelado": modeling,
        "inferencia": inference,
    }
    conclusions = build_bi_conclusions(results, verbose=verbose)
    results["conclusiones"] = conclusions
    return results
