"""Servicio de negocio que encapsula la migracion del notebook Banca 360.

La clase separa el paso a paso del notebook original en metodos reutilizables para
que el caso pueda ejecutarse tanto desde notebook como desde CLI sin duplicar codigo.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..core.metodologia import (
    analyze_correlation,
    audit_dataset,
    audit_missingness_mechanism,
    audit_sampling_representativeness,
    build_bandit_policy,
    build_consensus_gap_report,
    build_dataset_data_dictionary,
    build_deep_learning_governance_report,
    calculate_vif,
    check_normality,
    compare_groups,
    detect_simpsons_paradox,
    evaluate_dataset_drift,
    evaluate_probability_calibration,
    fit_ols_inference,
    fractional_difference,
    get_universal_methodology_reference,
    plot_ols_influence_diagnostics,
    plot_probability_calibration,
    report_pipeline_health,
    resolve_business_case_benchmark_models,
    run_bootstrap_prediction_intervals,
    run_fairness_audit,
    run_logistic_parsimony_study,
    run_multiverse_analysis,
    run_purged_temporal_validation,
    run_rfe_feature_selection,
    train_supervised_model,
)
from ..core.framework_bi_universal import (
    benchmark_bank_value_models,
    build_bank_client_case_dataset,
    build_bank_retention_dashboard,
    build_bank_retention_scorecard,
    estimate_bank_probabilistic_clv,
    evaluate_retention_thresholds,
    get_bi_framework_manual,
    get_metric_translation_guide,
    run_bi_pipeline_universal,
)
from ..core.plantilla_pipeline_ciencia_datos import ejecutar_pipeline_metodologico_universal
from ..core.segmentacion_nba import (
    ejecutar_segmentacion_kmeans,
    evaluar_kmeans_opciones,
    perfilar_segmentos,
    plot_dashboard_segmentacion_nba,
)


class Bank360CaseService:
    """Expone cada bloque funcional del caso de negocio como una unidad reusable."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def get_case_contract(self) -> dict[str, Any]:
        """Centraliza columnas, algoritmos y mapa conceptual del caso."""

        contract = {
            "target": self.config.case.target,
            "ols_target": self.config.case.ols_target,
            "business_case": self.config.case.model_routing.business_case,
            "problem_type": self.config.case.model_routing.problem_type,
            "active_benchmark_catalog": self.config.case.model_routing.active_benchmark_catalog,
            "benchmark_catalogs": dict(self.config.case.model_routing.benchmark_catalogs),
            "feature_cols": list(self.config.case.feature_cols),
            "benchmark_models": list(self.config.case.benchmark_models),
            "date_column": self.config.case.forecasting.date_column,
            "forecast_horizon": self.config.case.forecasting.horizon,
            "forecast_lag_count": self.config.case.forecasting.lag_count,
            "forecast_seasonality_period": self.config.case.forecasting.seasonality_period,
            "outlier_cols": list(self.config.case.outlier_cols),
            "vif_cols": list(self.config.case.vif_cols),
            "inference_features": list(self.config.case.inference_features),
            "value_model_features": list(self.config.case.value_model_features),
            "segment_features": list(self.config.case.segment_features),
        }
        mapa_variables = pd.DataFrame(
            {
                "bloque": [
                    "valor",
                    "vinculacion",
                    "comportamiento",
                    "friccion",
                    "canal",
                    "decision",
                ],
                "variables_clave": [
                    "ingreso_mensual, saldo_promedio_3m, valor_cliente_12m, rentabilidad_mensual_estimada",
                    "productos_activos, nomina_domiciliada, hipoteca_activa, producto_premium",
                    "compras_12m, transacciones_app_30d, indice_engagement, promociones_12m",
                    "reclamaciones, satisfaccion, atraso_30d, saldo_variacion_pct_3m",
                    "usa_app, canal_servicio, interacciones_sucursal_90d",
                    "abandono, probabilidad_abandono, probabilidad_activo_clv, clv_probabilistico_12m, valor_esperado_contacto",
                ],
                "lectura_de_negocio": [
                    "Mide cuanto valor esta realmente en juego.",
                    "Refleja profundidad de relacion con el banco.",
                    "Resume intensidad de uso y respuesta comercial.",
                    "Captura senales de desgaste o dolor del cliente.",
                    "Describe por donde conviene activar la retencion.",
                    "Convierte evidencia en priorizacion operativa con valor futuro probabilistico.",
                ],
            }
        )
        return {"contract": contract, "mapa_variables": mapa_variables}

    def _build_dataset_descriptions(self) -> dict[str, str]:
        """Define un glosario operativo por columna para el diccionario de datos v3."""

        return {
            "registro_id": "Identificador tecnico de registro generado para trazabilidad interna del pipeline.",
            "cliente_id": "Identificador unico de cliente visible para scorecards y acciones CRM.",
            "fecha_registro": "Fecha original de alta o captura historica del cliente en la fuente sintetica.",
            "fecha_corte": "Fecha de corte analitico usada para auditoria temporal, drift y ejecucion del caso.",
            "region": "Region geografica del cliente para revisar cobertura y mezcla poblacional.",
            "canal_captacion": "Canal principal de adquisicion del cliente.",
            "canal_servicio": "Canal dominante de atencion o relacion operativa del cliente.",
            "segmento": "Segmento comercial base del cliente.",
            "macro_segmento": "Agrupacion estrategica del cliente para priorizacion, benchmarking y control de representatividad.",
            "edad": "Edad del cliente en anos.",
            "antiguedad_meses": "Meses de relacion acumulada con la entidad.",
            "ingreso_mensual": "Ingreso mensual estimado del cliente utilizado como proxy de capacidad economica.",
            "ingreso_mensual_fracdiff_0_4": "Version con diferenciacion fraccional del ingreso mensual para estabilizar tendencia sin destruir memoria reciente.",
            "gasto_mensual": "Gasto mensual observado o estimado del cliente.",
            "gasto_mensual_fracdiff_0_4": "Version con diferenciacion fraccional del gasto mensual para capturar cambio estructural con memoria parcial.",
            "ticket_medio": "Importe medio por evento transaccional o compra.",
            "margen_estimado": "Margen estimado de contribucion generado por el cliente.",
            "visitas_web_30d": "Interacciones web observadas en los ultimos 30 dias.",
            "compras_12m": "Numero de compras o consumos relevantes en los ultimos 12 meses.",
            "promociones_12m": "Numero de activaciones promocionales o impactos comerciales en 12 meses.",
            "indice_engagement": "Indice sintetico de vinculacion digital y actividad reciente.",
            "satisfaccion": "Valoracion de satisfaccion declarada por el cliente.",
            "reclamaciones": "Conteo de incidencias o reclamaciones registradas.",
            "usa_app": "Indicador de uso de la app movil del banco.",
            "producto_premium": "Indicador de vinculacion con oferta premium.",
            "saldo_promedio_3m": "Saldo promedio de los ultimos tres meses como proxy de profundidad financiera.",
            "saldo_promedio_3m_fracdiff_0_4": "Version con diferenciacion fraccional del saldo medio para preservar memoria y reducir dependencia de tendencia en validaciones temporales.",
            "saldo_variacion_pct_3m": "Variacion porcentual reciente del saldo para capturar deterioro o crecimiento.",
            "productos_activos": "Numero de productos bancarios activos contratados.",
            "ratio_uso_credito": "Proporcion aproximada de uso del limite de credito.",
            "transacciones_app_30d": "Conteo de transacciones digitales realizadas via app en 30 dias.",
            "interacciones_sucursal_90d": "Conteo de interacciones presenciales o asistidas en 90 dias.",
            "nomina_domiciliada": "Indicador de domiciliacion de nomina.",
            "hipoteca_activa": "Indicador de producto hipotecario activo.",
            "atraso_30d": "Indicador de mora o atraso a 30 dias.",
            "rentabilidad_mensual_estimada": "Rentabilidad mensual esperada del cliente para lectura economica e inferencial.",
            "valor_cliente_12m": "Valor acumulado estimado a 12 meses del cliente.",
            "clv_t_dias": "Antiguedad efectiva en dias usada por el motor probabilistico de CLV.",
            "clv_frecuencia": "Frecuencia historica de eventos monetizables para el motor CLV.",
            "clv_t_ultima_compra_dias": "Dias transcurridos hasta la ultima transaccion relevante en el marco CLV.",
            "clv_dias_desde_ultima_transaccion": "Recencia inversa en dias desde la ultima actividad transaccional.",
            "clv_monetario_promedio": "Valor monetario medio por evento usado por el motor de valor futuro.",
            "abandono": "Variable objetivo binaria del caso de retencion; 1 indica riesgo de abandono materializado.",
        }

    def _build_pdf_alignment_reference(self) -> pd.DataFrame:
        """Resume la traduccion de la literatura metodologica a componentes operativos v2."""

        return pd.DataFrame(
            [
                {
                    "principio_pdf": "Muestreo probabilistico y estratificado",
                    "aterrizaje_v2": "Auditoria de representatividad por segmento, alertas de baja cobertura y plan de muestreo recomendado.",
                    "componente": "audit_sampling_representativeness",
                },
                {
                    "principio_pdf": "FAIR, metadata y trazabilidad",
                    "aterrizaje_v2": "Diccionario de datos procesable con roles semanticos, esquema tabular y ownership explicito.",
                    "componente": "build_dataset_data_dictionary",
                },
                {
                    "principio_pdf": "Calidad tabular y reglas CSV reutilizables",
                    "aterrizaje_v2": "Contrato tabular con cabeceras normalizadas, tokens nulos consistentes, unicidad de ids y chequeo de fechas.",
                    "componente": "audit_dataset.tabular_standards",
                },
                {
                    "principio_pdf": "EDA iterativo con tipado correcto y trazabilidad de decisiones",
                    "aterrizaje_v2": "Se mantiene la auditoria estadistica existente y se anade glosario estructurado para justificar variables y transformaciones.",
                    "componente": "build_dataset + run_methodology_validation",
                },
                {
                    "principio_pdf": "Parsimonia con penalizacion explicita",
                    "aterrizaje_v2": "Familia logistica anidada con AIC, BIC y pesos de Akaike para controlar complejidad antes de promover un modelo mas pesado.",
                    "componente": "run_logistic_parsimony_study",
                },
                {
                    "principio_pdf": "Incertidumbre y prediccion por intervalos",
                    "aterrizaje_v2": "Bootstrap multivariable sobre el holdout temporal para dejar intervalos de pronostico operativos y no solo medias puntuales.",
                    "componente": "run_bootstrap_prediction_intervals",
                },
                {
                    "principio_pdf": "Integridad temporal con purga y embargo",
                    "aterrizaje_v2": "Ventanas cronologicas que excluyen observaciones cercanas al bloque de prueba para reducir leakage serial invisible.",
                    "componente": "run_purged_temporal_validation",
                },
                {
                    "principio_pdf": "Gobernanza etica y equidad",
                    "aterrizaje_v2": "Fairness audit con paridad de seleccion, equal opportunity y equalized odds por edad y region; genero se reporta como brecha de dato si falta.",
                    "componente": "run_fairness_audit",
                },
            ]
        )

    def _build_v3_methodology_reference(self) -> pd.DataFrame:
        """Resume los pilares operativos nuevos de la metodologia v3."""

        return pd.DataFrame(
            [
                {
                    "pilar": "Parsimonia",
                    "que_cambia": "La seleccion deja de depender solo del mejor ROC AUC puntual.",
                    "operativizacion": "Benchmark logit anidado con AIC/BIC, pesos de Akaike y tolerancia Occam para promover una especificacion mas simple cuando el gap de rendimiento es pequeno.",
                },
                {
                    "pilar": "Incertidumbre",
                    "que_cambia": "El score deja de leerse como un numero unico y cerrado.",
                    "operativizacion": "Bootstrap multivariable sobre el bloque fuera de muestra para obtener intervalos de pronostico por cliente y resumen de dispersion de cartera.",
                },
                {
                    "pilar": "Temporalidad",
                    "que_cambia": "El holdout temporal se blinda frente a leakage serial y solapamiento cercano.",
                    "operativizacion": "Validacion cronologica con purga previa y embargo posterior por dias sobre fecha_corte.",
                },
                {
                    "pilar": "Consenso y operacion",
                    "que_cambia": "La accion comercial mide si la senal supera al consenso del segmento y prepara un esquema bandit.",
                    "operativizacion": "Brecha de score/valor frente a consenso y politica epsilon-greedy/Thompson para aprender operando.",
                },
                {
                    "pilar": "Equidad y DL",
                    "que_cambia": "La metodologia declara controles eticos y compuertas explicitas para redes neuronales.",
                    "operativizacion": "Fairness audit por grupo y checklist de early stopping, dropout, batch norm, MC Dropout y HPO bayesiana como gate de despliegue DL.",
                },
            ]
        )

    def _add_fractional_memory_features(self, df_bank: pd.DataFrame) -> pd.DataFrame:
        """Agrega features con diferenciacion fraccional para blindar senal temporal."""

        working = df_bank.copy()
        working["fecha_corte"] = pd.to_datetime(working["fecha_corte"], errors="coerce")
        ordered = working.sort_values(["fecha_corte", "cliente_id"]).copy()
        ordered["ingreso_mensual_fracdiff_0_4"] = fractional_difference(ordered["ingreso_mensual"], d=0.4)
        ordered["saldo_promedio_3m_fracdiff_0_4"] = fractional_difference(ordered["saldo_promedio_3m"], d=0.4)
        ordered["gasto_mensual_fracdiff_0_4"] = fractional_difference(ordered["gasto_mensual"], d=0.4)
        return ordered.sort_index()

    def build_context(self) -> dict[str, Any]:
        """Construye el marco metodologico reusable del proyecto."""

        manual_bi = get_bi_framework_manual(as_dataframe=True)
        guia_metricas = get_metric_translation_guide()
        referencia_metodologica = get_universal_methodology_reference(verbose=False)
        manual_resumen = manual_bi[["fase", "objetivo", "estandares"]].copy()
        referencia_frameworks = referencia_metodologica["frameworks"].copy()
        alineacion_metodologica_pdf = self._build_pdf_alignment_reference()
        metodologia_v3_resumen = self._build_v3_methodology_reference()
        return {
            "manual_bi": manual_bi,
            "guia_metricas": guia_metricas,
            "referencia_metodologica": referencia_metodologica,
            "manual_resumen": manual_resumen,
            "referencia_frameworks": referencia_frameworks,
            "alineacion_metodologica_pdf": alineacion_metodologica_pdf,
            "metodologia_v3_resumen": metodologia_v3_resumen,
        }

    def build_dataset(self) -> dict[str, Any]:
        """Genera el dataset bancario y ejecuta la auditoria inicial."""

        contract = self.get_case_contract()["contract"]
        df_bank = build_bank_client_case_dataset(
            n_registros=self.config.case.dataset_rows,
            semilla=self.config.seed,
        )
        df_bank = self._add_fractional_memory_features(df_bank)
        auditoria_inicial = audit_dataset(
            df_bank,
            target=contract["target"],
            id_columns=["registro_id", "cliente_id"],
            date_columns=["fecha_registro", "fecha_corte"],
            segment_columns=["region", "segmento", "macro_segmento", "canal_captacion", "canal_servicio"],
            verbose=False,
        )
        diccionario_datos = build_dataset_data_dictionary(
            df_bank,
            dataset_name="bank360_clientes_v3",
            descriptions=self._build_dataset_descriptions(),
            id_columns=["registro_id", "cliente_id"],
            target=contract["target"],
            date_columns=["fecha_registro", "fecha_corte"],
            source_system="banca_360_metodologia_v3",
            owner="senior_analytics_engineering",
            verbose=False,
        )
        resumen_numerico = (
            df_bank.select_dtypes(include=np.number)
            .describe()
            .T
            .reset_index()
            .rename(columns={"index": "variable"})
        )
        resumen_categorico = pd.DataFrame(
            {
                "variable": df_bank.select_dtypes(exclude=np.number).columns,
                "n_categorias": [
                    df_bank[col].nunique(dropna=False)
                    for col in df_bank.select_dtypes(exclude=np.number).columns
                ],
                "moda": [
                    df_bank[col].mode(dropna=False).iloc[0]
                    if not df_bank[col].mode(dropna=False).empty
                    else np.nan
                    for col in df_bank.select_dtypes(exclude=np.number).columns
                ],
            }
        )
        distribucion_objetivo = (
            df_bank[contract["target"]]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .rename_axis(contract["target"])
            .reset_index(name="pct_clientes")
        )
        summary_row = auditoria_inicial["summary"].iloc[0]
        return {
            "data": df_bank,
            "auditoria_inicial": auditoria_inicial,
            "tabular_standards": auditoria_inicial["tabular_standards"],
            "sampling_audit": auditoria_inicial["sampling_audit"],
            "data_dictionary": diccionario_datos,
            "resumen_numerico": resumen_numerico,
            "resumen_categorico": resumen_categorico,
            "distribucion_objetivo": distribucion_objetivo,
            "summary": {
                "filas": int(summary_row["filas"]),
                "columnas": int(summary_row["columnas"]),
                "pct_abandono": float(df_bank[contract["target"]].mean() * 100),
                "max_missing_pct": float(auditoria_inicial["missing_report"]["pct_nulos"].max()),
                "leakage_count": int(len(auditoria_inicial["leakage_candidates"])),
                "tabular_alerts": int(summary_row["alertas_tabulares"]),
                "sampling_alerts": int(summary_row["alertas_representatividad"]),
                "fracdiff_features": 3,
                "interpretation": auditoria_inicial["interpretation"],
            },
        }

    def run_benchmark(self, df_bank: pd.DataFrame) -> dict[str, Any]:
        """Compara algoritmos base para elegir el motor del score de churn."""

        contract = self.get_case_contract()["contract"]
        benchmark_models = list(
            resolve_business_case_benchmark_models(
                business_case=contract["business_case"],
                benchmark_models=contract["benchmark_models"],
            )
        )
        benchmark_rows: list[dict[str, Any]] = []
        benchmark_interpretations: dict[str, str] = {}
        skipped_rows: list[dict[str, Any]] = []
        complexity_rank = {
            "linear": 1,
            "lasso": 2,
            "logistic": 2,
            "ridge": 3,
            "elasticnet": 4,
            "knn": 5,
            "gam": 5,
            "mars": 6,
            "gradient_boosting": 7,
            "random_forest": 8,
            "xgboost": 9,
            "lightgbm": 9,
            "catboost": 10,
            "mlp": 11,
            "neural_network": 11,
            "arima": 4,
            "prophet": 5,
            "lstm": 12,
        }
        problem_type = str(contract["problem_type"])
        for algorithm in benchmark_models:
            try:
                benchmark_result = train_supervised_model(
                    df_bank,
                    target=contract["target"],
                    problem_type=problem_type,
                    algorithm=algorithm,
                    features=contract["feature_cols"],
                    test_size=self.config.case.test_size,
                    random_state=self.config.seed,
                    date_column=contract["date_column"],
                    business_case=contract["business_case"],
                    forecast_lags=contract["forecast_lag_count"],
                    forecast_horizon=contract["forecast_horizon"],
                    seasonality_period=contract["forecast_seasonality_period"],
                    verbose=False,
                )
            except Exception as exc:
                skipped_rows.append({"modelo": algorithm, "motivo": str(exc)})
                continue
            metrics = benchmark_result["metrics"].iloc[0].to_dict()
            row = {
                "modelo": algorithm,
                "complexity_rank": complexity_rank.get(algorithm, 99),
            }
            if problem_type == "classification":
                row.update(
                    {
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "roc_auc": metrics.get("roc_auc"),
                        "log_loss": metrics.get("log_loss"),
                    }
                )
            else:
                row.update(
                    {
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                        "error_medio": metrics.get("error_medio"),
                    }
                )
            benchmark_rows.append(row)
            benchmark_interpretations[algorithm] = benchmark_result["interpretation"]

        benchmark_df = pd.DataFrame(benchmark_rows)
        if benchmark_df.empty:
            skipped_df = pd.DataFrame(skipped_rows)
            raise ValueError(
                "No se pudo entrenar ningun modelo del benchmark activo. "
                + ("Modelos omitidos: " + "; ".join(skipped_df["modelo"].astype(str).tolist()) if not skipped_df.empty else "")
            )

        if problem_type == "classification":
            benchmark_df = benchmark_df.sort_values(
                ["roc_auc", "f1", "complexity_rank"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
        else:
            benchmark_df = benchmark_df.sort_values(
                ["rmse", "mae", "complexity_rank"],
                ascending=[True, True, True],
            ).reset_index(drop=True)

        champion_row = benchmark_df.iloc[0]
        runner_up = benchmark_df.iloc[1] if len(benchmark_df) > 1 else champion_row
        selected_features = list(contract["feature_cols"])
        champion_model = str(champion_row["modelo"])
        occam_promoted = False
        if problem_type == "classification":
            parsimony = run_logistic_parsimony_study(
                df_bank,
                target=contract["target"],
                features=contract["feature_cols"],
                feature_steps=self.config.case.parsimony.logistic_feature_steps,
                test_size=self.config.case.test_size,
                random_state=self.config.seed,
                verbose=False,
            )
            parsimonious_row = parsimony["summary"].iloc[0]
            selection_policy = (
                f"Se mantiene {champion_model} por liderazgo en ROC AUC y F1 dentro del benchmark guiado por '{contract['active_benchmark_catalog']}'."
            )
            performance_gap = float(champion_row["roc_auc"] - parsimonious_row["roc_auc"])
            if performance_gap <= self.config.case.parsimony.roc_auc_tolerance:
                champion_model = "logistic"
                selected_features = list(parsimony["recommended_features"])
                occam_promoted = str(champion_row["modelo"]) != "logistic"
                selection_policy = (
                    "La politica Occam v4 promueve una especificacion logistica mas simple porque su gap de ROC AUC queda dentro de la tolerancia definida y el BIC favorece menor complejidad."
                )
            lead_gap_primary_metric = float(champion_row["roc_auc"] - runner_up["roc_auc"])
        else:
            linear_candidates = benchmark_df[benchmark_df["modelo"].isin(["linear", "lasso"])]
            selection_policy = (
                f"Se mantiene {champion_model} por menor RMSE dentro del benchmark guiado por '{contract['active_benchmark_catalog']}'."
            )
            parsimony = {
                "summary": pd.DataFrame(),
                "interpretation": selection_policy,
                "recommended_features": selected_features,
            }
            if not linear_candidates.empty:
                best_linear = linear_candidates.sort_values(["rmse", "mae", "complexity_rank"]).iloc[0]
                rmse_tolerance = max(float(champion_row["rmse"]) * self.config.case.parsimony.roc_auc_tolerance, 1e-6)
                rmse_gap = float(best_linear["rmse"] - champion_row["rmse"])
                if rmse_gap <= rmse_tolerance:
                    champion_model = str(best_linear["modelo"])
                    occam_promoted = champion_model != str(champion_row["modelo"])
                    selection_policy = (
                        "La politica de parsimonia v4 prioriza un modelo lineal penalizado porque su RMSE queda dentro de la tolerancia definida y reduce complejidad operativa."
                    )
                    parsimony = {
                        "summary": pd.DataFrame(
                            [
                                {
                                    "modelo_promovido": champion_model,
                                    "rmse_gap_vs_champion": round(rmse_gap, 6),
                                    "rmse_tolerancia": round(rmse_tolerance, 6),
                                }
                            ]
                        ),
                        "interpretation": selection_policy,
                        "recommended_features": selected_features,
                    }
            lead_gap_primary_metric = float(runner_up["rmse"] - champion_row["rmse"])
        benchmark_df["selected_for_v4"] = benchmark_df["modelo"].eq(champion_model)
        return {
            "benchmark_df": benchmark_df,
            "skipped_models": pd.DataFrame(skipped_rows),
            "benchmark_interpretations": benchmark_interpretations,
            "model_name": champion_model,
            "selected_features": selected_features,
            "parsimonia": parsimony,
            "selection_policy": selection_policy,
            "occam_promoted": occam_promoted,
            "lead_gap_primary_metric": lead_gap_primary_metric,
        }

    def run_bi_layer(self, df_bank: pd.DataFrame, benchmark_result: dict[str, Any]) -> dict[str, Any]:
        """Ejecuta la capa BI reusable que traduce el caso a narrativa ejecutiva."""

        contract = self.get_case_contract()["contract"]
        model_name = str(benchmark_result["model_name"])
        selected_features = list(benchmark_result.get("selected_features", contract["feature_cols"]))
        bi_result = run_bi_pipeline_universal(
            df=df_bank,
            target=contract["target"],
            model_features=selected_features,
            normality_column="saldo_promedio_3m",
            group_column="macro_segmento",
            group_value_column=contract["ols_target"],
            id_columns=["registro_id", "cliente_id"],
            date_columns=["fecha_corte"],
            outlier_columns=contract["outlier_cols"],
            vif_columns=contract["vif_cols"],
            inference_features=contract["inference_features"],
            correlation_pair=["ingreso_mensual", "saldo_promedio_3m"],
            problem_type="classification",
            algorithm=model_name,
            verbose=False,
        )
        bi_metrics = bi_result["modelado"]["model"]["metrics"].iloc[0].to_dict()
        return {
            "bi_result": bi_result,
            "metrics": bi_metrics,
            "selected_features": selected_features,
            "selection_policy": benchmark_result.get("selection_policy"),
            "vif_max": float(bi_result["eda"]["vif"]["report"]["vif"].max()),
            "correlation": bi_result["eda"]["correlation"],
            "group_test": bi_result["inferencia"]["group_test"],
        }

    def run_methodology_validation(self, df_bank: pd.DataFrame, benchmark_result: dict[str, Any]) -> dict[str, Any]:
        """Reproduce la validacion rigurosa del notebook en una unidad reusable."""

        contract = self.get_case_contract()["contract"]
        model_name = str(benchmark_result["model_name"])
        selected_features = list(benchmark_result.get("selected_features", contract["feature_cols"]))
        fecha_referencia = pd.to_datetime(df_bank["fecha_corte"]).quantile(0.65)
        reference_df = df_bank[pd.to_datetime(df_bank["fecha_corte"]) <= fecha_referencia].copy()
        current_df = df_bank[pd.to_datetime(df_bank["fecha_corte"]) > fecha_referencia].copy()
        governance_audit = audit_dataset(
            df_bank,
            target=contract["target"],
            id_columns=["registro_id", "cliente_id"],
            date_columns=["fecha_registro", "fecha_corte"],
            segment_columns=["region", "segmento", "macro_segmento", "canal_captacion", "canal_servicio"],
            verbose=False,
        )
        missingness_audit = audit_missingness_mechanism(
            df_bank,
            columns=[*selected_features, contract["target"]],
            verbose=False,
        )
        drift_report = evaluate_dataset_drift(
            reference_df,
            current_df,
            columns=[*selected_features, contract["target"]],
            verbose=False,
        )
        health_validation_columns = [
            column for column in selected_features if not str(column).endswith("_fracdiff_0_4")
        ]
        validation_failed_rows = int(df_bank[health_validation_columns + [contract["target"]]].isna().any(axis=1).sum())
        health_config = self.config.case.pipeline_health
        health_report = report_pipeline_health(
            dataset_name="caso_banca_360",
            updated_at=pd.to_datetime(df_bank["fecha_corte"]).max(),
            expected_rows=len(reference_df),
            observed_rows=len(current_df),
            validation_failed_rows=validation_failed_rows,
            total_validated_rows=len(df_bank),
            freshness_threshold_hours=health_config.freshness_threshold_hours,
            count_tolerance_pct=health_config.count_tolerance_pct,
            critical_freshness_multiplier=health_config.critical_freshness_multiplier,
            critical_count_tolerance_pct=health_config.critical_count_tolerance_pct,
            validation_error_alert_pct=health_config.validation_error_alert_pct,
            validation_error_critical_pct=health_config.validation_error_critical_pct,
            verbose=False,
        )
        health_summary = health_report["summary"].iloc[0]
        health_decision = str(health_summary["decision_operativa"]).strip().lower()
        modeling_enabled = health_decision != "blocked"
        execution_gate = pd.DataFrame(
            [
                {
                    "dataset": str(health_summary["dataset"]),
                    "pipeline_health_profile": health_config.profile_name,
                    "severidad_global": str(health_summary["severidad_global"]),
                    "decision_operativa": str(health_summary["decision_operativa"]),
                    "alertas_activas": str(health_summary["alertas_activas"]),
                    "issues_criticos": int(health_summary["issues_criticos"]),
                    "modeling_enabled": modeling_enabled,
                    "accion_recomendada": str(health_summary["accion_recomendada"]),
                }
            ]
        )
        if modeling_enabled:
            rfe_report = run_rfe_feature_selection(
                df_bank,
                target=contract["target"],
                features=selected_features,
                problem_type="classification",
                n_features_to_select=min(6, len(selected_features)),
                verbose=False,
            )
        else:
            rfe_report = {
                "ranking": pd.DataFrame(columns=["feature", "ranking", "selected"]),
                "interpretation": "El ranking RFE se omitio porque la politica de salud del pipeline bloqueo el modelado.",
            }
        normalidad = check_normality(df_bank["saldo_promedio_3m"], verbose=False)
        normalidad_pruebas = normalidad["tests_table"].copy()
        normalidad_resumen = normalidad["shape_summary"].copy()
        normalidad_fila = normalidad_resumen.iloc[0]
        anderson_fila = normalidad_pruebas.loc[
            normalidad_pruebas["prueba"] == "Anderson-Darling"
        ].iloc[0]
        vif_report = calculate_vif(df_bank, columns=contract["vif_cols"], verbose=False)
        correlacion = analyze_correlation(
            df_bank,
            x_column="ingreso_mensual",
            y_column="saldo_promedio_3m",
            verbose=False,
        )
        simpson = detect_simpsons_paradox(
            df_bank,
            x_column="ingreso_mensual",
            y_column=contract["ols_target"],
            group_column="macro_segmento",
            verbose=False,
        )
        comparacion_segmentos = compare_groups(
            df_bank,
            value_column=contract["ols_target"],
            group_column="macro_segmento",
            verbose=False,
        )
        ols_result = fit_ols_inference(
            df_bank,
            target=contract["ols_target"],
            features=contract["inference_features"],
            verbose=False,
        )
        multiverse_result = run_multiverse_analysis(
            df_bank,
            target=contract["target"],
            features=selected_features,
            problem_type="classification",
            numeric_outlier_columns=contract["outlier_cols"],
            verbose=False,
        )
        metodologia_result = None
        calibration_result = {
            "metrics": pd.DataFrame(
                [
                    {
                        "brier_score": np.nan,
                        "n_bins": 0,
                        "brier_score_raw": np.nan,
                        "brier_score_delta": np.nan,
                        "calibration_method": "not_run",
                    }
                ]
            ),
            "calibration_curve": pd.DataFrame(columns=["estimated_probability", "observed_frequency"]),
            "bin_summary": pd.DataFrame(columns=["bin", "n", "mean_probability", "observed_rate", "calibration_gap"]),
            "comparison": pd.DataFrame(columns=["method", "brier_score", "brier_score_delta", "applied", "selected_for_scoring", "calibration_rows", "note"]),
            "interpretation": "La calibracion no se ejecuto porque la politica de salud del pipeline bloqueo el modelado.",
        }
        calibration_figure = None
        temporal_validation = {
            "summary": pd.DataFrame(
                [{"n_folds": 0, "mean_roc_auc": np.nan, "std_roc_auc": np.nan, "mean_log_loss": np.nan, "mean_brier_score": np.nan}]
            ),
            "fold_report": pd.DataFrame(),
            "interpretation": "La validacion temporal purgada no se ejecuto porque el modelado quedo bloqueado.",
        }
        uncertainty = {
            "prediction_intervals": pd.DataFrame(),
            "portfolio_summary": pd.DataFrame(
                [{"bootstrap_iterations_effective": 0, "mean_probability_base": np.nan, "mean_interval_width": np.nan, "high_risk_share_p05": np.nan, "high_risk_share_p95": np.nan}]
            ),
            "interpretation": "La incertidumbre bootstrap no se ejecuto porque el modelado quedo bloqueado.",
        }
        fairness_audit = {
            "summary": pd.DataFrame([{"sensitive_feature": "not_run", "status": "not_run"}]),
            "group_metrics": pd.DataFrame(),
            "interpretation": "La auditoria de equidad no se ejecuto porque el modelado quedo bloqueado.",
        }
        if modeling_enabled:
            metodologia_result = ejecutar_pipeline_metodologico_universal(
                df=df_bank,
                objetivo=contract["target"],
                features_modelo=selected_features,
                columna_fecha=contract["date_column"],
                columna_normalidad="saldo_promedio_3m",
                columna_grupo="macro_segmento",
                columna_valor_grupo=contract["ols_target"],
                columnas_outliers=contract["outlier_cols"],
                columnas_vif=contract["vif_cols"],
                id_columns=["registro_id", "cliente_id"],
                problema_modelado="classification",
                algoritmo_modelado=model_name,
            )
            temporal_validation = run_purged_temporal_validation(
                df_bank,
                target=contract["target"],
                features=selected_features,
                date_column=contract["date_column"],
                algorithm=model_name,
                n_splits=self.config.case.temporal_validation.n_splits,
                purge_gap_days=self.config.case.temporal_validation.purge_gap_days,
                embargo_gap_days=self.config.case.temporal_validation.embargo_gap_days,
                random_state=self.config.seed,
                verbose=False,
            )
            uncertainty = run_bootstrap_prediction_intervals(
                df_bank,
                target=contract["target"],
                features=selected_features,
                date_column=contract["date_column"],
                algorithm=model_name,
                n_iterations=self.config.case.uncertainty.bootstrap_iterations,
                alpha=self.config.case.uncertainty.prediction_interval_alpha,
                random_state=self.config.seed,
                verbose=False,
            )
            fairness_audit = run_fairness_audit(
                df_bank,
                metodologia_result["modelado"],
                target=contract["target"],
                sensitive_columns=self.config.case.fairness.sensitive_columns,
                age_bins=self.config.case.fairness.age_bins,
                verbose=False,
            )
        plt.close("all")
        ols_figure, _ = plot_ols_influence_diagnostics(ols_result)
        if modeling_enabled and metodologia_result is not None:
            calibration_result = evaluate_probability_calibration(metodologia_result["modelado"], verbose=False)
            calibration_figure, _ = plot_probability_calibration(calibration_result)
        deep_learning_governance = build_deep_learning_governance_report(
            deep_learning_enabled=self.config.case.deep_learning.enabled,
            require_dropout=self.config.case.deep_learning.require_dropout,
            require_batch_norm=self.config.case.deep_learning.require_batch_norm,
            require_early_stopping=self.config.case.deep_learning.require_early_stopping,
            require_bayesian_optimization=self.config.case.deep_learning.require_bayesian_optimization,
            require_mc_dropout=self.config.case.deep_learning.require_mc_dropout,
            active_algorithms=[model_name],
            verbose=False,
        )
        mcar_test = missingness_audit["little_mcar"].iloc[0]
        drift_summary = drift_report["summary"].iloc[0]
        top_vif = float(vif_report["report"]["vif"].max())
        top_condition_index = float(vif_report["scaled_condition_number"])
        critical_belsley_count = int(len(vif_report["critical_components"]))
        normalidad_accion = str(normalidad_fila["accion_recomendada"])
        calibration_metrics = calibration_result["metrics"].iloc[0]
        tabular_summary = governance_audit["tabular_standards"]["summary"].iloc[0]
        sampling_alerts = governance_audit["sampling_audit"]["alerts"]
        temporal_summary = temporal_validation["summary"].iloc[0]
        uncertainty_summary = uncertainty["portfolio_summary"].iloc[0]
        fairness_summary = fairness_audit["summary"]
        fairness_gap_columns = [
            column
            for column in ["demographic_parity_difference", "equal_opportunity_difference", "equalized_odds_difference"]
            if column in fairness_summary.columns
        ]
        fairness_max_gap = float(
            np.nanmax(fairness_summary[fairness_gap_columns].to_numpy(dtype=float))
        ) if fairness_gap_columns and not fairness_summary.empty else float("nan")
        dl_required_gaps = int((deep_learning_governance["checklist"]["status"] == "gap").sum())
        risk_flags = pd.DataFrame(
            [
                {
                    "dimension": "Missingness",
                    "estado": "Alerta" if not bool(mcar_test["is_mcar_compatible"]) else "OK",
                    "implicacion": "Las ausencias no lucen plenamente MCAR; conviene sostener imputacion y monitoreo por causa."
                    if not bool(mcar_test["is_mcar_compatible"])
                    else "No hay evidencia fuerte de sesgo por faltantes en esta corrida.",
                },
                {
                    "dimension": "Normalidad estructural",
                    "estado": "Seguimiento" if not bool(normalidad["is_normal"]) else "OK",
                    "implicacion": normalidad_accion,
                },
                {
                    "dimension": "Parsimonia",
                    "estado": "OK" if len(selected_features) < len(contract["feature_cols"]) else "Seguimiento",
                    "implicacion": str(benchmark_result.get("selection_policy", "No se informo politica de seleccion.")),
                },
                {
                    "dimension": "Drift PSI",
                    "estado": "OK" if int(drift_summary["columnas_con_drift_severo"]) == 0 else "Alerta",
                    "implicacion": "La distribucion luce estable entre ventanas."
                    if int(drift_summary["columnas_con_drift_severo"]) == 0
                    else "Hay columnas que exigen revision antes de escalar el score.",
                },
                {
                    "dimension": "Salud pipeline",
                    "estado": "Bloqueado" if health_decision == "blocked" else "Alerta" if health_decision == "degraded" else "OK",
                    "implicacion": str(health_summary["accion_recomendada"]),
                },
                {
                    "dimension": "Validacion temporal purgada",
                    "estado": "OK"
                    if int(temporal_summary.get("n_folds", 0)) > 0 and float(temporal_summary.get("std_roc_auc", np.inf)) <= 0.06
                    else "Seguimiento",
                    "implicacion": temporal_validation["interpretation"],
                },
                {
                    "dimension": "Calibracion",
                    "estado": "Bloqueado"
                    if not modeling_enabled
                    else "Seguimiento"
                    if calibration_metrics["brier_score"] > 0.16
                    else "OK",
                    "implicacion": "La calibracion se omitio porque la salud del pipeline no permite entrenar o recalibrar con seguridad."
                    if not modeling_enabled
                    else "El score ordena bien, pero la probabilidad aun requiere monitoreo fino por bins."
                    if calibration_metrics["brier_score"] > 0.16
                    else "La calibracion es suficiente para soportar umbrales operativos.",
                },
                {
                    "dimension": "Incertidumbre",
                    "estado": "OK"
                    if float(uncertainty_summary.get("mean_interval_width", np.inf)) <= 0.25
                    else "Seguimiento",
                    "implicacion": uncertainty["interpretation"],
                },
                {
                    "dimension": "Estabilidad coeficientes",
                    "estado": "OK"
                    if top_vif < 5 and top_condition_index < 30 and critical_belsley_count == 0 and not bool(simpson["paradox_detected"])
                    else "Seguimiento",
                    "implicacion": "La estabilidad de coeficientes y la lectura estructural no muestran senales fuertes de colinealidad, inestabilidad matricial o paradojas agregadas."
                    if top_vif < 5 and top_condition_index < 30 and critical_belsley_count == 0 and not bool(simpson["paradox_detected"])
                    else "Conviene revisar la lectura causal agregada antes de extrapolar decisiones.",
                },
                {
                    "dimension": "Contrato tabular",
                    "estado": "OK" if int(tabular_summary.sum()) == 0 else "Seguimiento",
                    "implicacion": governance_audit["tabular_standards"]["interpretation"],
                },
                {
                    "dimension": "Equidad",
                    "estado": "OK"
                    if np.isnan(fairness_max_gap) or fairness_max_gap <= 0.10
                    else "Seguimiento",
                    "implicacion": fairness_audit["interpretation"],
                },
                {
                    "dimension": "Representatividad",
                    "estado": "OK" if sampling_alerts.empty else "Seguimiento",
                    "implicacion": governance_audit["sampling_audit"]["interpretation"],
                },
                {
                    "dimension": "Deep Learning Governance",
                    "estado": "OK" if dl_required_gaps == 0 else "Seguimiento",
                    "implicacion": deep_learning_governance["interpretation"],
                },
            ]
        )
        return {
            "execution_gate": execution_gate,
            "governance_audit": governance_audit,
            "missingness_audit": missingness_audit,
            "drift_report": drift_report,
            "health_report": health_report,
            "rfe_report": rfe_report,
            "normalidad": normalidad,
            "vif_report": vif_report,
            "correlacion": correlacion,
            "simpson": simpson,
            "comparacion_segmentos": comparacion_segmentos,
            "ols_result": ols_result,
            "ols_figure": ols_figure,
            "multiverse_result": multiverse_result,
            "parsimonia": benchmark_result["parsimonia"],
            "metodologia_result": metodologia_result,
            "temporal_validation": temporal_validation,
            "uncertainty": uncertainty,
            "fairness_audit": fairness_audit,
            "deep_learning_governance": deep_learning_governance,
            "calibration_result": calibration_result,
            "calibration_figure": calibration_figure,
            "risk_flags": risk_flags,
            "summary": {
                "brier_score": float(calibration_metrics["brier_score"]),
                "top_vif": top_vif,
                "top_condition_index": top_condition_index,
                "critical_belsley_count": critical_belsley_count,
                "coefficient_stability_alert_level": "alert" if top_vif >= 5 or top_condition_index >= 30 or critical_belsley_count > 0 else "ok",
                "simpson_detected": bool(simpson["paradox_detected"]),
                "drift_columns": int(drift_summary["columnas_con_drift_severo"]),
                "drift_alert_level": "alert" if int(drift_summary["columnas_con_drift_severo"]) > 0 else "ok",
                "tabular_alerts": int(tabular_summary.sum()),
                "sampling_alerts": int(len(sampling_alerts)),
                "pipeline_health_profile": health_config.profile_name,
                "pipeline_health_decision": str(health_summary["decision_operativa"]),
                "modeling_enabled": modeling_enabled,
                "selected_feature_count": int(len(selected_features)),
                "temporal_cv_roc_auc": float(temporal_summary.get("mean_roc_auc", np.nan)),
                "temporal_cv_std": float(temporal_summary.get("std_roc_auc", np.nan)),
                "bootstrap_interval_width_mean": float(uncertainty_summary.get("mean_interval_width", np.nan)),
                "fairness_max_gap": fairness_max_gap,
                "calibration_method": str(calibration_metrics.get("calibration_method", "not_run")),
                "normalidad_consenso": str(normalidad_fila["consenso"]),
                "anderson_alert": bool(anderson_fila["rechaza_normalidad"]),
            },
        }

    def run_shap_transparency(self, bi_result: dict[str, Any]) -> dict[str, Any]:
        """Extrae la capa SHAP del framework comun y la resume para gobierno del score."""

        shap_explainability = bi_result["modelado"]["explainability"]
        if shap_explainability["method"] != "shap" or shap_explainability["shap_payload"] is None:
            return {
                "available": False,
                "summary": pd.DataFrame(),
                "consistency": pd.DataFrame(),
                "dependence_audit": pd.DataFrame(),
                "local_feature_table": pd.DataFrame(),
                "summary_figure": None,
                "dependence_figure": None,
                "message": "SHAP no estuvo disponible en la corrida actual.",
            }

        shap_summary = shap_explainability["summary"].copy().rename(
            columns={"score": "mean_abs_shap", "dispersion": "std_abs_shap"}
        )
        shap_consistency = shap_explainability["consistency_report"].copy()
        shap_dependence_audit = shap_explainability["dependence_audit"].copy()
        shap_payload = shap_explainability["shap_payload"]
        shap_pipeline = bi_result["modelado"]["model"]["pipeline"]
        shap_sample = shap_payload["sample"].copy()
        shap_sample_probability = shap_pipeline.predict_proba(shap_sample)[:, -1]
        local_position = int(np.argmax(shap_sample_probability))
        local_cliente_id = (
            shap_sample.iloc[local_position]["cliente_id"]
            if "cliente_id" in shap_sample.columns
            else shap_sample.index[local_position]
        )
        local_feature_table = pd.DataFrame(
            {
                "feature": shap_payload["transformed_sample"].columns,
                "shap_value": shap_payload["values"][local_position],
                "abs_shap_value": np.abs(shap_payload["values"][local_position]),
                "valor_transformado": shap_payload["transformed_sample"].iloc[local_position].to_numpy(),
            }
        ).sort_values("abs_shap_value", ascending=False)
        local_feature_table["direccion_score"] = np.where(
            local_feature_table["shap_value"] >= 0,
            "Empuja al alza el score de abandono",
            "Reduce el score de abandono",
        )
        consistency_row = shap_consistency.iloc[0]
        return {
            "available": True,
            "summary": shap_summary,
            "consistency": shap_consistency,
            "dependence_audit": shap_dependence_audit,
            "local_feature_table": local_feature_table,
            "summary_figure": shap_explainability["figures"].get("summary_plot"),
            "dependence_figure": shap_explainability["figures"].get("dependence_plot"),
            "summary_metrics": {
                "local_cliente_id": int(local_cliente_id) if pd.notna(local_cliente_id) else str(local_cliente_id),
                "local_probability": float(shap_sample_probability[local_position]),
                "consistency_error_mean": float(consistency_row["error_abs_medio"]),
                "consistency_error_max": float(consistency_row["error_abs_max"]),
            },
        }

    def run_clv_activation(self, df_bank: pd.DataFrame, bi_result: dict[str, Any]) -> dict[str, Any]:
        """Convierte el score en umbral economico, scorecard y segmentacion CRM."""

        contract = self.get_case_contract()["contract"]
        clv_result = estimate_bank_probabilistic_clv(
            df_bank,
            horizons_months=self.config.case.clv_horizons,
            verbose=False,
        )
        df_enriched = clv_result["data"].copy()
        value_model_features = [
            column for column in contract["value_model_features"] if column in df_enriched.columns
        ]
        value_benchmark = benchmark_bank_value_models(
            df_enriched,
            target_column="clv_valor_observado_12m",
            feature_columns=value_model_features,
            algorithms=("random_forest", "neural_network"),
            catastrophic_error_weight=self.config.case.financial_error_asymmetry,
            test_size=self.config.case.test_size,
            random_state=self.config.seed,
            verbose=False,
        )
        df_enriched = value_benchmark["scored_data"].copy()
        df_enriched["valor_ml_champion_12m"] = df_enriched["prediccion_valor_champion"].round(2)
        df_enriched["gap_relativo_valor_ml_vs_clv"] = np.where(
            np.abs(df_enriched["valor_ml_champion_12m"]) > 1.0,
            (
                (df_enriched["clv_probabilistico_12m"] - df_enriched["valor_ml_champion_12m"])
                / np.abs(df_enriched["valor_ml_champion_12m"])
            ).round(4),
            np.nan,
        )
        threshold_result = evaluate_retention_thresholds(
            model_result=bi_result["modelado"]["model"],
            reference_df=df_enriched,
            value_column="clv_probabilistico_12m",
            value_is_period_total=True,
            contact_cost=self.config.case.contact_cost,
            retention_success_rate=self.config.case.retention_success_rate,
            max_contact_share=self.config.case.max_contact_share,
            verbose=False,
        )
        scorecard_result = build_bank_retention_scorecard(
            model_result=bi_result["modelado"]["model"],
            reference_df=df_enriched,
            threshold_result=threshold_result,
            value_column="clv_probabilistico_12m",
            contact_cost=self.config.case.contact_cost,
            retention_success_rate=self.config.case.retention_success_rate,
            verbose=False,
        )
        dashboard_retencion = build_bank_retention_dashboard(
            scorecard_result=scorecard_result,
            threshold_result=threshold_result,
            verbose=False,
        )
        consensus_gap = build_consensus_gap_report(
            scorecard_result["scorecard"],
            group_column="macro_segmento" if "macro_segmento" in scorecard_result["scorecard"].columns else "segmento",
            score_column="probabilidad_abandono",
            value_column="valor_esperado_contacto",
            verbose=False,
        )
        scorecard = consensus_gap["scorecard"].copy()
        scorecard_result["scorecard"] = scorecard
        if "shortlist" in scorecard_result and not scorecard_result["shortlist"].empty and "cliente_id" in scorecard_result["shortlist"].columns and "cliente_id" in scorecard.columns:
            gap_columns = ["cliente_id", "brecha_score_vs_consenso", "brecha_valor_vs_consenso"]
            scorecard_result["shortlist"] = scorecard_result["shortlist"].merge(
                scorecard[gap_columns],
                on="cliente_id",
                how="left",
            )
        bandit_policy = build_bandit_policy(
            scorecard,
            action_column="next_best_offer",
            reward_column="valor_esperado_contacto",
            epsilon=self.config.case.bandit.epsilon,
            verbose=False,
        )
        segment_options, _, _ = evaluar_kmeans_opciones(
            scorecard,
            columnas=contract["segment_features"],
            ks=range(2, 6),
            random_state=self.config.seed,
        )
        segment_best_k = int(
            segment_options.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"]
        )
        segment_result = ejecutar_segmentacion_kmeans(
            scorecard,
            columnas=contract["segment_features"],
            n_clusters=segment_best_k,
            random_state=self.config.seed,
            cluster_col="segmento_kmeans",
        )
        scorecard_segmentado = segment_result["data"].copy()
        perfil_segmentos = perfilar_segmentos(
            scorecard_segmentado,
            cluster_col="segmento_kmeans",
            aggregations={
                "cliente_id": "count",
                "probabilidad_abandono": "mean",
                "probabilidad_activo_clv": "mean",
                "valor_esperado_contacto": "mean",
                "clv_probabilistico_6m": "mean",
                "prioridad_integrada": "mean",
            },
        )
        ranking_valor = perfil_segmentos["clv_probabilistico_6m"].rank(method="dense", ascending=False)
        ranking_riesgo = perfil_segmentos["probabilidad_abandono"].rank(method="dense", ascending=False)
        segment_labels: dict[Any, str] = {}
        for idx, row in perfil_segmentos.iterrows():
            cluster_id = row["segmento_kmeans"]
            if ranking_valor.loc[idx] == 1 and ranking_riesgo.loc[idx] <= 2:
                segment_labels[cluster_id] = "Riesgo critico de alto CLV"
            elif ranking_riesgo.loc[idx] == 1 or ranking_valor.loc[idx] <= 2:
                segment_labels[cluster_id] = "Riesgo alto monetizable"
            else:
                segment_labels[cluster_id] = "Monitoreo preventivo de valor medio"
        perfil_segmentos["segmento_nombre"] = perfil_segmentos["segmento_kmeans"].map(segment_labels)
        scorecard_segmentado["segmento_nombre"] = scorecard_segmentado["segmento_kmeans"].map(segment_labels)
        resumen_segmentos = (
            scorecard_segmentado.groupby("segmento_nombre")
            .agg(
                clientes=("cliente_id", "size"),
                score_medio=("probabilidad_abandono", "mean"),
                probabilidad_activo_media=("probabilidad_activo_clv", "mean"),
                clv_6m_medio=("clv_probabilistico_6m", "mean"),
                valor_esperado_medio=("valor_esperado_contacto", "mean"),
            )
            .round(3)
            .reset_index()
        )
        playbook_segmentos = (
            scorecard_segmentado.groupby("segmento_nombre")
            .agg(
                clientes=("cliente_id", "size"),
                pct_priorizados=("abandono_predicho", "mean"),
                score_medio=("probabilidad_abandono", "mean"),
                clv_6m_medio=("clv_probabilistico_6m", "mean"),
                valor_esperado_medio=("valor_esperado_contacto", "mean"),
                oferta_dominante=(
                    "next_best_offer",
                    lambda values: values.mode().iloc[0] if not values.mode().empty else "Sin oferta dominante",
                ),
            )
            .round(3)
            .reset_index()
        )
        if not bandit_policy["summary"].empty and "recommended_arm" in bandit_policy["summary"].columns:
            playbook_segmentos["bandit_arm_recomendado"] = str(bandit_policy["summary"].iloc[0]["recommended_arm"])
        fig_segmentacion, _ = plot_dashboard_segmentacion_nba(
            df_segmentado=scorecard_segmentado,
            perfil_segmentos=perfil_segmentos,
            segment_col="segmento_kmeans",
            score_col="probabilidad_abandono",
            score_label="Probabilidad media de abandono",
            action_col="next_best_offer",
            value_col="valor_esperado_contacto",
            label_col="segmento_nombre",
            title="Dashboard ejecutivo de segmentacion de retencion con CLV probabilistico",
        )
        recommended_threshold = threshold_result["recommended_threshold"].iloc[0]
        best_silhouette = float(
            segment_options.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["silhouette"]
        )
        return {
            "data": df_enriched,
            "clv_result": clv_result,
            "value_benchmark": value_benchmark,
            "threshold_result": threshold_result,
            "scorecard_result": scorecard_result,
            "dashboard_retencion": dashboard_retencion,
            "consensus_gap": consensus_gap,
            "bandit_policy": bandit_policy,
            "segment_options": segment_options,
            "segment_result": segment_result,
            "scorecard_segmentado": scorecard_segmentado,
            "perfil_segmentos": perfil_segmentos,
            "resumen_segmentos": resumen_segmentos,
            "playbook_segmentos": playbook_segmentos,
            "segment_dashboard_figure": fig_segmentacion,
            "summary": {
                "clv_engine": str(clv_result["model_summary"].iloc[0]["motor"]),
                "champion_valor": str(value_benchmark["summary"].iloc[0]["modelo"]),
                "threshold": float(recommended_threshold["umbral"]),
                "roi_estimado": float(recommended_threshold["roi_estimado"]),
                "valor_esperado_neto": float(recommended_threshold["valor_esperado_neto"]),
                "clientes_priorizados": int((scorecard_segmentado["abandono_predicho"] == 1).sum()),
                "segment_best_k": segment_best_k,
                "consensus_gap_mean": float(consensus_gap["summary"]["brecha_valor_media"].mean()),
                "bandit_recommended_arm": str(bandit_policy["summary"].iloc[0]["recommended_arm"]) if not bandit_policy["summary"].empty else "not_available",
                "best_silhouette": best_silhouette,
            },
        }

    def build_execution_summary(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        """Sintetiza las metricas clave para CLI, JSON y tracking."""

        benchmark = artifacts["benchmark"]
        bi_layer = artifacts["bi_layer"]
        methodology = artifacts["methodology"]
        activation = artifacts["activation"]
        methodology_summary = methodology.get("summary", {})
        summary = {
            "champion_model": benchmark["model_name"],
            "benchmark_primary_metric": float(
                benchmark["benchmark_df"].iloc[0]["roc_auc"]
                if "roc_auc" in benchmark["benchmark_df"].columns
                else benchmark["benchmark_df"].iloc[0]["rmse"]
            ),
            "benchmark_secondary_metric": float(
                benchmark["benchmark_df"].iloc[0]["f1"]
                if "f1" in benchmark["benchmark_df"].columns
                else benchmark["benchmark_df"].iloc[0]["mae"]
            ),
            "bi_roc_auc": float(bi_layer["metrics"].get("roc_auc", np.nan)),
            "bi_log_loss": float(bi_layer["metrics"].get("log_loss", np.nan)),
            "brier_score": float(methodology_summary.get("brier_score", np.nan)),
            "drift_columns": int(methodology_summary.get("drift_columns", 0)),
            "drift_alert_level": str(methodology_summary.get("drift_alert_level", "unknown")),
            "pipeline_health_profile": str(methodology_summary.get("pipeline_health_profile", "unknown")),
            "pipeline_health_decision": str(methodology_summary.get("pipeline_health_decision", "unknown")),
            "selected_feature_count": int(methodology_summary.get("selected_feature_count", len(benchmark.get("selected_features", [])))),
            "temporal_cv_roc_auc": float(methodology_summary.get("temporal_cv_roc_auc", np.nan)),
            "fairness_max_gap": float(methodology_summary.get("fairness_max_gap", np.nan)),
            "bootstrap_interval_width_mean": float(methodology_summary.get("bootstrap_interval_width_mean", np.nan)),
            "coefficient_stability_alert_level": str(methodology_summary.get("coefficient_stability_alert_level", "unknown")),
            "threshold": float(activation["summary"]["threshold"]),
            "roi_estimado": float(activation["summary"]["roi_estimado"]),
            "valor_esperado_neto": float(activation["summary"]["valor_esperado_neto"]),
            "clientes_priorizados": int(activation["summary"]["clientes_priorizados"]),
            "segment_best_k": int(activation["summary"]["segment_best_k"]),
            "best_silhouette": float(activation["summary"]["best_silhouette"]),
            "bandit_recommended_arm": str(activation["summary"].get("bandit_recommended_arm", "not_available")),
        }
        summary["lectura_ejecutiva"] = (
            "El caso queda industrializado en una topologia modular con benchmark dinamico por negocio, parsimonia, "
            "validacion temporal purgada, fairness audit, SHAP y un semaforo ejecutivo que ya incorpora drift PSI y estabilidad de coeficientes."
        )
        return summary