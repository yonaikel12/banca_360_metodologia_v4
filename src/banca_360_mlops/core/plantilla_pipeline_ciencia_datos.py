"""Plantilla base para ejecutar un pipeline metodologico de ciencia de datos.

La idea de este archivo es ofrecer un punto de partida listo para copiar y adaptar.
Cada bloque llama a las funciones del modulo `metodologia` y devuelve un diccionario
con resultados tabulares, interpretaciones y objetos graficos.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .datasets_sinteticos import generar_dataset_clientes_sintetico
from .metodologia import (
    analyze_correlation,
    audit_dataset,
    audit_missingness_mechanism,
    build_preprocessing_pipeline,
    calculate_vif,
    check_normality,
    compare_groups,
    compare_power_transformations,
    detect_simpsons_paradox,
    evaluate_dataset_drift,
    evaluate_probability_calibration,
    evaluate_competitive_event_predictions,
    fit_ols_inference,
    handle_outliers,
    impute_missing_values,
    get_universal_methodology_reference,
    normalize_competitive_event_probabilities,
    plot_competitive_event_diagnostics,
    plot_feature_importance,
    plot_group_distributions,
    plot_missingness_heatmap,
    plot_model_diagnostics,
    plot_ols_influence_diagnostics,
    plot_probability_calibration,
    plot_structural_dispersion_diagnostics,
    plot_power_transformations,
    plot_qq_diagnostic,
    report_pipeline_health,
    run_multiverse_analysis,
    run_rfe_feature_selection,
    train_competitive_event_model,
    train_supervised_model,
)


def ejecutar_pipeline_base(
    df: pd.DataFrame,
    objetivo: str,
    features_modelo: Sequence[str],
    columna_normalidad: str,
    columna_grupo: str,
    columna_valor_grupo: str,
    columnas_outliers: Sequence[str] | None = None,
    columnas_vif: Sequence[str] | None = None,
    problema_modelado: str = "auto",
    algoritmo_modelado: str = "auto",
) -> dict[str, Any]:
    """Ejecuta una plantilla metodologica de extremo a extremo sobre un DataFrame.

    Entradas:
        df: Dataset estructurado ya cargado en memoria.
        objetivo: Nombre de la variable objetivo para modelado.
        features_modelo: Predictores a usar en el modelo supervisado.
        columna_normalidad: Variable numerica sobre la que validar normalidad y transformaciones.
        columna_grupo: Variable categorica para comparacion de grupos.
        columna_valor_grupo: Variable numerica a comparar entre grupos.
        columnas_outliers: Columnas numericas que recibiran tratamiento de outliers.
        columnas_vif: Columnas numericas a usar para VIF.
        problema_modelado: 'classification', 'regression' o 'auto'.
        algoritmo_modelado: Algoritmo del modelo supervisado.

    Salidas:
        Diccionario con resultados por fase y figuras listas para mostrar o guardar.

    Pruebas ejecutadas:
        Auditoria de calidad, stack de normalidad estructural, transformaciones de potencia, VIF,
        comparacion de grupos, dispersion estructural del OLS y entrenamiento supervisado con permutation importance.
    """
    # Fase 1: audita sesgos obvios, faltantes y posibles riesgos de leakage.
    auditoria = audit_dataset(df, target=objetivo, verbose=True)
    figura_faltantes, _ = plot_missingness_heatmap(df)

    # Fase 2: controla extremos y deja montado el pipeline reproducible de preprocesado.
    etl = handle_outliers(df, columns=columnas_outliers, method="clip_iqr", verbose=True)
    preprocesado = build_preprocessing_pipeline(
        etl["data"][list(features_modelo)],
        apply_power_transform=True,
        power_method="yeo-johnson",
        verbose=True,
    )

    # Fase 3: revisa forma, colas y transformaciones antes de defender supuestos parametricos.
    normalidad = check_normality(etl["data"][columna_normalidad], verbose=True)
    figura_qq, _ = plot_qq_diagnostic(etl["data"][columna_normalidad])
    transformaciones = compare_power_transformations(etl["data"][columna_normalidad], verbose=True)
    figura_transformaciones, _ = plot_power_transformations(etl["data"][columna_normalidad])
    vif = calculate_vif(etl["data"], columns=columnas_vif, verbose=True)
    correlacion = analyze_correlation(
        etl["data"],
        x_column=features_modelo[0],
        y_column=columna_normalidad,
        verbose=True,
    )

    # Fase 4: entrena un baseline reproducible y extrae diagnosticos visuales e importancia.
    modelo = train_supervised_model(
        etl["data"],
        target=objetivo,
        problem_type=problema_modelado,
        algorithm=algoritmo_modelado,
        features=features_modelo,
        apply_power_transform=True,
        power_method="yeo-johnson",
        verbose=True,
    )
    figura_importancia, _ = plot_feature_importance(modelo["feature_importance"])
    figura_modelo, _ = plot_model_diagnostics(modelo)

    # Fase 5: combina contraste entre grupos e inferencia OLS para cerrar la lectura estadistica.
    inferencia_grupos = compare_groups(
        etl["data"],
        value_column=columna_valor_grupo,
        group_column=columna_grupo,
        verbose=True,
    )
    figura_grupos, _ = plot_group_distributions(
        etl["data"],
        value_column=columna_valor_grupo,
        group_column=columna_grupo,
    )

    inferencia_ols = fit_ols_inference(
        etl["data"],
        target=columna_valor_grupo,
        features=[column for column in features_modelo if column != columna_valor_grupo],
        group_column=columna_grupo,
        verbose=True,
    )
    figura_dispersion_ols, _ = plot_structural_dispersion_diagnostics(inferencia_ols)

    return {
        "auditoria": auditoria,
        "etl": etl,
        "eda": {
            "normalidad": normalidad,
            "normalidad_resumen": normalidad.get("shape_summary"),
            "normalidad_detalle": normalidad.get("tests_table"),
            "transformaciones": transformaciones,
            "vif": vif,
            "correlacion": correlacion,
        },
        "modelado": modelo,
        "inferencia": {
            "comparacion_grupos": inferencia_grupos,
            "ols": inferencia_ols,
            "dispersion_estructural": inferencia_ols.get("dispersion_audit"),
        },
        "figuras": {
            "faltantes": figura_faltantes,
            "qq_plot": figura_qq,
            "transformaciones": figura_transformaciones,
            "importancia_variables": figura_importancia,
            "diagnostico_modelo": figura_modelo,
            "dispersion_estructural": figura_dispersion_ols,
            "grupos": figura_grupos,
        },
        "preprocesado": preprocesado,
    }


def ejemplo_con_dataset_sintetico() -> dict[str, Any]:
    """Ejecuta la plantilla completa sobre un dataset sintetico de clientes."""
    # Este ejemplo deja un caso reproducible para probar el boilerplate sin depender de datos externos.
    df = generar_dataset_clientes_sintetico(n_registros=800, semilla=42)
    return ejecutar_pipeline_base(
        df=df,
        objetivo="Abandono",
        features_modelo=[
            "Edad",
            "Ingreso Mensual",
            "Gasto Mensual",
            "Visitas Web 30D",
            "Compras 12M",
            "Satisfaccion",
            "Segmento",
            "Usa App",
        ],
        columna_normalidad="Ingreso Mensual",
        columna_grupo="Segmento",
        columna_valor_grupo="Gasto Mensual",
        columnas_outliers=["Ingreso Mensual", "Gasto Mensual"],
        columnas_vif=["Edad", "Ingreso Mensual", "Gasto Mensual", "Visitas Web 30D", "Compras 12M"],
        problema_modelado="classification",
        algoritmo_modelado="random_forest",
    )


def ejecutar_pipeline_evento_competitivo(
    df: pd.DataFrame,
    objetivo: str,
    columna_evento: str,
    features_modelo: Sequence[str],
    columna_normalidad: str,
    columna_valor_competidor: str,
    columnas_outliers: Sequence[str] | None = None,
    columnas_vif: Sequence[str] | None = None,
    algoritmo_modelado: str = "logistic",
    top_k: int = 3,
) -> dict[str, Any]:
    """Ejecuta un pipeline competitivo por evento para problemas de ganador por carrera.

    Entradas:
        df: Dataset con una fila por competidor.
        objetivo: Variable binaria que marca el ganador del evento.
        columna_evento: Identificador de carrera o evento competitivo.
        features_modelo: Predictores previos al evento.
        columna_normalidad: Variable numerica para revisar forma y transformaciones.
        columna_valor_competidor: Variable numerica de interes para comparar ganadores vs no ganadores.
        columnas_outliers: Columnas numericas donde controlar extremos.
        columnas_vif: Columnas numericas para multicolinealidad.
        algoritmo_modelado: Algoritmo competitivo base.
        top_k: Profundidad usada en ranking y diagnosticos.

    Salidas:
        Diccionario con auditoria, ETL, EDA, modelado competitivo y figuras.

    Pruebas ejecutadas:
        Auditoria, normalidad estructural, transformaciones, VIF, comparacion ganadores vs no ganadores,
        entrenamiento competitivo por evento y ranking/calida de probabilidades por carrera.
    """
    # Adapta la plantilla general a escenarios donde los competidores se comparan dentro de un mismo evento.
    auditoria = audit_dataset(df, target=objetivo, verbose=True)
    figura_faltantes, _ = plot_missingness_heatmap(df)

    # El preprocesado se ajusta una vez antes del baseline competitivo para preservar comparabilidad entre participantes.
    etl = handle_outliers(df, columns=columnas_outliers, method="clip_iqr", verbose=True)
    preprocesado = build_preprocessing_pipeline(
        etl["data"][list(features_modelo)],
        apply_power_transform=True,
        power_method="yeo-johnson",
        verbose=True,
    )

    normalidad = check_normality(etl["data"][columna_normalidad], verbose=True)
    figura_qq, _ = plot_qq_diagnostic(etl["data"][columna_normalidad])
    transformaciones = compare_power_transformations(etl["data"][columna_normalidad], verbose=True)
    figura_transformaciones, _ = plot_power_transformations(etl["data"][columna_normalidad])
    vif = calculate_vif(etl["data"], columns=columnas_vif, verbose=True)
    correlacion = analyze_correlation(
        etl["data"],
        x_column=features_modelo[0],
        y_column=columna_normalidad,
        verbose=True,
    )

    # El modelo competitivo aprende probabilidades por competidor y luego las normaliza dentro de cada evento.
    modelo = train_competitive_event_model(
        etl["data"],
        target=objetivo,
        group_column=columna_evento,
        features=features_modelo,
        algorithm=algoritmo_modelado,
        apply_power_transform=True,
        power_method="yeo-johnson",
        top_k=top_k,
        verbose=True,
    )
    figura_importancia, _ = plot_feature_importance(modelo["feature_importance"])
    figura_modelo, _ = plot_competitive_event_diagnostics(modelo)

    inferencia_ganador = compare_groups(
        etl["data"],
        value_column=columna_valor_competidor,
        group_column=objetivo,
        verbose=True,
    )
    figura_grupos, _ = plot_group_distributions(
        etl["data"],
        value_column=columna_valor_competidor,
        group_column=objetivo,
    )

    return {
        "auditoria": auditoria,
        "etl": etl,
        "eda": {
            "normalidad": normalidad,
            "normalidad_resumen": normalidad.get("shape_summary"),
            "normalidad_detalle": normalidad.get("tests_table"),
            "transformaciones": transformaciones,
            "vif": vif,
            "correlacion": correlacion,
        },
        "modelado": modelo,
        "inferencia": {
            "ganador_vs_no_ganador": inferencia_ganador,
        },
        "figuras": {
            "faltantes": figura_faltantes,
            "qq_plot": figura_qq,
            "transformaciones": figura_transformaciones,
            "importancia_variables": figura_importancia,
            "diagnostico_competitivo": figura_modelo,
            "grupos": figura_grupos,
        },
        "preprocesado": preprocesado,
    }


def ejecutar_pipeline_metodologico_universal(
    df: pd.DataFrame,
    objetivo: str,
    features_modelo: Sequence[str],
    columna_fecha: str,
    columna_normalidad: str,
    columna_grupo: str,
    columna_valor_grupo: str,
    columnas_outliers: Sequence[str] | None = None,
    columnas_vif: Sequence[str] | None = None,
    id_columns: Sequence[str] | None = None,
    problema_modelado: str = "auto",
    algoritmo_modelado: str = "random_forest",
) -> dict[str, Any]:
    """Ejecuta la metodologia universal ampliada sobre un caso de negocio tabular.

    Entradas:
        df: Dataset estructurado cargado en memoria.
        objetivo: Variable objetivo del caso.
        features_modelo: Predictores para el modelo principal.
        columna_fecha: Variable temporal para construir snapshot de drift y salud operativa.
        columna_normalidad: Variable numerica a revisar por forma y transformacion.
        columna_grupo: Variable categorica para inferencia por segmentos.
        columna_valor_grupo: Variable numerica para contraste entre grupos.
        columnas_outliers: Columnas numericas donde controlar extremos.
        columnas_vif: Columnas numericas para VIF.
        id_columns: Identificadores a auditar por unicidad.
        problema_modelado: Tipo de problema o auto.
        algoritmo_modelado: Algoritmo principal del caso.

    Salidas:
        Diccionario con fases metodologicas, tablas maestras y figuras listas para uso en notebook.

    Pruebas ejecutadas:
        Auditoria, faltantes, drift, ETL, normalidad estructural, Simpson, VIF, RFE, modelado,
        calibracion, dispersion estructural del OLS, diagnostico OLS y multiverse analysis.
    """
    # Orquesta la metodologia universal sobre un caso tabular sin dejar pasos criticos fuera del flujo.
    working = df.copy()
    working[columna_fecha] = pd.to_datetime(working[columna_fecha], errors="coerce")
    if working[columna_fecha].isna().all():
        raise ValueError("columna_fecha no contiene fechas validas para construir snapshots de drift.")

    # El corte temporal separa una referencia historica de un tramo mas reciente para auditar drift y salud operativa.
    reference_cut = working[columna_fecha].quantile(0.65)
    reference_df = working[working[columna_fecha] <= reference_cut].copy()
    current_df = working[working[columna_fecha] > reference_cut].copy()
    if reference_df.empty or current_df.empty:
        midpoint = len(working) // 2
        reference_df = working.iloc[:midpoint].copy()
        current_df = working.iloc[midpoint:].copy()

    framework_reference = get_universal_methodology_reference(verbose=True)
    auditoria = audit_dataset(working, target=objetivo, id_columns=id_columns, verbose=True)
    figura_faltantes, _ = plot_missingness_heatmap(working)
    auditoria_faltantes = audit_missingness_mechanism(working, columns=[*features_modelo, objetivo], verbose=True)
    drift = evaluate_dataset_drift(reference_df, current_df, columns=[*features_modelo, objetivo, columna_grupo], verbose=True)
    # Los NaN iniciales de fracdiff son warm-up esperado y no deben contarse como fallo estructural de calidad.
    columnas_validacion = [column for column in features_modelo if not str(column).endswith("_fracdiff_0_4")]
    filas_con_error = int(working[columnas_validacion + [objetivo]].isna().any(axis=1).sum())
    salud = report_pipeline_health(
        dataset_name="caso_negocio_tabular",
        updated_at=working[columna_fecha].max(),
        expected_rows=len(reference_df),
        observed_rows=len(current_df),
        validation_failed_rows=filas_con_error,
        total_validated_rows=len(working),
        verbose=True,
    )

    etl = handle_outliers(working, columns=columnas_outliers, method="clip_iqr", verbose=True)
    imputacion = impute_missing_values(
        etl["data"],
        strategy="mice",
        columns=list(dict.fromkeys([*features_modelo, objetivo, columna_normalidad, columna_valor_grupo, columna_grupo])),
        verbose=True,
    )
    prepared_data = imputacion["data"]
    preprocesado = build_preprocessing_pipeline(
        prepared_data[list(features_modelo)],
        numeric_imputer="iterative",
        apply_power_transform=True,
        power_method="yeo-johnson",
        verbose=True,
    )

    normalidad = check_normality(prepared_data[columna_normalidad], verbose=True)
    figura_qq, _ = plot_qq_diagnostic(prepared_data[columna_normalidad])
    transformaciones = compare_power_transformations(prepared_data[columna_normalidad], verbose=True)
    figura_transformaciones, _ = plot_power_transformations(prepared_data[columna_normalidad])
    vif = calculate_vif(prepared_data, columns=columnas_vif, verbose=True)
    correlacion = analyze_correlation(prepared_data, x_column=features_modelo[0], y_column=columna_normalidad, verbose=True)
    simpson = detect_simpsons_paradox(
        prepared_data,
        x_column="Ingreso Mensual" if "Ingreso Mensual" in prepared_data.columns else features_modelo[0],
        y_column=columna_valor_grupo,
        group_column=columna_grupo,
        verbose=True,
    )
    rfe = run_rfe_feature_selection(
        prepared_data,
        target=objetivo,
        features=features_modelo,
        problem_type=problema_modelado,
        n_features_to_select=min(6, len(features_modelo)),
        verbose=True,
    )

    modelo = train_supervised_model(
        prepared_data,
        target=objetivo,
        problem_type=problema_modelado,
        algorithm=algoritmo_modelado,
        features=features_modelo,
        numeric_imputer="iterative",
        apply_power_transform=True,
        power_method="yeo-johnson",
        verbose=True,
    )
    figura_importancia, _ = plot_feature_importance(modelo["feature_importance"])
    figura_modelo, _ = plot_model_diagnostics(modelo)

    calibracion = None
    figura_calibracion = None
    # Solo se calibra cuando existe salida probabilistica binaria; en regresion o etiquetas duras no tiene sentido tecnico.
    if "predicted_probability" in modelo["predictions"].columns and prepared_data[objetivo].nunique(dropna=True) == 2:
        calibracion = evaluate_probability_calibration(modelo, verbose=True)
        figura_calibracion, _ = plot_probability_calibration(calibracion)

    inferencia_grupos = compare_groups(prepared_data, value_column=columna_valor_grupo, group_column=columna_grupo, verbose=True)
    figura_grupos, _ = plot_group_distributions(prepared_data, value_column=columna_valor_grupo, group_column=columna_grupo)
    inferencia_ols = fit_ols_inference(
        prepared_data,
        target=columna_valor_grupo,
        features=[column for column in features_modelo if column != columna_valor_grupo],
        group_column=columna_grupo,
        verbose=True,
    )
    dispersion_estructural = inferencia_ols.get("dispersion_audit")
    figura_dispersion_ols, _ = plot_structural_dispersion_diagnostics(inferencia_ols)
    figura_ols, _ = plot_ols_influence_diagnostics(inferencia_ols)

    multiverse = run_multiverse_analysis(
        prepared_data,
        target=objetivo,
        features=features_modelo,
        problem_type=problema_modelado,
        numeric_outlier_columns=columnas_outliers,
        verbose=True,
    )

    return {
        "framework_reference": framework_reference,
        "auditoria": auditoria,
        "auditoria_faltantes": auditoria_faltantes,
        "drift": drift,
        "salud_pipeline": salud,
        "etl": etl,
        "imputacion": imputacion,
        "preprocesado": preprocesado,
        "eda": {
            "normalidad": normalidad,
            "normalidad_resumen": normalidad.get("shape_summary"),
            "normalidad_detalle": normalidad.get("tests_table"),
            "transformaciones": transformaciones,
            "vif": vif,
            "correlacion": correlacion,
            "simpson": simpson,
            "rfe": rfe,
        },
        "modelado": modelo,
        "calibracion": calibracion,
        "inferencia": {
            "comparacion_grupos": inferencia_grupos,
            "dispersion_estructural": dispersion_estructural,
            "ols": inferencia_ols,
            "multiverse": multiverse,
        },
        "figuras": {
            "faltantes": figura_faltantes,
            "qq_plot": figura_qq,
            "transformaciones": figura_transformaciones,
            "importancia_variables": figura_importancia,
            "diagnostico_modelo": figura_modelo,
            "dispersion_estructural": figura_dispersion_ols,
            "diagnostico_ols": figura_ols,
            "grupos": figura_grupos,
            "calibracion": figura_calibracion,
        },
    }


def ejemplo_caso_negocio_metodologia_universal() -> dict[str, Any]:
    """Ejecuta la metodologia universal ampliada sobre un caso sintetico de abandono de clientes."""
    # Deja un caso de negocio reproducible para demostrar la metodologia completa sobre retencion de clientes.
    df = generar_dataset_clientes_sintetico(n_registros=1200, semilla=42)
    return ejecutar_pipeline_metodologico_universal(
        df=df,
        objetivo="Abandono",
        features_modelo=[
            "Edad",
            "Ingreso Mensual",
            "Gasto Mensual",
            "Visitas Web 30D",
            "Compras 12M",
            "Satisfaccion",
            "Reclamaciones",
            "Segmento",
            "Canal Captacion",
            "Usa App",
            "Producto Premium",
        ],
        columna_fecha="Fecha Registro",
        columna_normalidad="Ingreso Mensual",
        columna_grupo="Segmento",
        columna_valor_grupo="Gasto Mensual",
        columnas_outliers=["Ingreso Mensual", "Gasto Mensual", "Visitas Web 30D", "Compras 12M", "Reclamaciones"],
        columnas_vif=["Edad", "Ingreso Mensual", "Gasto Mensual", "Visitas Web 30D", "Compras 12M", "Reclamaciones"],
        id_columns=["Cliente ID"],
        problema_modelado="classification",
        algoritmo_modelado="random_forest",
    )
