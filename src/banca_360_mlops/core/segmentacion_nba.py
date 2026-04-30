"""Utilidades reutilizables para segmentacion de clientes y next-best-action.

Este modulo separa la logica reusable de clustering, perfilado de segmentos y
asignacion de acciones recomendadas para que pueda reutilizarse desde notebooks,
scripts o dashboards sin duplicar codigo de negocio.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _ensure_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esperaba un pandas.DataFrame como entrada.")
    if df.empty:
        raise ValueError("El dataframe esta vacio. No hay informacion para segmentar.")


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"No se encontraron las columnas requeridas: {missing}")


def preparar_matriz_segmentacion(
    df: pd.DataFrame,
    columnas: Sequence[str],
    imputer_strategy: Literal["mean", "median"] = "median",
) -> dict[str, Any]:
    """Imputa y escala una matriz numerica lista para clustering."""
    _ensure_dataframe(df)
    _ensure_columns(df, columnas)

    trabajo = df[list(columnas)].copy()
    trabajo = pd.DataFrame(
        SimpleImputer(strategy=imputer_strategy).fit_transform(trabajo),
        columns=list(columnas),
        index=df.index,
    )
    matriz = StandardScaler().fit_transform(trabajo)
    return {
        "data": trabajo,
        "matrix": matriz,
        "columns": list(columnas),
        "imputer_strategy": imputer_strategy,
    }


def evaluar_kmeans_opciones(
    df: pd.DataFrame,
    columnas: Sequence[str],
    ks: range = range(3, 7),
    random_state: int = 42,
    imputer_strategy: Literal["mean", "median"] = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compara varias opciones de KMeans con silhouette e inercia."""
    preparado = preparar_matriz_segmentacion(
        df,
        columnas=columnas,
        imputer_strategy=imputer_strategy,
    )
    filas = []
    for k in ks:
        modelo = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        etiquetas = modelo.fit_predict(preparado["matrix"])
        filas.append(
            {
                "k": k,
                "silhouette": round(float(silhouette_score(preparado["matrix"], etiquetas)), 4),
                "inertia": round(float(modelo.inertia_), 2),
            }
        )
    return pd.DataFrame(filas), preparado["data"], preparado["matrix"]


def ejecutar_segmentacion_kmeans(
    df: pd.DataFrame,
    columnas: Sequence[str],
    n_clusters: int,
    random_state: int = 42,
    pca_components: int = 2,
    cluster_col: str = "cluster_id",
    imputer_strategy: Literal["mean", "median"] = "median",
) -> dict[str, Any]:
    """Ejecuta KMeans y proyecta los segmentos a un espacio PCA interpretable."""
    preparado = preparar_matriz_segmentacion(
        df,
        columnas=columnas,
        imputer_strategy=imputer_strategy,
    )
    modelo = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    etiquetas = modelo.fit_predict(preparado["matrix"])

    df_segmentado = df.copy()
    df_segmentado[cluster_col] = etiquetas

    resultado = {
        "data": df_segmentado,
        "matrix": preparado["matrix"],
        "prepared_data": preparado["data"],
        "model": modelo,
        "cluster_col": cluster_col,
    }

    if pca_components >= 2:
        pca = PCA(n_components=pca_components, random_state=random_state)
        coords = pca.fit_transform(preparado["matrix"])
        for index in range(min(pca_components, coords.shape[1])):
            df_segmentado[f"pc{index + 1}"] = coords[:, index]
        resultado["pca"] = pca
        resultado["coords"] = coords

    return resultado


def perfilar_segmentos(
    df: pd.DataFrame,
    cluster_col: str,
    aggregations: dict[str, str | list[str] | Any],
    round_digits: int = 3,
) -> pd.DataFrame:
    """Agrega metricas por segmento para construir perfiles de negocio."""
    _ensure_dataframe(df)
    _ensure_columns(df, [cluster_col, *aggregations.keys()])
    return df.groupby(cluster_col).agg(aggregations).round(round_digits).reset_index()


def asignar_etiquetas_segmento(
    perfil_segmentos: pd.DataFrame,
    rules: Sequence[dict[str, str]],
    cluster_col: str = "cluster_id",
    label_col: str = "segmento_negocio",
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Asigna etiquetas de negocio a clusters usando reglas de prioridad por metrica.

    Cada regla debe incluir:
    - label: nombre de negocio del segmento.
    - metric: columna del perfil para ordenar clusters.
    - direction: 'max' o 'min'.
    """
    _ensure_dataframe(perfil_segmentos)
    _ensure_columns(perfil_segmentos, [cluster_col])

    disponibles = set(perfil_segmentos[cluster_col].tolist())
    mapping: dict[int, str] = {}

    for rule in rules:
        metric = rule["metric"]
        direction = rule.get("direction", "max")
        label = rule["label"]
        if not disponibles or metric not in perfil_segmentos.columns:
            continue
        subset = perfil_segmentos[perfil_segmentos[cluster_col].isin(disponibles)]
        cluster = subset[metric].idxmax() if direction == "max" else subset[metric].idxmin()
        cluster_id = int(perfil_segmentos.loc[cluster, cluster_col])
        mapping[cluster_id] = label
        disponibles.discard(cluster_id)

    remaining_labels = [rule["label"] for rule in rules if rule["label"] not in set(mapping.values())]
    for cluster_id, label in zip(sorted(disponibles), remaining_labels, strict=False):
        mapping[int(cluster_id)] = label
    for cluster_id in sorted(disponibles):
        mapping.setdefault(int(cluster_id), f"Segmento {cluster_id}")

    perfil = perfil_segmentos.copy()
    perfil[label_col] = perfil[cluster_col].map(mapping)
    return perfil, mapping


def etiquetar_segmentos_negocio(
    df: pd.DataFrame,
    cluster_col: str,
    rules: Sequence[dict[str, str]],
    aggregations: dict[str, str | list[str] | Any],
    label_col: str = "segmento_negocio",
    round_digits: int = 3,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Perfila segmentos y les asigna nombres de negocio en una sola llamada."""
    perfil = perfilar_segmentos(
        df,
        cluster_col=cluster_col,
        aggregations=aggregations,
        round_digits=round_digits,
    )
    return asignar_etiquetas_segmento(
        perfil,
        rules=rules,
        cluster_col=cluster_col,
        label_col=label_col,
    )


def asignar_next_best_action(
    df_scoring: pd.DataFrame,
    modelos: dict[str, dict[str, Any]],
    features: Sequence[str],
    beneficios: dict[str, float | pd.Series],
    costos: dict[str, float],
    nombres_accion: dict[str, str],
    action_col: str = "next_best_action",
    value_col: str = "valor_esperado_nba",
    fallback_action: str = "Monitorear",
    min_value_threshold: float = 0.0,
) -> pd.DataFrame:
    """Asigna la mejor accion a cada fila segun probabilidad y valor esperado."""
    _ensure_dataframe(df_scoring)
    _ensure_columns(df_scoring, features)

    trabajo = df_scoring.copy()
    probability_cols: list[str] = []
    value_cols: list[str] = []

    for accion, modelo in modelos.items():
        if accion not in beneficios or accion not in costos or accion not in nombres_accion:
            raise ValueError(f"La accion '{accion}' debe existir en beneficios, costos y nombres_accion.")
        if "pipeline" not in modelo:
            raise ValueError(f"El modelo para '{accion}' no contiene una clave 'pipeline'.")
        pipeline = modelo["pipeline"]
        if not hasattr(pipeline, "predict_proba"):
            raise ValueError(f"El modelo para '{accion}' no soporta predict_proba.")

        probability_col = f"proba_{accion}"
        value_action_col = f"valor_esperado_{accion}"
        probability = pipeline.predict_proba(trabajo[list(features)])[:, 1]
        benefit = beneficios[accion]
        trabajo[probability_col] = probability
        trabajo[value_action_col] = probability * benefit - costos[accion]
        probability_cols.append(probability_col)
        value_cols.append(value_action_col)

    trabajo["mejor_columna_accion"] = trabajo[value_cols].idxmax(axis=1)
    trabajo[action_col] = trabajo["mejor_columna_accion"].map(
        {f"valor_esperado_{accion}": nombre for accion, nombre in nombres_accion.items()}
    )
    trabajo[value_col] = trabajo[value_cols].max(axis=1)
    trabajo.loc[trabajo[value_col] <= min_value_threshold, action_col] = fallback_action
    return trabajo


def resumir_next_best_action(
    df: pd.DataFrame,
    segment_col: str,
    action_col: str = "next_best_action",
    value_col: str = "valor_esperado_nba",
    probability_cols: Sequence[str] | None = None,
    round_digits: int = 3,
) -> dict[str, pd.DataFrame]:
    """Construye tablas ejecutivas de mix de acciones y valor esperado por segmento."""
    _ensure_dataframe(df)
    required = [segment_col, action_col, value_col]
    if probability_cols is not None:
        required.extend(probability_cols)
    _ensure_columns(df, required)

    aggregations: dict[str, str] = {value_col: "mean"}
    if probability_cols is not None:
        aggregations.update({column: "mean" for column in probability_cols})

    resumen_segmento = df.groupby(segment_col).agg(aggregations).round(round_digits)
    mix_accion = pd.crosstab(df[segment_col], df[action_col], normalize="index").round(round_digits)
    distribucion_accion = (
        df[action_col].value_counts(normalize=True).mul(100).round(2).rename("pct_clientes").to_frame()
    )

    return {
        "summary": resumen_segmento,
        "action_mix": mix_accion,
        "action_distribution": distribucion_accion,
    }


def plot_dashboard_segmentacion_nba(
    df_segmentado: pd.DataFrame,
    perfil_segmentos: pd.DataFrame,
    segment_col: str,
    score_col: str,
    score_label: str,
    action_col: str = "next_best_action",
    value_col: str = "valor_esperado_nba",
    label_col: str = "segmento_negocio",
    title: str = "Dashboard ejecutivo de segmentacion y next-best-action",
) -> tuple[plt.Figure, np.ndarray]:
    """Dibuja un dashboard compacto para segmentacion y decisioning."""
    _ensure_dataframe(df_segmentado)
    _ensure_dataframe(perfil_segmentos)
    _ensure_columns(df_segmentado, [segment_col, score_col, action_col, value_col])
    _ensure_columns(perfil_segmentos, [label_col, score_col])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.ravel()

    kpis_texto = (
        f"Filas analizadas: {len(df_segmentado):,}\n"
        f"Segmentos descubiertos: {df_segmentado[segment_col].nunique()}\n"
        f"{score_label} medio: {df_segmentado[score_col].mean():,.0f}\n"
        f"Accion dominante: {df_segmentado[action_col].mode().iat[0]}"
    )
    axes[0].axis("off")
    axes[0].text(0.02, 0.95, "KPIs del Caso", fontsize=15, fontweight="bold", va="top")
    axes[0].text(0.02, 0.78, kpis_texto, fontsize=12, va="top")

    perfil_plot = perfil_segmentos.sort_values(score_col)
    axes[1].barh(perfil_plot[label_col], perfil_plot[score_col], color="#2563EB")
    axes[1].set_title(f"{score_label} medio por segmento")
    axes[1].set_xlabel(score_label)

    acciones = df_segmentado[action_col].value_counts().sort_values()
    axes[2].barh(acciones.index, acciones.values, color="#0F766E")
    axes[2].set_title("Distribucion de next-best-action")
    axes[2].set_xlabel("Clientes")

    tabla_valor = df_segmentado.groupby(segment_col)[value_col].mean().sort_values()
    axes[3].barh(tabla_valor.index, tabla_valor.values, color="#EA580C")
    axes[3].set_title("Valor esperado medio por segmento")
    axes[3].set_xlabel("Valor esperado")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig, axes
