"""Funciones de visualizacion reutilizables para notebooks y reportes."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .configuracion import aplicar_tema_profesional
from .limpieza import resumir_nulos


def grafico_nulos(df: pd.DataFrame, limite: int = 20) -> tuple[plt.Figure, plt.Axes]:
	"""Dibuja las columnas con mayor porcentaje de nulos.

	Es una visual rapida para priorizar la limpieza. Si no hay nulos, devuelve un lienzo
	vacio con un mensaje claro para no romper el flujo del notebook ni obligarte a tratar
	casos especiales manualmente.
	"""
	aplicar_tema_profesional()
	resumen = resumir_nulos(df)
	resumen = resumen[resumen["nulos"] > 0].head(limite).sort_values("pct_nulos")

	fig, ax = plt.subplots()
	if resumen.empty:
		ax.text(0.5, 0.5, "No se detectaron valores nulos", ha="center", va="center", fontsize=13)
		ax.set_axis_off()
		return fig, ax

	ax.barh(resumen.index, resumen["pct_nulos"], color="#EA580C")
	ax.set_title("Porcentaje de nulos por columna")
	ax.set_xlabel("Nulos (%)")
	ax.set_ylabel("")
	return fig, ax


def grafico_distribuciones_numericas(
	df: pd.DataFrame,
	columnas: Iterable[str] | None = None,
	max_columnas: int = 6,
) -> tuple[plt.Figure, np.ndarray]:
	"""Muestra histogramas con curva KDE para varias variables numericas.

	La funcion selecciona automaticamente columnas numericas si no indicas una lista.
	Esto acelera una primera revision de forma, dispersion y asimetria, y te ayuda a ver
	si conviene transformar variables o revisar outliers antes de seguir avanzando.
	"""
	aplicar_tema_profesional()
	seleccionadas = list(columnas) if columnas is not None else list(df.select_dtypes(include=np.number).columns[:max_columnas])

	if not seleccionadas:
		fig, ax = plt.subplots()
		ax.text(0.5, 0.5, "No hay columnas numericas disponibles", ha="center", va="center", fontsize=13)
		ax.set_axis_off()
		return fig, np.array([ax])

	n_cols = min(2, len(seleccionadas))
	n_rows = int(np.ceil(len(seleccionadas) / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
	axes = np.array(axes).reshape(-1)

	for ax, columna in zip(axes, seleccionadas):
		sns.histplot(df[columna].dropna(), kde=True, ax=ax, color="#2563EB")
		ax.set_title(f"Distribucion: {columna}")
		ax.set_xlabel(columna)

	for ax in axes[len(seleccionadas):]:
		ax.set_axis_off()

	fig.tight_layout()
	return fig, axes


def grafico_conteos_categoricos(
	df: pd.DataFrame,
	columnas: Iterable[str] | None = None,
	top_n: int = 10,
) -> tuple[plt.Figure, np.ndarray]:
	"""Muestra las categorias mas frecuentes de varias variables categoricas.

	Sirve para revisar desequilibrios, codificaciones sospechosas o categorias dominantes.
	La salida deja cada variable en un panel separado para facilitar comparaciones rapidas
	durante la exploracion o al preparar una narrativa para presentacion.
	"""
	aplicar_tema_profesional()
	seleccionadas = list(columnas) if columnas is not None else list(
		df.select_dtypes(include=["object", "category", "bool"]).columns[:4]
	)

	if not seleccionadas:
		fig, ax = plt.subplots()
		ax.text(0.5, 0.5, "No hay columnas categoricas disponibles", ha="center", va="center", fontsize=13)
		ax.set_axis_off()
		return fig, np.array([ax])

	n_cols = min(2, len(seleccionadas))
	n_rows = int(np.ceil(len(seleccionadas) / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
	axes = np.array(axes).reshape(-1)

	for ax, columna in zip(axes, seleccionadas):
		conteos = df[columna].fillna("<NA>").value_counts().head(top_n).sort_values()
		ax.barh(conteos.index.astype(str), conteos.values, color="#0F766E")
		ax.set_title(f"Categorias principales: {columna}")
		ax.set_xlabel("Conteo")

	for ax in axes[len(seleccionadas):]:
		ax.set_axis_off()

	fig.tight_layout()
	return fig, axes


def grafico_mapa_correlacion(
	df: pd.DataFrame,
	metodo: str = "pearson",
) -> tuple[plt.Figure, plt.Axes]:
	"""Representa una matriz de correlacion para variables numericas.

	Es util para detectar relaciones lineales fuertes, redundancias entre variables y
	posibles problemas de multicolinealidad. Si no hay suficientes columnas numericas,
	devuelve una figura vacia informativa en lugar de lanzar un error.
	"""
	aplicar_tema_profesional()
	correlacion = df.select_dtypes(include=np.number).corr(method=metodo)

	fig, ax = plt.subplots(figsize=(10, 8))
	if correlacion.empty:
		ax.text(0.5, 0.5, "No hay suficientes columnas numericas para correlacion", ha="center", va="center", fontsize=13)
		ax.set_axis_off()
		return fig, ax

	sns.heatmap(correlacion, cmap="crest", annot=False, square=True, ax=ax)
	ax.set_title("Mapa de correlacion")
	return fig, ax
