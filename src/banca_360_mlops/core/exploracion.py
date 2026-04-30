"""Funciones para exploracion y perfilado rapido del dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .limpieza import detectar_outliers_iqr, reporte_calidad_datos, resumir_nulos


def resumen_numerico(df: pd.DataFrame) -> pd.DataFrame:
	"""Resume las columnas numericas con estadistica descriptiva ampliada.

	Ademas del resumen clasico de `describe`, incorpora IQR, asimetria y curtosis.
	Esto te permite detectar dispersion elevada, distribuciones sesgadas o colas
	pronunciadas antes de decidir transformaciones, winsorizacion o escalado.
	"""
	numericas = df.select_dtypes(include=np.number)
	if numericas.empty:
		return pd.DataFrame()

	resumen = numericas.describe().T
	resumen["iqr"] = resumen["75%"] - resumen["25%"]
	resumen["asimetria"] = numericas.skew(numeric_only=True)
	resumen["curtosis"] = numericas.kurt(numeric_only=True)
	return resumen.round(3).sort_values("std", ascending=False)


def resumen_categorico(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
	"""Resume variables categoricas mostrando cardinalidad y categorias dominantes.

	Es util para localizar columnas con pocas categorias, etiquetas inconsistentes o
	valores dominantes que puedan sesgar un modelo o una lectura superficial del dataset.
	El parametro `top_n` controla cuantas categorias frecuentes quieres listar.
	"""
	categoricas = df.select_dtypes(include=["object", "category", "bool"])
	if categoricas.empty:
		return pd.DataFrame()

	filas = []
	for columna in categoricas.columns:
		conteos = categoricas[columna].fillna("<NA>").value_counts(dropna=False)
		filas.append(
			{
				"columna": columna,
				"valores_unicos": categoricas[columna].nunique(dropna=True),
				"valores_principales": ", ".join(
					f"{indice}: {conteo}"
					for indice, conteo in conteos.head(top_n).items()
				),
			}
		)
	return pd.DataFrame(filas).sort_values("valores_unicos")


def reporte_eda_rapido(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
	"""Agrupa en un solo diccionario los reportes base de una EDA inicial.

	La salida esta pensada para acelerar el flujo de trabajo: puedes inspeccionar de
	forma ordenada calidad, nulos, variables numericas, variables categoricas y outliers
	sin tener que lanzar cinco funciones por separado cada vez que empiezas un estudio.
	"""
	return {
		"calidad": reporte_calidad_datos(df),
		"nulos": resumir_nulos(df),
		"numerico": resumen_numerico(df),
		"categorico": resumen_categorico(df),
		"outliers": detectar_outliers_iqr(df),
	}


def checklist_analitico(tipo_trabajo: str = "eda") -> list[str]:
	"""Devuelve una lista de control segun el tipo de trabajo analitico.

	Sirve como guia operativa cuando quieres que tus notebooks tengan un orden estable.
	Puedes usarlo al principio del cuaderno para recordar que pasos no deberias saltarte,
	tanto en limpieza como en exploracion, modelado o preparacion de dashboards.
	"""
	checklists = {
		"eda": [
			"Revisar dimensiones, tipos de datos y variables clave.",
			"Medir nulos, duplicados, cardinalidad y posibles incoherencias.",
			"Estudiar distribuciones, outliers y relaciones entre variables.",
			"Documentar hallazgos, supuestos y decisiones de limpieza.",
		],
		"limpieza": [
			"Estandarizar nombres de columnas y formatos de fecha.",
			"Tratar nulos con una regla explicita por variable.",
			"Detectar duplicados, valores imposibles y categorias inconsistentes.",
			"Guardar una version limpia y reproducible del dataset.",
		],
		"modelado": [
			"Definir variable objetivo y criterio de evaluacion.",
			"Separar train, validation y test evitando fuga de datos.",
			"Construir una linea base antes de modelos complejos.",
			"Comparar metricas, errores y variables influyentes.",
		],
		"dashboard": [
			"Definir audiencia, preguntas de negocio y KPIs.",
			"Preparar tablas limpias y metricas consistentes.",
			"Elegir visualizaciones segun comparacion, tendencia o composicion.",
			"Validar legibilidad, filtros y narrativa final.",
		],
	}
	return checklists.get(tipo_trabajo.lower(), checklists["eda"])
