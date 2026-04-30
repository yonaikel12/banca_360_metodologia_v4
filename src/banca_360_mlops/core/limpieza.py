"""Funciones de limpieza y control de calidad de datos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import unicodedata

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResumenOutlier:
	"""Representa el resumen de outliers de una columna numerica."""

	columna: str
	limite_inferior: float
	limite_superior: float
	total_outliers: int
	pct_outliers: float


def _quitar_acentos(texto: str) -> str:
	texto_normalizado = unicodedata.normalize("NFKD", texto)
	return "".join(caracter for caracter in texto_normalizado if not unicodedata.combining(caracter))


def normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
	"""Devuelve una copia del dataframe con nombres de columnas listos para trabajar.

	La funcion elimina espacios sobrantes, convierte a minusculas, quita acentos y
	reemplaza simbolos conflictivos por guiones bajos. Es especialmente util cuando
	vas a encadenar transformaciones, escribir formulas o reutilizar el dataset en
	varios notebooks sin tener que recordar nombres con espacios o caracteres raros.
	"""
	columnas_limpias = (
		pd.Index(_quitar_acentos(str(columna)) for columna in df.columns)
		.str.strip()
		.str.lower()
		.str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
		.str.strip("_")
	)
	return df.rename(columns=dict(zip(df.columns, columnas_limpias)))


def resumir_nulos(df: pd.DataFrame) -> pd.DataFrame:
	"""Resume la cantidad y el porcentaje de valores nulos por columna.

	Sirve como primera parada en cualquier trabajo de limpieza porque te dice donde
	hay problemas de completitud, que tipo de dato tiene cada variable y en que
	columnas conviene decidir una estrategia de imputacion, eliminacion o revision.
	"""
	summary = pd.DataFrame(
		{
			"nulos": df.isna().sum(),
			"pct_nulos": df.isna().mean().mul(100).round(2),
			"tipo_dato": df.dtypes.astype(str),
		}
	)
	return summary.sort_values(["nulos", "pct_nulos"], ascending=False)


def eliminar_duplicados(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
	"""Devuelve una copia sin filas duplicadas.

	La idea es centralizar una operacion que se repite muchisimo. Mantiene por defecto
	la primera aparicion de cada fila, aunque puedes cambiar la politica con el parametro
	`keep` si el caso requiere conservar la ultima o eliminar todas las duplicadas.
	"""
	return df.drop_duplicates(keep=keep).copy()


def imputar_nulos_basico(
	df: pd.DataFrame,
	reglas: dict[str, Any] | None = None,
	estrategia_numerica: str = "mediana",
	estrategia_categorica: str = "moda",
) -> pd.DataFrame:
	"""Imputa nulos con reglas explicitas o con una estrategia simple por tipo.

	Si pasas `reglas`, esas columnas usaran exactamente el valor indicado. Para el resto,
	las columnas numericas pueden rellenarse con mediana o media y las categoricas con la
	moda o una etiqueta fija. Es una solucion pragmatica para tener un baseline limpio
	y reproducible antes de plantear imputaciones mas sofisticadas.
	"""
	resultado = df.copy()
	reglas = reglas or {}

	for columna, valor in reglas.items():
		if columna in resultado.columns:
			resultado[columna] = resultado[columna].fillna(valor)

	for columna in resultado.columns:
		if columna in reglas:
			continue

		serie = resultado[columna]
		if pd.api.types.is_numeric_dtype(serie):
			if estrategia_numerica == "media":
				fill_value = serie.mean()
			else:
				fill_value = serie.median()
		else:
			if estrategia_categorica == "moda" and not serie.mode(dropna=True).empty:
				fill_value = serie.mode(dropna=True).iloc[0]
			else:
				fill_value = "desconocido"

		resultado[columna] = serie.fillna(fill_value)

	return resultado


def reporte_calidad_datos(df: pd.DataFrame) -> pd.DataFrame:
	"""Construye un informe compacto de calidad de datos por columna.

	Este reporte junta en una sola tabla la informacion que mas suele consultarse al
	inicio de un analisis: tipo de dato, volumen de nulos, cardinalidad, porcentaje de
	ceros en variables numericas, memoria aproximada y una pequena muestra de valores.
	Es muy util para detectar columnas problematicas y priorizar la limpieza.
	"""
	reporte = pd.DataFrame(
		{
			"tipo_dato": df.dtypes.astype(str),
			"no_nulos": df.notna().sum(),
			"nulos": df.isna().sum(),
			"pct_nulos": df.isna().mean().mul(100).round(2),
			"valores_unicos": df.nunique(dropna=True),
			"pct_duplicados_columna": df.apply(lambda serie: serie.duplicated().mean() * 100).round(2),
		}
	)

	reporte["muestra"] = [
		", ".join(map(str, df[columna].dropna().astype(str).head(3).tolist()))
		for columna in df.columns
	]
	reporte["pct_ceros"] = [
		round(pd.to_numeric(df[columna], errors="coerce").eq(0).mean() * 100, 2)
		if pd.api.types.is_numeric_dtype(df[columna])
		else np.nan
		for columna in df.columns
	]
	reporte["memoria_kb"] = (df.memory_usage(deep=True) / 1024).round(2).reindex(df.columns)
	reporte.attrs["filas"] = len(df)
	return reporte.sort_values(["pct_nulos", "valores_unicos"], ascending=[False, True])


def detectar_outliers_iqr(
	df: pd.DataFrame,
	columnas: Iterable[str] | None = None,
) -> pd.DataFrame:
	"""Detecta outliers numericos usando el criterio del rango intercuartil.

	Para cada columna seleccionada calcula Q1, Q3 e IQR, y marca como outlier cualquier
	valor por debajo de `Q1 - 1.5 * IQR` o por encima de `Q3 + 1.5 * IQR`. Esto ayuda a
	priorizar que variables necesitan revision antes de describir distribuciones o entrenar
	modelos sensibles a valores extremos.
	"""
	columnas_objetivo = list(columnas) if columnas is not None else list(df.select_dtypes(include=np.number).columns)
	resumenes: list[ResumenOutlier] = []

	for columna in columnas_objetivo:
		serie = pd.to_numeric(df[columna], errors="coerce").dropna()
		if serie.empty:
			continue

		q1 = serie.quantile(0.25)
		q3 = serie.quantile(0.75)
		iqr = q3 - q1
		limite_inferior = q1 - 1.5 * iqr
		limite_superior = q3 + 1.5 * iqr
		mask = (serie < limite_inferior) | (serie > limite_superior)

		resumenes.append(
			ResumenOutlier(
				columna=columna,
				limite_inferior=float(limite_inferior),
				limite_superior=float(limite_superior),
				total_outliers=int(mask.sum()),
				pct_outliers=round(mask.mean() * 100, 2),
			)
		)

	if not resumenes:
		return pd.DataFrame(
			columns=["columna", "limite_inferior", "limite_superior", "total_outliers", "pct_outliers"]
		)

	return pd.DataFrame(resumenes).sort_values("pct_outliers", ascending=False)


def recortar_outliers_iqr(
	df: pd.DataFrame,
	columnas: Iterable[str] | None = None,
) -> pd.DataFrame:
	"""Recorta valores extremos al rango permitido por IQR y devuelve una copia.

	No elimina filas: sustituye los extremos por el limite inferior o superior calculado.
	Es una opcion util cuando quieres mantener el tamano de la muestra pero reducir el
	impacto de valores anormalmente altos o bajos en graficos, escalados o modelos.
	"""
	resultado = df.copy()
	reporte = detectar_outliers_iqr(resultado, columnas=columnas)

	for fila in reporte.itertuples(index=False):
		resultado[fila.columna] = resultado[fila.columna].clip(fila.limite_inferior, fila.limite_superior)

	return resultado
