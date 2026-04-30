"""Generadores de datasets sinteticos para practicar y validar funciones."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generar_dataset_clientes_sintetico(n_registros: int = 1200, semilla: int = 42) -> pd.DataFrame:
	"""Crea un dataset sintetico de clientes con problemas realistas de calidad.

	El dataset incluye variables numericas y categoricas, una fecha de registro, una
	variable objetivo binaria para clasificacion y varias columnas utiles para dashboard.
	Tambien introduce algunos nulos y outliers para que puedas probar las funciones de
	limpieza, exploracion, visualizacion y modelado en un solo flujo.
	"""
	rng = np.random.default_rng(semilla)
	fechas = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, size=n_registros), unit="D")
	regiones = rng.choice(["Norte", "Sur", "Centro", "Este", "Oeste"], size=n_registros, p=[0.2, 0.2, 0.25, 0.15, 0.2])
	canales = rng.choice(["Web", "Tienda", "Afiliado", "Call Center"], size=n_registros, p=[0.42, 0.3, 0.18, 0.1])
	segmentos = rng.choice(["Basico", "Estandar", "Premium"], size=n_registros, p=[0.35, 0.45, 0.2])

	edad = np.clip(rng.normal(39, 11, size=n_registros).round(), 18, 78)
	ingreso = rng.lognormal(mean=7.7, sigma=0.35, size=n_registros)
	compras = rng.poisson(lam=5.5, size=n_registros)
	visitas = rng.poisson(lam=18, size=n_registros)
	satisfaccion = np.clip(rng.normal(7.1, 1.5, size=n_registros), 1, 10).round(1)
	reclamaciones = rng.poisson(lam=np.where(satisfaccion < 5, 1.6, 0.5), size=n_registros)
	usa_app = rng.choice(["Si", "No"], size=n_registros, p=[0.72, 0.28])
	premium = rng.choice(["Si", "No"], size=n_registros, p=[0.27, 0.73])
	gasto = ingreso * rng.uniform(0.12, 0.55, size=n_registros) + compras * rng.uniform(5, 25, size=n_registros)

	logit = (
		-1.6
		+ 0.55 * reclamaciones
		- 0.28 * satisfaccion
		- 0.10 * compras
		+ 0.00018 * np.maximum(2200 - ingreso, 0)
		+ 0.35 * (usa_app == "No")
	)
	prob_abandono = 1 / (1 + np.exp(-logit))
	abandono = rng.binomial(1, np.clip(prob_abandono, 0.03, 0.92))

	df = pd.DataFrame(
		{
			"Cliente ID": np.arange(1, n_registros + 1),
			"Fecha Registro": fechas,
			"Region": regiones,
			"Canal Captacion": canales,
			"Segmento": segmentos,
			"Edad": edad,
			"Ingreso Mensual": ingreso.round(2),
			"Gasto Mensual": gasto.round(2),
			"Visitas Web 30D": visitas,
			"Compras 12M": compras,
			"Satisfaccion": satisfaccion,
			"Reclamaciones": reclamaciones,
			"Usa App": usa_app,
			"Producto Premium": premium,
			"Abandono": abandono,
		}
	)

	indices_nulos_ingreso = rng.choice(df.index, size=max(10, n_registros // 25), replace=False)
	indices_nulos_satisfaccion = rng.choice(df.index, size=max(8, n_registros // 30), replace=False)
	indices_outliers = rng.choice(df.index, size=max(6, n_registros // 80), replace=False)

	df.loc[indices_nulos_ingreso, "Ingreso Mensual"] = np.nan
	df.loc[indices_nulos_satisfaccion, "Satisfaccion"] = np.nan
	df.loc[indices_outliers, "Gasto Mensual"] = df.loc[indices_outliers, "Gasto Mensual"] * 4.5

	return df
