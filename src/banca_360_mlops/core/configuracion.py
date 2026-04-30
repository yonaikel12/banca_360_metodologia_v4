"""Configuracion visual compartida para notebooks y analisis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


TEMA_PROFESIONAL = {
	"figure.figsize": (10, 6),
	"axes.facecolor": "#F8FAFC",
	"figure.facecolor": "white",
	"axes.edgecolor": "#CBD5E1",
	"axes.labelcolor": "#0F172A",
	"axes.titleweight": "bold",
	"axes.titlecolor": "#0F172A",
	"xtick.color": "#334155",
	"ytick.color": "#334155",
	"grid.color": "#E2E8F0",
	"grid.linestyle": "--",
	"grid.alpha": 0.6,
	"font.size": 11,
}

PALETA_PROFESIONAL = [
	"#0F766E",
	"#2563EB",
	"#EA580C",
	"#B45309",
	"#7C3AED",
	"#BE123C",
	"#1D4ED8",
]


def aplicar_tema_profesional() -> None:
	"""Aplica un estilo visual consistente y mas cuidado que el predeterminado.

	Usa esta funcion al inicio de un notebook para que todos los graficos compartan
	la misma paleta, fondos claros, rejillas suaves y tipografia mas legible.
	Resulta util cuando quieres pasar de una exploracion tecnica a una entrega que
	pueda mostrarse en clase, en un informe o en una presentacion sin retocar cada figura.
	"""
	sns.set_theme(style="whitegrid", palette=PALETA_PROFESIONAL)
	plt.rcParams.update(TEMA_PROFESIONAL)
