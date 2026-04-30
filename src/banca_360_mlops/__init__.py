"""Proyecto MLOps modular para el caso Banca 360."""

import matplotlib

# Fuerza un backend no interactivo para ejecuciones por CLI y tracking headless.
matplotlib.use("Agg")

from .config import ProjectConfig, load_project_config

__all__ = ["ProjectConfig", "load_project_config"]