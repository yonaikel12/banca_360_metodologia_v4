"""Persistencia ligera para datasets, tablas, figuras y resumenes del pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectConfig


def ensure_runtime_layout(config: ProjectConfig) -> None:
    """Garantiza que las carpetas operativas existan antes de persistir salidas."""

    for directory in (
        config.raw_data_dir,
        config.interim_data_dir,
        config.processed_data_dir,
        config.figures_dir,
        config.tracking.tracking_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_dataframe(df: pd.DataFrame, destination: Path) -> None:
    """Persiste una tabla en CSV para facilitar auditoria fuera de notebook."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, encoding="utf-8", na_rep="NA")


def save_json(payload: dict[str, Any], destination: Path) -> None:
    """Persiste un diccionario en JSON usando conversiones seguras."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_figure(figure: Any, destination: Path) -> None:
    """Guarda una figura matplotlib si la salida esta disponible."""

    if figure is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)