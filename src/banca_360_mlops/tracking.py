"""Capa opcional de tracking con MLflow para ejecucion industrial del caso."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import ProjectConfig


class ExperimentTracker:
    """Envuelve MLflow con degradacion segura si la libreria no esta disponible."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._mlflow = None
        if config.tracking.enabled:
            try:
                import mlflow
            except ImportError:
                self._mlflow = None
            else:
                tracking_dir = config.tracking.tracking_dir
                tracking_dir.mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
                mlflow.set_experiment(config.tracking.experiment_name)
                self._mlflow = mlflow

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    @contextmanager
    def active_run(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        with self._mlflow.start_run(run_name=self.config.tracking.run_name):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for key, value in params.items():
            self._mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        sanitized = {key: float(value) for key, value in metrics.items() if value is not None}
        self._mlflow.log_metrics(sanitized)

    def log_artifacts(self, path: Path) -> None:
        if not self.enabled or not path.exists():
            return
        self._mlflow.log_artifacts(str(path))