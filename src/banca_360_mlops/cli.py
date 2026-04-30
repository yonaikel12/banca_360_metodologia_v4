"""CLI unica para ejecutar el pipeline completo desde consola."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
import traceback
from datetime import datetime

from .pipeline.orchestrator import run_pipeline


class _TeeStream:
    """Replica stdout/stderr en consola y archivo sin depender del shell."""

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _resolve_project_root(project_root: str | None) -> Path:
    return Path(project_root).resolve() if project_root else Path.cwd().resolve()


def _resolve_log_path(project_root: Path, log_path: str | None) -> Path:
    if log_path:
        candidate = Path(log_path)
        return candidate if candidate.is_absolute() else (project_root / candidate).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (project_root / "data" / "processed" / f"pipeline_run_{timestamp}_v4.log").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline MLOps del caso Banca 360 V4.")
    parser.add_argument("command", choices=["run"], help="Comando a ejecutar sobre el proyecto.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Ruta del proyecto si la ejecucion no ocurre desde la raiz del repositorio.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Ruta opcional del log. Si no se informa, la CLI genera un archivo timestamped en data/processed/.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        project_root = _resolve_project_root(args.project_root)
        log_path = _resolve_log_path(project_root, args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            stdout_tee = _TeeStream(sys.stdout, log_file)
            stderr_tee = _TeeStream(sys.stderr, log_file)
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                print(f"timestamp={datetime.now().isoformat()}")
                print(f"project_root={project_root}")
                print(f"log_path={log_path}")
                try:
                    artifacts = run_pipeline(project_root)
                    summary = artifacts["summary"]
                    print("Pipeline Banca 360 V4 ejecutado correctamente.")
                    print(f"- Modelo champion: {summary['champion_model']}")
                    print(f"- Variables operativas: {summary['selected_feature_count']}")
                    print(f"- Metrica primaria benchmark: {summary['benchmark_primary_metric']:.4f}")
                    print(f"- ROC AUC temporal purgado: {summary['temporal_cv_roc_auc']:.4f}")
                    print(f"- Umbral recomendado: {summary['threshold']:.2f}")
                    print(f"- ROI estimado: {summary['roi_estimado']:.4f}")
                    print(f"- Clientes priorizados: {summary['clientes_priorizados']}")
                    print(f"- Log persistido: {log_path}")
                except Exception:
                    print("Pipeline Banca 360 V4 finalizo con error.")
                    traceback.print_exc()
                    raise


if __name__ == "__main__":
    main()