"""Orquestador principal del pipeline Banca 360.

La orquestacion ejecuta el caso de extremo a extremo, persiste artefactos en la
estructura cookiecutter y registra metricas en MLflow cuando esta habilitado.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import ProjectConfig, load_project_config
from ..io import ensure_runtime_layout, save_dataframe, save_figure, save_json
from ..services.bank360_case import Bank360CaseService
from ..tracking import ExperimentTracker
from ..utils.reproducibility import set_global_seed
from .nodes import (
    activation_node,
    benchmark_node,
    bi_layer_node,
    build_context_node,
    build_dataset_node,
    methodology_node,
    shap_node,
)


class Bank360PipelineOrchestrator:
    """Ejecuta el pipeline narrativo y productivo con topologia modular."""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or load_project_config()
        self.service = Bank360CaseService(self.config)
        self.tracker = ExperimentTracker(self.config)

    async def run(self) -> dict[str, Any]:
        """Ejecuta todas las fases del caso y persiste sus artefactos principales."""

        ensure_runtime_layout(self.config)
        set_global_seed(self.config.seed)

        with self.tracker.active_run():
            context, dataset = await asyncio.gather(
                build_context_node(self.service),
                build_dataset_node(self.service),
            )
            benchmark = await benchmark_node(self.service, dataset["data"])
            bi_layer, methodology = await asyncio.gather(
                bi_layer_node(self.service, dataset["data"], benchmark),
                methodology_node(self.service, dataset["data"], benchmark),
            )
            shap_result = await shap_node(self.service, bi_layer["bi_result"])
            activation = await activation_node(self.service, dataset["data"], bi_layer["bi_result"])

            artifacts = {
                "context": context,
                "dataset": dataset,
                "benchmark": benchmark,
                "bi_layer": bi_layer,
                "methodology": methodology,
                "shap": shap_result,
                "activation": activation,
            }
            summary = self.service.build_execution_summary(artifacts)
            artifacts["summary"] = summary

            self._persist_artifacts(artifacts)
            self._track_run(summary)
            return artifacts

    def _track_run(self, summary: dict[str, Any]) -> None:
        primary_metric = float(summary.get("benchmark_primary_metric", summary.get("benchmark_roc_auc", float("nan"))))
        secondary_metric = float(summary.get("benchmark_secondary_metric", summary.get("benchmark_f1", float("nan"))))
        self.tracker.log_params(
            {
                "seed": self.config.seed,
                "dataset_rows": self.config.case.dataset_rows,
                "target": self.config.case.target,
                "test_size": self.config.case.test_size,
                "selected_feature_count": summary["selected_feature_count"],
            }
        )
        self.tracker.log_metrics(
            {
                "benchmark_primary_metric": primary_metric,
                "benchmark_secondary_metric": secondary_metric,
                "bi_roc_auc": summary["bi_roc_auc"],
                "bi_log_loss": summary["bi_log_loss"],
                "brier_score": summary["brier_score"],
                "temporal_cv_roc_auc": summary["temporal_cv_roc_auc"],
                "fairness_max_gap": summary["fairness_max_gap"],
                "bootstrap_interval_width_mean": summary["bootstrap_interval_width_mean"],
                "threshold": summary["threshold"],
                "roi_estimado": summary["roi_estimado"],
                "valor_esperado_neto": summary["valor_esperado_neto"],
                "best_silhouette": summary["best_silhouette"],
            }
        )
        self.tracker.log_artifacts(self.config.processed_data_dir)

    def _persist_artifacts(self, artifacts: dict[str, Any]) -> None:
        save_dataframe(artifacts["context"]["manual_resumen"], self.config.raw_data_dir / "manual_bi_resumen.csv")
        save_dataframe(artifacts["context"]["guia_metricas"], self.config.raw_data_dir / "guia_metricas.csv")
        save_dataframe(
            artifacts["context"]["metodologia_v3_resumen"],
            self.config.raw_data_dir / "metodologia_v3_resumen.csv",
        )
        save_dataframe(
            artifacts["context"]["alineacion_metodologica_pdf"],
            self.config.raw_data_dir / "alineacion_metodologica_pdf.csv",
        )
        save_dataframe(artifacts["dataset"]["data"], self.config.raw_data_dir / "bank360_dataset.csv")
        save_dataframe(
            artifacts["dataset"]["data_dictionary"]["dictionary"],
            self.config.raw_data_dir / "bank360_data_dictionary.csv",
        )
        save_dataframe(
            artifacts["dataset"]["data_dictionary"]["metadata_table"],
            self.config.raw_data_dir / "bank360_data_dictionary_metadata.csv",
        )
        save_json(
            artifacts["dataset"]["data_dictionary"]["schema"],
            self.config.raw_data_dir / "bank360_data_dictionary_schema.json",
        )

        save_dataframe(artifacts["benchmark"]["benchmark_df"], self.config.interim_data_dir / "benchmark_modelos.csv")
        save_dataframe(
            artifacts["benchmark"]["parsimonia"]["summary"],
            self.config.interim_data_dir / "benchmark_parsimonia.csv",
        )
        save_dataframe(
            artifacts["bi_layer"]["bi_result"]["conclusiones"]["summary"],
            self.config.interim_data_dir / "bi_conclusiones.csv",
        )
        save_dataframe(
            artifacts["dataset"]["tabular_standards"]["summary"],
            self.config.interim_data_dir / "tabular_contract_summary.csv",
        )
        save_dataframe(
            artifacts["dataset"]["tabular_standards"]["column_name_audit"],
            self.config.interim_data_dir / "column_name_audit.csv",
        )
        save_dataframe(
            artifacts["dataset"]["sampling_audit"]["sampling_plan"],
            self.config.interim_data_dir / "sampling_plan.csv",
        )
        if not artifacts["dataset"]["sampling_audit"]["alerts"].empty:
            save_dataframe(
                artifacts["dataset"]["sampling_audit"]["alerts"],
                self.config.interim_data_dir / "sampling_alerts.csv",
            )
        save_dataframe(artifacts["methodology"]["risk_flags"], self.config.interim_data_dir / "riesgos_metodologicos.csv")
        save_dataframe(
            artifacts["methodology"]["governance_audit"]["tabular_standards"]["summary"],
            self.config.interim_data_dir / "methodology_tabular_summary.csv",
        )
        if not artifacts["methodology"]["governance_audit"]["sampling_audit"]["segment_distribution"].empty:
            save_dataframe(
                artifacts["methodology"]["governance_audit"]["sampling_audit"]["segment_distribution"],
                self.config.interim_data_dir / "segment_distribution_audit.csv",
            )
        save_dataframe(
            artifacts["methodology"]["temporal_validation"]["summary"],
            self.config.interim_data_dir / "temporal_validation_summary.csv",
        )
        save_dataframe(
            artifacts["methodology"]["temporal_validation"]["fold_report"],
            self.config.interim_data_dir / "temporal_validation_folds.csv",
        )
        save_dataframe(
            artifacts["methodology"]["uncertainty"]["portfolio_summary"],
            self.config.interim_data_dir / "uncertainty_portfolio_summary.csv",
        )
        save_dataframe(
            artifacts["methodology"]["fairness_audit"]["summary"],
            self.config.interim_data_dir / "fairness_summary.csv",
        )
        if not artifacts["methodology"]["fairness_audit"]["group_metrics"].empty:
            save_dataframe(
                artifacts["methodology"]["fairness_audit"]["group_metrics"],
                self.config.interim_data_dir / "fairness_group_metrics.csv",
            )
        save_dataframe(
            artifacts["methodology"]["deep_learning_governance"]["checklist"],
            self.config.interim_data_dir / "deep_learning_governance.csv",
        )

        save_dataframe(
            artifacts["activation"]["scorecard_result"]["scorecard"],
            self.config.processed_data_dir / "scorecard_retencion.csv",
        )
        save_dataframe(
            artifacts["activation"]["scorecard_result"]["shortlist"],
            self.config.processed_data_dir / "shortlist_retencion.csv",
        )
        save_dataframe(
            artifacts["activation"]["perfil_segmentos"],
            self.config.processed_data_dir / "perfil_segmentos.csv",
        )
        save_dataframe(
            artifacts["activation"]["resumen_segmentos"],
            self.config.processed_data_dir / "resumen_segmentos.csv",
        )
        save_dataframe(
            artifacts["activation"]["playbook_segmentos"],
            self.config.processed_data_dir / "playbook_segmentos.csv",
        )
        save_dataframe(
            artifacts["activation"]["consensus_gap"]["summary"],
            self.config.processed_data_dir / "brecha_valor_consenso.csv",
        )
        save_dataframe(
            artifacts["activation"]["bandit_policy"]["policy_table"],
            self.config.processed_data_dir / "bandit_policy.csv",
        )
        if artifacts["shap"]["available"]:
            save_dataframe(artifacts["shap"]["summary"], self.config.processed_data_dir / "shap_summary.csv")
            save_dataframe(
                artifacts["shap"]["local_feature_table"],
                self.config.processed_data_dir / "shap_lectura_local.csv",
            )

        save_figure(
            artifacts["activation"]["dashboard_retencion"]["figure"],
            self.config.figures_dir / "dashboard_retencion.png",
        )
        save_figure(
            artifacts["activation"]["segment_dashboard_figure"],
            self.config.figures_dir / "dashboard_segmentacion.png",
        )
        save_figure(
            artifacts["activation"]["threshold_result"]["figure"],
            self.config.figures_dir / "tradeoff_umbral.png",
        )
        save_figure(
            artifacts["methodology"]["calibration_figure"],
            self.config.figures_dir / "calibracion.png",
        )
        save_figure(
            artifacts["methodology"]["ols_figure"],
            self.config.figures_dir / "diagnostico_ols.png",
        )
        if artifacts["shap"]["available"]:
            save_figure(artifacts["shap"]["summary_figure"], self.config.figures_dir / "shap_summary.png")
            save_figure(
                artifacts["shap"]["dependence_figure"],
                self.config.figures_dir / "shap_dependence.png",
            )

        save_json(artifacts["summary"], self.config.processed_data_dir / "execution_summary.json")


def run_pipeline(project_root: Path | None = None) -> dict[str, Any]:
    """Punto de entrada sincronico para CLI y notebook."""

    config = load_project_config(project_root=project_root)
    orchestrator = Bank360PipelineOrchestrator(config)
    return asyncio.run(orchestrator.run())