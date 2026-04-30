"""Configuracion externa del proyecto y resolucion de rutas.

La configuracion centraliza semilla, parametros del caso y tracking para que la
ejecucion por notebook y por CLI use exactamente el mismo contrato operativo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    experiment_name: str
    run_name: str
    tracking_dir: Path


@dataclass(frozen=True)
class PipelineHealthConfig:
    profile_name: str
    freshness_threshold_hours: float
    count_tolerance_pct: float
    critical_freshness_multiplier: float
    critical_count_tolerance_pct: float
    validation_error_alert_pct: float
    validation_error_critical_pct: float


@dataclass(frozen=True)
class ParsimonyConfig:
    logistic_feature_steps: tuple[int, ...]
    roc_auc_tolerance: float


@dataclass(frozen=True)
class UncertaintyConfig:
    bootstrap_iterations: int
    prediction_interval_alpha: float


@dataclass(frozen=True)
class TemporalValidationConfig:
    n_splits: int
    purge_gap_days: int
    embargo_gap_days: int


@dataclass(frozen=True)
class FairnessConfig:
    sensitive_columns: tuple[str, ...]
    age_bins: tuple[int, ...]


@dataclass(frozen=True)
class BanditConfig:
    epsilon: float


@dataclass(frozen=True)
class DeepLearningConfig:
    enabled: bool
    require_dropout: bool
    require_batch_norm: bool
    require_early_stopping: bool
    require_bayesian_optimization: bool
    require_mc_dropout: bool


@dataclass(frozen=True)
class ModelRoutingConfig:
    business_case: str
    problem_type: str
    active_benchmark_catalog: str
    benchmark_catalogs: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ForecastingConfig:
    date_column: str
    horizon: int
    lag_count: int
    seasonality_period: int


@dataclass(frozen=True)
class CaseConfig:
    dataset_rows: int
    target: str
    ols_target: str
    model_routing: ModelRoutingConfig
    benchmark_models: tuple[str, ...]
    forecasting: ForecastingConfig
    feature_cols: tuple[str, ...]
    outlier_cols: tuple[str, ...]
    vif_cols: tuple[str, ...]
    inference_features: tuple[str, ...]
    value_model_features: tuple[str, ...]
    segment_features: tuple[str, ...]
    test_size: float
    clv_horizons: tuple[int, ...]
    financial_error_asymmetry: float
    contact_cost: float
    retention_success_rate: float
    max_contact_share: float
    parsimony: ParsimonyConfig
    uncertainty: UncertaintyConfig
    temporal_validation: TemporalValidationConfig
    fairness: FairnessConfig
    bandit: BanditConfig
    deep_learning: DeepLearningConfig
    pipeline_health: PipelineHealthConfig


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    config_path: Path
    seed: int
    case: CaseConfig
    tracking: TrackingConfig

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def interim_data_dir(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.processed_data_dir / "figures"


def _resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parents[2]


def load_project_config(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> ProjectConfig:
    """Carga la configuracion externa del proyecto desde YAML."""

    resolved_root = _resolve_project_root(project_root)
    resolved_config = config_path or (resolved_root / "conf" / "settings.yaml")
    payload = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))

    case = payload["case"]
    tracking = payload["tracking"]
    pipeline_health_block = case.get("pipeline_health", {})
    parsimony_block = case.get("parsimony", {})
    uncertainty_block = case.get("uncertainty", {})
    temporal_validation_block = case.get("temporal_validation", {})
    fairness_block = case.get("fairness", {})
    bandit_block = case.get("bandit", {})
    deep_learning_block = case.get("deep_learning", {})
    model_routing_block = case.get("model_routing", {})
    forecasting_block = case.get("forecasting", {})
    pipeline_health_profiles = pipeline_health_block.get("profiles")
    benchmark_catalogs_raw = model_routing_block.get("benchmark_catalogs", {})
    if not isinstance(benchmark_catalogs_raw, dict) or not benchmark_catalogs_raw:
        benchmark_catalogs_raw = {"default": tuple(case.get("benchmark_models", ())) }
    normalized_benchmark_catalogs = {
        str(name): tuple(str(algorithm) for algorithm in algorithms)
        for name, algorithms in benchmark_catalogs_raw.items()
    }
    active_benchmark_catalog = str(model_routing_block.get("active_benchmark_catalog", next(iter(normalized_benchmark_catalogs))))
    if active_benchmark_catalog not in normalized_benchmark_catalogs:
        available_catalogs = ", ".join(sorted(normalized_benchmark_catalogs))
        raise ValueError(
            f"Catalogo de benchmark '{active_benchmark_catalog}' no soportado. Usa uno de: {available_catalogs}."
        )
    resolved_benchmark_models = normalized_benchmark_catalogs[active_benchmark_catalog]
    if isinstance(pipeline_health_profiles, dict) and pipeline_health_profiles:
        active_pipeline_health_profile = str(pipeline_health_block.get("active_profile", "demo"))
        if active_pipeline_health_profile not in pipeline_health_profiles:
            available_profiles = ", ".join(sorted(str(key) for key in pipeline_health_profiles))
            raise ValueError(
                f"Perfil pipeline_health '{active_pipeline_health_profile}' no soportado. Usa uno de: {available_profiles}."
            )
        pipeline_health = pipeline_health_profiles[active_pipeline_health_profile]
        pipeline_health_profile_name = active_pipeline_health_profile
    else:
        pipeline_health = pipeline_health_block
        pipeline_health_profile_name = str(pipeline_health_block.get("active_profile", "default"))
    return ProjectConfig(
        project_root=resolved_root,
        config_path=resolved_config,
        seed=int(payload["seed"]),
        case=CaseConfig(
            dataset_rows=int(case["dataset_rows"]),
            target=str(case["target"]),
            ols_target=str(case["ols_target"]),
            model_routing=ModelRoutingConfig(
                business_case=str(model_routing_block.get("business_case", "classification_general")),
                problem_type=str(model_routing_block.get("problem_type", "classification")),
                active_benchmark_catalog=active_benchmark_catalog,
                benchmark_catalogs=normalized_benchmark_catalogs,
            ),
            benchmark_models=resolved_benchmark_models,
            forecasting=ForecastingConfig(
                date_column=str(forecasting_block.get("date_column", case.get("date_column", "fecha_corte"))),
                horizon=int(forecasting_block.get("horizon", 30)),
                lag_count=int(forecasting_block.get("lag_count", 6)),
                seasonality_period=int(forecasting_block.get("seasonality_period", 12)),
            ),
            feature_cols=tuple(case["feature_cols"]),
            outlier_cols=tuple(case["outlier_cols"]),
            vif_cols=tuple(case["vif_cols"]),
            inference_features=tuple(case["inference_features"]),
            value_model_features=tuple(case["value_model_features"]),
            segment_features=tuple(case["segment_features"]),
            test_size=float(case["test_size"]),
            clv_horizons=tuple(int(item) for item in case["clv_horizons"]),
            financial_error_asymmetry=float(case["financial_error_asymmetry"]),
            contact_cost=float(case["contact_cost"]),
            retention_success_rate=float(case["retention_success_rate"]),
            max_contact_share=float(case["max_contact_share"]),
            parsimony=ParsimonyConfig(
                logistic_feature_steps=tuple(int(item) for item in parsimony_block.get("logistic_feature_steps", (4, 8, 12))),
                roc_auc_tolerance=float(parsimony_block.get("roc_auc_tolerance", 0.015)),
            ),
            uncertainty=UncertaintyConfig(
                bootstrap_iterations=int(uncertainty_block.get("bootstrap_iterations", 40)),
                prediction_interval_alpha=float(uncertainty_block.get("prediction_interval_alpha", 0.10)),
            ),
            temporal_validation=TemporalValidationConfig(
                n_splits=int(temporal_validation_block.get("n_splits", 4)),
                purge_gap_days=int(temporal_validation_block.get("purge_gap_days", 7)),
                embargo_gap_days=int(temporal_validation_block.get("embargo_gap_days", 14)),
            ),
            fairness=FairnessConfig(
                sensitive_columns=tuple(str(item) for item in fairness_block.get("sensitive_columns", ("edad", "region"))),
                age_bins=tuple(int(item) for item in fairness_block.get("age_bins", (30, 45, 60))),
            ),
            bandit=BanditConfig(
                epsilon=float(bandit_block.get("epsilon", 0.10)),
            ),
            deep_learning=DeepLearningConfig(
                enabled=bool(deep_learning_block.get("enabled", True)),
                require_dropout=bool(deep_learning_block.get("require_dropout", True)),
                require_batch_norm=bool(deep_learning_block.get("require_batch_norm", True)),
                require_early_stopping=bool(deep_learning_block.get("require_early_stopping", True)),
                require_bayesian_optimization=bool(deep_learning_block.get("require_bayesian_optimization", True)),
                require_mc_dropout=bool(deep_learning_block.get("require_mc_dropout", True)),
            ),
            pipeline_health=PipelineHealthConfig(
                profile_name=pipeline_health_profile_name,
                freshness_threshold_hours=float(pipeline_health.get("frescura_horas", pipeline_health.get("freshness_threshold_hours", 24.0))),
                count_tolerance_pct=float(pipeline_health.get("count_tolerance_pct", 20.0)),
                critical_freshness_multiplier=float(pipeline_health.get("critical_freshness_multiplier", 2.0)),
                critical_count_tolerance_pct=float(pipeline_health.get("critical_count_tolerance_pct", max(float(pipeline_health.get("count_tolerance_pct", 20.0)) * 2, 40.0))),
                validation_error_alert_pct=float(pipeline_health.get("validation_error_alert_pct", 5.0)),
                validation_error_critical_pct=float(pipeline_health.get("validation_error_critical_pct", 10.0)),
            ),
        ),
        tracking=TrackingConfig(
            enabled=bool(tracking["enabled"]),
            experiment_name=str(tracking["experiment_name"]),
            run_name=str(tracking["run_name"]),
            tracking_dir=(resolved_root / tracking["tracking_dir"]).resolve(),
        ),
    )