"""Boilerplate metodologico para ciencia de datos con rigor estadistico.

Este modulo convierte el framework metodologico del notebook en funciones reutilizables
para proyectos con datos estructurados. Cada funcion devuelve resultados tabulares,
objetos reutilizables y una interpretacion en lenguaje natural para acelerar la toma de
decisiones sin perder trazabilidad estadistica.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, SplineTransformer, StandardScaler
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt, het_white, lilliefors
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsmodelsConvergenceWarning, HessianInversionWarning
from statsmodels.tsa.arima.model import ARIMA

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    from pygam import LinearGAM, LogisticGAM
except ImportError:
    LinearGAM = None
    LogisticGAM = None

try:
    from pyearth import Earth
except ImportError:
    Earth = None

try:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras
except ImportError:
    keras = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

from .configuracion import aplicar_tema_profesional
from .exploracion import resumen_categorico, resumen_numerico
from .limpieza import (
    detectar_outliers_iqr,
    normalizar_nombres_columnas,
    reporte_calidad_datos,
    resumir_nulos,
)
from .visualizacion import grafico_mapa_correlacion, grafico_nulos


ProblemType = Literal["classification", "regression", "forecasting"]

DISTANCE_SENSITIVE_ALGORITHMS = {"knn"}
FORECASTING_ALGORITHMS = {"arima", "prophet", "lstm"}
DEEP_LEARNING_ALGORITHMS = {"mlp", "neural_network", "lstm"}
BUSINESS_CASE_BENCHMARKS: dict[str, tuple[str, ...]] = {
    "banking_classification": ("logistic", "random_forest", "gradient_boosting", "lightgbm", "mlp"),
    "classification_general": ("logistic", "random_forest", "gradient_boosting", "lightgbm", "mlp"),
    "value_prediction": ("linear", "lasso", "xgboost", "knn"),
    "rentals_forecasting": ("arima", "prophet", "lstm"),
}


UNIVERSAL_METHODOLOGY_FRAMEWORKS = [
    {
        "metodologia": "CRISP-DM",
        "fases_principales": "Negocio, Datos, Preparacion, Modelado, Evaluacion, Despliegue",
        "fortalezas_tecnicas": "Alineacion estrategica end-to-end y trazabilidad entre decision y pipeline.",
        "aplicabilidad_2025": "Estandar de oro para proyectos corporativos, consultoria y gobierno analitico.",
    },
    {
        "metodologia": "SEMMA",
        "fases_principales": "Sample, Explore, Modify, Model, Assess",
        "fortalezas_tecnicas": "Profundiza en exploracion, transformacion y seleccion de variables.",
        "aplicabilidad_2025": "Muy util en laboratorios analiticos y experimentacion estadistica intensiva.",
    },
    {
        "metodologia": "KDD",
        "fases_principales": "Seleccion, Limpieza, Transformacion, Mineria, Interpretacion",
        "fortalezas_tecnicas": "Enfatiza descubrimiento de patrones y extraccion de conocimiento.",
        "aplicabilidad_2025": "Adecuado para investigacion aplicada y mineria de grandes volumenes de datos.",
    },
    {
        "metodologia": "TDSP",
        "fases_principales": "Negocio, Adquisicion, Modelado, Despliegue, Aceptacion",
        "fortalezas_tecnicas": "Escalabilidad cloud, colaboracion y disciplina MLOps/DataOps.",
        "aplicabilidad_2025": "Preferido en entornos agiles con CI/CD, observabilidad y despliegue continuo.",
    },
]

UNIVERSAL_PHASE_REFERENCE = [
    {
        "fase": "Ingesta y auditoria",
        "pregunta_critica": "Los datos son representativos, trazables y libres de leakage operativo.",
        "metricas_clave": "missingness, duplicados, drift, PSI, leakage, cardinalidad",
        "visual_analytics": "heatmap de faltantes, drift dashboard, conteos por cohorte",
    },
    {
        "fase": "ETL y estabilizacion",
        "pregunta_critica": "La preparacion preserva significado y evita fuga entre train y test.",
        "metricas_clave": "cobertura, error rate de validacion, skewness, varianza, linaje",
        "visual_analytics": "densidades antes/despues, Sankey ETL, control de volumen",
    },
    {
        "fase": "EDA y diagnostico",
        "pregunta_critica": "Existen confundidores, colinealidad o patrones agregados engañosos.",
        "metricas_clave": "Shapiro-Wilk, Anderson-Darling, Lilliefors, Jarque-Bera, Brown-Forsythe, VIF, Simpson, correlacion segmentada",
        "visual_analytics": "Q-Q plot, histograma, scatter segmentado, heatmap de correlacion",
    },
    {
        "fase": "Modelado y supuestos",
        "pregunta_critica": "El modelo equilibra sesgo-varianza y deja errores diagnosticables.",
        "metricas_clave": "RMSE/MAE o ROC-AUC/F1, Durbin-Watson, Breusch-Pagan, White, Goldfeld-Quandt, Cook, calibracion",
        "visual_analytics": "curvas de aprendizaje, confusion matrix, residuos vs ajustados, scale-location, diagnostico OLS, SHAP beeswarm y dependence plot",
    },
    {
        "fase": "Calibracion e inferencia",
        "pregunta_critica": "La probabilidad es confiable y el efecto es material, no solo significativo.",
        "metricas_clave": "Brier, reliability, Cohen d, eta2, analisis multiverse",
        "visual_analytics": "reliability diagram, estimation plot, tabla multiescenario",
    },
]

UNIVERSAL_MODEL_CATALOG = [
    {
        "familia": "Lineales penalizados",
        "cuando_usar": "Regresion y scorecards donde importa controlar multicolinealidad y priorizar parsimonia.",
        "implementacion_practica": "linear, ridge, lasso y elasticnet conviven con AIC/BIC para promover especificaciones simples cuando el gap de error es pequeno.",
    },
    {
        "familia": "Modelos por similitud",
        "cuando_usar": "Casos de comparables, tasacion o vecindad local donde la distancia entre observaciones es interpretable.",
        "implementacion_practica": "knn requiere escalado obligatorio dentro del pipeline para no sesgar la distancia por magnitud.",
    },
    {
        "familia": "Gradient Boosting / XGBoost / LightGBM / CatBoost",
        "cuando_usar": "Datos tabulares con no linealidad e interacciones relevantes.",
        "implementacion_practica": "gradient_boosting sigue como baseline interno y la v4 habilita ensambles externos cuando el entorno soporta las dependencias.",
    },
    {
        "familia": "Random Forest",
        "cuando_usar": "Baseline robusto con buena tolerancia a ruido y relaciones no lineales.",
        "implementacion_practica": "random_forest para clasificacion o regresion con permutation importance.",
    },
    {
        "familia": "GAM / MARS",
        "cuando_usar": "Tendencias suaves o relaciones no lineales que requieren mas interpretabilidad que un ensamble profundo.",
        "implementacion_practica": "GAM y MARS quedan como extensiones opcionales del catalogo universal cuando las dependencias estan disponibles.",
    },
    {
        "familia": "MLP / LSTM",
        "cuando_usar": "Patrones no lineales complejos o secuencias con memoria temporal donde los modelos lineales ya no bastan.",
        "implementacion_practica": "MLP corre en la stack tabular; LSTM activa una compuerta adicional de gobernanza para early stopping, dropout y monitoreo reforzado.",
    },
    {
        "familia": "ARIMA / Prophet",
        "cuando_usar": "Pronostico temporal con autocorrelacion, estacionalidad o changepoints.",
        "implementacion_practica": "La v4 incorpora un pipeline temporal especifico con split cronologico y soporte de Purga + Embargo para validacion conservadora.",
    },
    {
        "familia": "K-Means",
        "cuando_usar": "Segmentacion RFM o clustering exploratorio con distancia interpretable.",
        "implementacion_practica": "Se recomienda combinar con silueta, elbow y perfilado posterior.",
    },
    {
        "familia": "NLP clasico o Transformers",
        "cuando_usar": "Reseñas, sentimiento o texto libre como insumo de negocio.",
        "implementacion_practica": "Fuera de este modulo tabular, pero integrado en la taxonomia metodologica.",
    },
]

PIPELINE_HEALTH_REFERENCE = [
    {
        "metrica": "frescura_horas",
        "objetivo": "Controlar latencia entre la ultima carga y el momento de uso.",
    },
    {
        "metrica": "desviacion_conteo_pct",
        "objetivo": "Detectar cambios de volumen fuera del rango esperado.",
    },
    {
        "metrica": "validation_error_rate",
        "objetivo": "Capturar errores silenciosos en reglas de negocio durante ETL.",
    },
]


def _emit_interpretation(message: str, verbose: bool) -> None:
    # Centraliza la salida humana para que todas las funciones hablen el mismo idioma.
    if verbose:
        print(message)


def _ensure_dataframe(df: pd.DataFrame) -> None:
    # Corta pronto si la entrada no permite aplicar estadistica de forma segura.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esperaba un pandas.DataFrame como entrada.")
    if df.empty:
        raise ValueError("El dataframe esta vacio. No hay informacion que analizar.")


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    # Obliga a declarar las dependencias minimas antes de entrar en calculos costosos.
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"No se encontraron las columnas requeridas: {missing}")


def _numeric_series(series: pd.Series, name: str | None = None) -> pd.Series:
    # Convierte a numerico y limpia nulos para que los tests no fallen por tipos mixtos.
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        label = name or series.name or "serie"
        raise ValueError(f"La columna '{label}' no contiene suficientes valores numericos validos.")
    return numeric


def _infer_problem_type(y: pd.Series) -> ProblemType:
    # Usa una heuristica conservadora para elegir entre clasificacion y regresion.
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"

    unique_values = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and unique_values <= 10:
        return "classification"
    return "regression"


def _resolve_scaler(scaler: Literal["standard", "robust", "none"]) -> Any:
    # Traduce una opcion declarativa a un objeto reutilizable del pipeline.
    if scaler == "standard":
        return StandardScaler()
    if scaler == "robust":
        return RobustScaler()
    if scaler == "none":
        return "passthrough"
    raise ValueError("Escalador no soportado. Usa 'standard', 'robust' o 'none'.")


def resolve_business_case_benchmark_models(
    business_case: str | None = None,
    benchmark_models: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resuelve el benchmark activo segun el caso de negocio o una lista explicita."""
    if benchmark_models:
        return tuple(str(algorithm).strip().lower() for algorithm in benchmark_models)
    normalized_business_case = str(business_case or "classification_general").strip().lower()
    return BUSINESS_CASE_BENCHMARKS.get(normalized_business_case, BUSINESS_CASE_BENCHMARKS["classification_general"])


def _normalize_algorithm_name(algorithm: str) -> str:
    # Normaliza la declaracion del algoritmo para resolver catalogos de forma estable.
    return str(algorithm).strip().lower()


def _resolve_effective_scaler(
    algorithm: str,
    scaler: Literal["standard", "robust", "none"],
) -> Literal["standard", "robust", "none"]:
    # Obliga escalado en modelos de distancia para proteger la metrica de similitud.
    normalized_algorithm = _normalize_algorithm_name(algorithm)
    if normalized_algorithm in DISTANCE_SENSITIVE_ALGORITHMS and scaler == "none":
        return "standard"
    return scaler


def _require_dependency(dependency: Any, dependency_name: str, algorithm: str) -> Any:
    # Falla con un mensaje accionable cuando un algoritmo opcional no esta disponible.
    if dependency is None:
        raise ValueError(
            f"El algoritmo '{algorithm}' requiere la dependencia opcional '{dependency_name}'. Instalala en requirements.txt antes de usar este benchmark."
        )
    return dependency


def _build_mars_surrogate(
    problem_type: Literal["classification", "regression"],
    random_state: int,
) -> Pipeline:
    """Construye una aproximacion spline-based a MARS cuando py-earth no esta disponible."""
    final_estimator: Any
    if problem_type == "classification":
        final_estimator = LogisticRegression(max_iter=5000, random_state=random_state, solver="lbfgs")
    else:
        final_estimator = Ridge(alpha=1.0, random_state=random_state)
    return Pipeline(
        steps=[
            ("basis", SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="linear")),
            ("model", final_estimator),
        ]
    )


def _build_mars_estimator(
    problem_type: Literal["classification", "regression"],
    random_state: int,
) -> Any:
    """Devuelve MARS nativo cuando existe y un surrogate mantenible cuando no."""
    if Earth is not None:
        return Earth()
    return _build_mars_surrogate(problem_type, random_state)


def _build_supervised_estimator(problem_type: Literal["classification", "regression"], algorithm: str, random_state: int) -> Any:
    """Construye el estimador supervisado homologado para benchmark, validacion y scoring."""
    normalized_algorithm = _normalize_algorithm_name(algorithm)
    classification_builders: dict[str, Any] = {
        "logistic": lambda: LogisticRegression(max_iter=5000, random_state=random_state, solver="liblinear"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=random_state),
        "lightgbm": lambda: _require_dependency(LGBMClassifier, "lightgbm", normalized_algorithm)(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
        ),
        "xgboost": lambda: _require_dependency(XGBClassifier, "xgboost", normalized_algorithm)(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
        ),
        "catboost": lambda: _require_dependency(CatBoostClassifier, "catboost", normalized_algorithm)(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
        "knn": lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "gam": lambda: _require_dependency(LogisticGAM, "pygam", normalized_algorithm)(),
        "mars": lambda: _build_mars_estimator("classification", random_state),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.0008,
            early_stopping=True,
            max_iter=800,
            random_state=random_state,
        ),
        "neural_network": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.0008,
            early_stopping=True,
            max_iter=800,
            random_state=random_state,
        ),
    }
    regression_builders: dict[str, Any] = {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(random_state=random_state),
        "lasso": lambda: Lasso(random_state=random_state),
        "elasticnet": lambda: ElasticNet(random_state=random_state),
        "random_forest": lambda: RandomForestRegressor(n_estimators=400, random_state=random_state),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=random_state),
        "xgboost": lambda: _require_dependency(XGBRegressor, "xgboost", normalized_algorithm)(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        ),
        "lightgbm": lambda: _require_dependency(LGBMRegressor, "lightgbm", normalized_algorithm)(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
        ),
        "catboost": lambda: _require_dependency(CatBoostRegressor, "catboost", normalized_algorithm)(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
        "knn": lambda: KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "gam": lambda: _require_dependency(LinearGAM, "pygam", normalized_algorithm)(),
        "mars": lambda: _build_mars_estimator("regression", random_state),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=0.0005,
            learning_rate_init=0.001,
            early_stopping=True,
            max_iter=1200,
            random_state=random_state,
        ),
        "neural_network": lambda: MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=0.0005,
            learning_rate_init=0.001,
            early_stopping=True,
            max_iter=1200,
            random_state=random_state,
        ),
    }
    model_builders = classification_builders if problem_type == "classification" else regression_builders
    if normalized_algorithm not in model_builders:
        raise ValueError(
            f"Algoritmo '{algorithm}' no soportado para '{problem_type}'. Revisa problem_type y benchmark_models del caso."
        )
    return model_builders[normalized_algorithm]()


def _build_temporal_supervised_frame(
    df: pd.DataFrame,
    target: str,
    date_column: str,
    features: Sequence[str],
    lag_count: int,
    seasonality_period: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Construye una matriz supervisada temporal con lags y rolling stats del objetivo."""
    base_columns = list(dict.fromkeys([date_column, target, *features]))
    working = df[base_columns].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column, target]).sort_values(date_column).reset_index(drop=True)

    generated_features: list[str] = []
    target_numeric = pd.to_numeric(working[target], errors="coerce")
    for lag in range(1, max(int(lag_count), 1) + 1):
        column_name = f"{target}_lag_{lag}"
        working[column_name] = target_numeric.shift(lag)
        generated_features.append(column_name)

    working[f"{target}_rolling_mean_3"] = target_numeric.shift(1).rolling(3).mean()
    generated_features.append(f"{target}_rolling_mean_3")
    if seasonality_period > 1:
        seasonal_column = f"{target}_seasonal_lag_{seasonality_period}"
        working[seasonal_column] = target_numeric.shift(seasonality_period)
        generated_features.append(seasonal_column)

    supervised_features = list(dict.fromkeys([*features, *generated_features]))
    working = working.dropna(subset=supervised_features + [target]).reset_index(drop=True)
    return working, supervised_features


def _train_temporal_forecasting_model(
    df: pd.DataFrame,
    target: str,
    algorithm: str,
    features: Sequence[str] | None,
    test_size: float,
    date_column: str | None,
    random_state: int,
    lag_count: int,
    seasonality_period: int,
    forecast_horizon: int | None,
    verbose: bool,
) -> dict[str, Any]:
    """Entrena un modelo temporal especifico para forecasting con split cronologico."""
    if date_column is None:
        raise ValueError("Los modelos de forecasting requieren date_column para respetar el orden temporal.")

    feature_list = list(features) if features is not None else [
        column for column in df.columns if column not in {target, date_column}
    ]
    working, supervised_features = _build_temporal_supervised_frame(
        df=df,
        target=target,
        date_column=date_column,
        features=feature_list,
        lag_count=lag_count,
        seasonality_period=seasonality_period,
    )
    if len(working) < max(60, lag_count * 8):
        raise ValueError("No hay suficiente historia limpia para entrenar el pipeline temporal v4.")

    effective_horizon = max(int(forecast_horizon or 1), 1)
    holdout_rows = max(effective_horizon, int(np.ceil(len(working) * test_size)))
    if holdout_rows >= len(working):
        holdout_rows = max(1, len(working) // 4)

    train_df = working.iloc[:-holdout_rows].copy()
    test_df = working.iloc[-holdout_rows:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("El split temporal no dejo suficientes filas para train/test.")

    normalized_algorithm = _normalize_algorithm_name(algorithm)
    predictions: np.ndarray
    fitted_model: Any

    if normalized_algorithm == "arima":
        fitted_model = ARIMA(pd.to_numeric(train_df[target], errors="coerce").astype(float), order=(1, 1, 1)).fit()
        predictions = np.asarray(fitted_model.forecast(steps=len(test_df)), dtype=float)
    elif normalized_algorithm == "prophet":
        prophet_class = _require_dependency(Prophet, "prophet", normalized_algorithm)
        prophet_train = train_df[[date_column, target]].rename(columns={date_column: "ds", target: "y"})
        fitted_model = prophet_class(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        for feature in feature_list:
            if feature in train_df.columns and pd.api.types.is_numeric_dtype(train_df[feature]):
                fitted_model.add_regressor(feature)
        fitted_model.fit(prophet_train.assign(**{feature: train_df[feature].to_numpy() for feature in feature_list if feature in train_df.columns and pd.api.types.is_numeric_dtype(train_df[feature])}))
        future = test_df[[date_column]].rename(columns={date_column: "ds"}).copy()
        for feature in feature_list:
            if feature in test_df.columns and pd.api.types.is_numeric_dtype(test_df[feature]):
                future[feature] = test_df[feature].to_numpy()
        forecast = fitted_model.predict(future)
        predictions = forecast["yhat"].to_numpy(dtype=float)
    elif normalized_algorithm == "lstm":
        keras_module = _require_dependency(keras, "tensorflow", normalized_algorithm)
        sequence_columns = [column for column in supervised_features if column.startswith(f"{target}_lag_")]
        X_train = train_df[sequence_columns].to_numpy(dtype=float).reshape(len(train_df), len(sequence_columns), 1)
        X_test = test_df[sequence_columns].to_numpy(dtype=float).reshape(len(test_df), len(sequence_columns), 1)
        y_train = pd.to_numeric(train_df[target], errors="coerce").astype(float).to_numpy()
        y_test = pd.to_numeric(test_df[target], errors="coerce").astype(float).to_numpy()
        fitted_model = keras_module.Sequential(
            [
                keras_module.layers.Input(shape=(len(sequence_columns), 1)),
                keras_module.layers.LSTM(32, dropout=0.15, recurrent_dropout=0.0),
                keras_module.layers.Dense(16, activation="relu"),
                keras_module.layers.Dense(1),
            ]
        )
        fitted_model.compile(optimizer="adam", loss="mse")
        callbacks = [
            keras_module.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ]
        fitted_model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=80,
            batch_size=min(32, max(8, len(train_df) // 6)),
            verbose=0,
            callbacks=callbacks,
        )
        predictions = fitted_model.predict(X_test, verbose=0).reshape(-1)
    else:
        raise ValueError(
            f"Algoritmo temporal '{algorithm}' no soportado. Usa uno de: {sorted(FORECASTING_ALGORITHMS)}."
        )

    actual = pd.to_numeric(test_df[target], errors="coerce").astype(float).to_numpy()
    residuals = actual - predictions
    metrics = {
        "mae": round(float(mean_absolute_error(actual, predictions)), 4),
        "rmse": round(float(mean_squared_error(actual, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(actual, predictions)), 4),
        "error_medio": round(float(np.mean(residuals)), 4),
    }
    prediction_frame = pd.DataFrame(
        {
            date_column: test_df[date_column].reset_index(drop=True),
            "actual": actual,
            "predicted": np.asarray(predictions, dtype=float),
        }
    )
    feature_importance = pd.DataFrame(
        {
            "feature": supervised_features,
            "importance_mean": np.zeros(len(supervised_features), dtype=float),
            "importance_std": np.zeros(len(supervised_features), dtype=float),
        }
    )
    interpretation = (
        f"Se entreno un modelo temporal '{normalized_algorithm}' con split cronologico sobre '{date_column}'. "
        f"La lectura principal debe apoyarse en RMSE {metrics['rmse']:.4f} y MAE {metrics['mae']:.4f}, no en accuracy tabular. "
        "Para validacion formal de este caso se debe complementar con Purga y Embargo antes de promover el forecast a operacion."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "pipeline": fitted_model,
        "problem_type": "forecasting",
        "algorithm": normalized_algorithm,
        "metrics": pd.DataFrame([metrics]),
        "predictions": prediction_frame,
        "feature_importance": feature_importance,
        "X_test": test_df[supervised_features].reset_index(drop=True),
        "y_test": pd.Series(actual),
        "probability_calibration": None,
        "calibration_comparison": pd.DataFrame(),
        "interpretation": interpretation,
        "preprocessing": {
            "date_column": date_column,
            "lag_count": int(lag_count),
            "seasonality_period": int(seasonality_period),
            "supervised_features": supervised_features,
        },
    }


def _summarize_metric(metric_name: str, value: float) -> str:
    # Convierte metrica cruda en una lectura ejecutiva breve y consistente.
    if metric_name == "roc_auc":
        if value >= 0.8:
            return "Ranking solido: el modelo separa bastante bien las clases."
        if value >= 0.7:
            return "Ranking razonable: hay senal, pero aun puede haber solapamiento importante."
        return "Capacidad de ranking debil: conviene revisar variables, leakage o especificacion."

    if metric_name == "r2":
        if value >= 0.7:
            return "La varianza explicada es alta para un baseline generalista."
        if value >= 0.4:
            return "La explicacion es util, pero todavia hay estructura no capturada."
        if value >= 0:
            return "El modelo mejora ligeramente al promedio, pero la capacidad explicativa es limitada."
        return "El modelo rinde peor que predecir la media del objetivo."

    if metric_name == "f1":
        if value >= 0.8:
            return "El equilibrio entre precision y recall es fuerte."
        if value >= 0.65:
            return "El equilibrio es aceptable para un primer baseline."
        return "La clasificacion aun es fragil; revisa desbalance, variables y umbral."

    return "Interpreta esta metrica en funcion del coste del error y del contexto del proyecto."


def _cohen_d(group_a: pd.Series, group_b: pd.Series) -> float:
    # Resume la diferencia media en unidades de desviacion tipica combinada.
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")

    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))
    if pooled_std == 0:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def _eta_squared(groups: Sequence[pd.Series]) -> float:
    # Estima cuanta varianza total queda explicada por la separacion entre grupos.
    clean_groups = [group.dropna() for group in groups if not group.dropna().empty]
    if len(clean_groups) < 2:
        return float("nan")

    combined = pd.concat(clean_groups, ignore_index=True)
    grand_mean = combined.mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in clean_groups)
    ss_total = ((combined - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


class StableIterativeImputer(BaseEstimator, TransformerMixin):
    """Wrapper pragmatica sobre IterativeImputer para usarla en pipelines sin ruido excesivo."""

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 30,
        tol: float = 5e-3,
        initial_strategy: str = "median",
        skip_complete: bool = True,
    ) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.skip_complete = skip_complete

    def _build_delegate(self) -> Any:
        from sklearn.experimental import enable_iterative_imputer  # type: ignore  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer(
            random_state=self.random_state,
            sample_posterior=False,
            max_iter=self.max_iter,
            tol=self.tol,
            initial_strategy=self.initial_strategy,
            skip_complete=self.skip_complete,
        )

    def fit(self, X: Any, y: Any = None) -> "StableIterativeImputer":
        # Se silencian warnings no bloqueantes tras endurecer la configuracion para que el notebook solo muestre incidencias accionables.
        self.delegate_ = self._build_delegate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SklearnConvergenceWarning)
            self.delegate_.fit(X, y)
        self.n_features_in_ = getattr(self.delegate_, "n_features_in_", None)
        self.n_iter_ = getattr(self.delegate_, "n_iter_", None)
        self.hit_iteration_limit_ = bool(self.n_iter_ is not None and self.n_iter_ >= self.max_iter)
        return self

    def transform(self, X: Any) -> Any:
        return self.delegate_.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self.delegate_, "feature_names_in_", [])
        return np.asarray(input_features, dtype=object)


def _resolve_numeric_imputer(strategy: Literal["mean", "median", "knn", "iterative"]) -> Any:
    # Permite escalar el rigor de imputacion sin cambiar el contrato del pipeline principal.
    if strategy in {"mean", "median"}:
        return SimpleImputer(strategy=strategy)
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    if strategy == "iterative":
        return StableIterativeImputer(random_state=42, max_iter=30, tol=5e-3, initial_strategy="median")
    raise ValueError("Imputador numerico no soportado. Usa 'mean', 'median', 'knn' o 'iterative'.")


def _safe_distribution_ratio(numerator: pd.Series, denominator: pd.Series, epsilon: float = 1e-6) -> pd.Series:
    # Evita divisiones por cero al comparar distribuciones historicas y actuales.
    numerator = numerator.astype(float).clip(lower=epsilon)
    denominator = denominator.astype(float).clip(lower=epsilon)
    return numerator / denominator


def _calculate_population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    # Resume drift distribucional en una sola metrica comparable entre columnas.
    ref_numeric = pd.to_numeric(reference, errors="coerce")
    cur_numeric = pd.to_numeric(current, errors="coerce")

    if ref_numeric.notna().sum() >= max(20, bins * 2) and cur_numeric.notna().sum() >= max(20, bins * 2):
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(ref_numeric.dropna().quantile(quantiles).to_numpy())
        if len(edges) < 3:
            lower = float(ref_numeric.min())
            upper = float(ref_numeric.max())
            if np.isclose(lower, upper):
                return 0.0
            edges = np.linspace(lower, upper, bins + 1)

        ref_bins = pd.cut(ref_numeric, bins=edges, include_lowest=True, duplicates="drop")
        cur_bins = pd.cut(cur_numeric, bins=edges, include_lowest=True, duplicates="drop")
        categories = ref_bins.cat.categories
        ref_dist = ref_bins.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        cur_dist = cur_bins.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    else:
        ref_text = reference.astype(str).fillna("<NA>")
        cur_text = current.astype(str).fillna("<NA>")
        categories = sorted(set(ref_text.unique()).union(cur_text.unique()))
        ref_dist = ref_text.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        cur_dist = cur_text.value_counts(normalize=True).reindex(categories, fill_value=0.0)

    ratio = _safe_distribution_ratio(cur_dist, ref_dist)
    psi = ((cur_dist - ref_dist) * np.log(ratio)).sum()
    return float(psi)


def get_universal_methodology_reference(verbose: bool = False) -> dict[str, pd.DataFrame]:
    """Expone tablas maestras del framework metodologico universal.

    Entradas:
        verbose: Si es True, imprime una lectura ejecutiva resumida.

    Salidas:
        Diccionario con tablas de marcos metodologicos, fases, catalogo de modelos y observabilidad.

    Pruebas ejecutadas:
        No ejecuta contrastes; empaqueta conocimiento metodologico reusable para notebooks y reportes.
    """
    # Centraliza el marco teorico para que notebooks y scripts hablen el mismo idioma metodologico.
    interpretation = (
        "El framework universal combina CRISP-DM, SEMMA, KDD y TDSP con una capa explicita de rigor estadistico, "
        "visual analytics, auditoria estructural de normalidad, diagnostico de dispersion residual y observabilidad para pipelines analiticos modernos."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "frameworks": pd.DataFrame(UNIVERSAL_METHODOLOGY_FRAMEWORKS),
        "phases": pd.DataFrame(UNIVERSAL_PHASE_REFERENCE),
        "model_catalog": pd.DataFrame(UNIVERSAL_MODEL_CATALOG),
        "pipeline_health": pd.DataFrame(PIPELINE_HEALTH_REFERENCE),
        "interpretation": pd.DataFrame([{"mensaje": interpretation}]),
    }


def audit_missingness_mechanism(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict[str, Any]:
    """Audita el patron de faltantes y aproxima la lectura tipo Little MCAR.

    Entradas:
        df: DataFrame fuente.
        columns: Subconjunto opcional de columnas a revisar. Si es None, usa todas.
        alpha: Nivel de significacion para interpretar la prueba aproximada.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con reporte por columna, estadistico MCAR aproximado e interpretacion.

    Pruebas ejecutadas:
        Resumen de missingness por variable y aproximacion operacional inspirada en Little's MCAR test.
    """
    # Diferencia entre faltantes inocuos y faltantes potencialmente informativos antes de imputar.
    _ensure_dataframe(df)
    selected_columns = list(columns) if columns is not None else list(df.columns)
    _ensure_columns(df, selected_columns)

    working = df[selected_columns].copy()
    scope_label = (
        f"el subconjunto auditado de {len(selected_columns)} columnas"
        if columns is not None
        else "el dataset auditado completo"
    )
    missing_share = working.isna().mean().sort_values(ascending=False)
    column_report = pd.DataFrame(
        {
            "columna": missing_share.index,
            "n_faltantes": working.isna().sum().reindex(missing_share.index).to_numpy(),
            "pct_faltantes": (100 * missing_share).round(2).to_numpy(),
        }
    )
    columns_with_missing = column_report[column_report["n_faltantes"] > 0]["columna"].tolist()

    numeric = working.apply(pd.to_numeric, errors="coerce")
    usable_numeric = numeric.loc[:, numeric.notna().sum() >= 5]
    if usable_numeric.empty or usable_numeric.isna().sum().sum() == 0:
        test_result = {
            "method": "Little MCAR aproximado",
            "statistic": 0.0,
            "degrees_of_freedom": 0,
            "p_value": 1.0,
            "is_mcar_compatible": True,
        }
    else:
        patterns = usable_numeric.isna().astype(int).astype(str).agg("".join, axis=1)
        mean_vector = usable_numeric.mean()
        covariance = usable_numeric.cov(min_periods=3)
        statistic = 0.0
        degrees_of_freedom = 0

        for pattern, group in usable_numeric.groupby(patterns):
            observed_columns = [column for flag, column in zip(pattern, usable_numeric.columns) if flag == "0"]
            if not observed_columns:
                continue
            group_mean = group[observed_columns].mean()
            diff = (group_mean - mean_vector[observed_columns]).to_numpy(dtype=float)
            covariance_slice = covariance.loc[observed_columns, observed_columns].to_numpy(dtype=float)
            covariance_inverse = np.linalg.pinv(covariance_slice)
            statistic += float(len(group) * diff.T @ covariance_inverse @ diff)
            degrees_of_freedom += len(observed_columns)

        degrees_of_freedom = max(int(degrees_of_freedom - usable_numeric.shape[1]), 1)
        p_value = float(1 - stats.chi2.cdf(statistic, degrees_of_freedom))
        test_result = {
            "method": "Little MCAR aproximado",
            "statistic": round(float(statistic), 4),
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": round(p_value, 4),
            "is_mcar_compatible": bool(p_value >= alpha),
        }

    if columns_with_missing:
        if test_result["is_mcar_compatible"]:
            interpretation = (
                f"En {scope_label}, los patrones de faltantes son compatibles con MCAR segun la aproximacion aplicada. "
                "El complete-case analysis o una imputacion simple podrian ser defendibles si la perdida de potencia es aceptable."
            )
        else:
            interpretation = (
                f"En {scope_label}, la auditoria sugiere que los faltantes no son plenamente MCAR. "
                "Conviene priorizar KNN o MICE/Iterative Imputation y documentar el mecanismo como MAR o potencialmente MNAR."
            )
    else:
        interpretation = f"No se detectaron faltantes en {scope_label}; la capa de imputacion no es necesaria para este corte."

    _emit_interpretation(interpretation, verbose)
    return {
        "column_report": column_report,
        "little_mcar": pd.DataFrame([test_result]),
        "interpretation": interpretation,
    }


def impute_missing_values(
    df: pd.DataFrame,
    strategy: Literal["median", "mean", "knn", "mice"] = "median",
    columns: Sequence[str] | None = None,
    categorical_strategy: Literal["most_frequent", "constant"] = "most_frequent",
    verbose: bool = True,
) -> dict[str, Any]:
    """Imputa valores faltantes con alternativas simples o multivariables.

    Entradas:
        df: DataFrame fuente.
        strategy: Estrategia numerica principal.
        columns: Subconjunto opcional de columnas a tratar.
        categorical_strategy: Estrategia para categoricas.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con dataframe imputado, resumen antes/despues e interpretacion.

    Pruebas ejecutadas:
        Imputacion simple, KNN o MICE aproximado mediante IterativeImputer.
    """
    # Permite pasar de una imputacion baseline a una mas rica sin reescribir el flujo del proyecto.
    _ensure_dataframe(df)
    target_columns = list(columns) if columns is not None else list(df.columns)
    _ensure_columns(df, target_columns)

    working = df.copy()
    subset = working[target_columns].copy()
    numeric_columns = subset.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in target_columns if column not in numeric_columns]
    before_missing = subset.isna().sum().sum()

    if numeric_columns:
        if strategy in {"median", "mean"}:
            numeric_imputer = SimpleImputer(strategy=strategy)
        elif strategy == "knn":
            numeric_imputer = KNNImputer(n_neighbors=5)
        elif strategy == "mice":
            numeric_imputer = _resolve_numeric_imputer("iterative")
        else:
            raise ValueError("strategy no soportada. Usa 'median', 'mean', 'knn' o 'mice'.")

        working[numeric_columns] = numeric_imputer.fit_transform(working[numeric_columns])

    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy, fill_value="missing")
        working[categorical_columns] = categorical_imputer.fit_transform(working[categorical_columns])

    after_missing = working[target_columns].isna().sum().sum()
    summary = pd.DataFrame(
        [
            {
                "estrategia": strategy,
                "columnas_tratadas": len(target_columns),
                "faltantes_antes": int(before_missing),
                "faltantes_despues": int(after_missing),
            }
        ]
    )
    interpretation = (
        f"Se aplico imputacion '{strategy}' sobre {len(target_columns)} columnas. "
        "Usa KNN cuando esperes vecindad local y MICE/Iterative cuando la estructura multivariable aporte senal para reconstruir faltantes."
    )
    _emit_interpretation(interpretation, verbose)
    return {"data": working, "summary": summary, "interpretation": interpretation}


def evaluate_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    psi_threshold: float = 0.2,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compara dos cortes del dataset para detectar drift de esquema y distribucion.

    Entradas:
        reference_df: Corte historico o baseline.
        current_df: Corte actual a monitorear.
        columns: Subconjunto opcional de columnas comunes a comparar.
        psi_threshold: Umbral de alerta para PSI.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con resumen de esquema, drift por columna e interpretacion.

    Pruebas ejecutadas:
        Schema drift, cambio de volumen y Population Stability Index por variable.
    """
    # Permite decidir si el pipeline esta viendo el mismo fenomeno o una distribucion ya desplazada.
    _ensure_dataframe(reference_df)
    _ensure_dataframe(current_df)

    reference_columns = set(reference_df.columns)
    current_columns = set(current_df.columns)
    added_columns = sorted(current_columns - reference_columns)
    removed_columns = sorted(reference_columns - current_columns)
    common_columns = sorted(reference_columns & current_columns)
    if columns is not None:
        common_columns = [column for column in columns if column in common_columns]

    schema_rows: list[dict[str, Any]] = []
    for column in common_columns:
        reference_dtype = str(reference_df[column].dtype)
        current_dtype = str(current_df[column].dtype)
        schema_rows.append(
            {
                "columna": column,
                "dtype_referencia": reference_dtype,
                "dtype_actual": current_dtype,
                "schema_changed": reference_dtype != current_dtype,
            }
        )

    drift_rows: list[dict[str, Any]] = []
    for column in common_columns:
        psi_value = _calculate_population_stability_index(reference_df[column], current_df[column])
        drift_rows.append(
            {
                "columna": column,
                "psi": round(float(psi_value), 4),
                "drift_severo": bool(psi_value >= psi_threshold),
            }
        )

    schema_report = pd.DataFrame(schema_rows)
    drift_report = pd.DataFrame(drift_rows).sort_values("psi", ascending=False) if drift_rows else pd.DataFrame()
    severe_drift = drift_report[drift_report["drift_severo"]] if not drift_report.empty else pd.DataFrame()

    summary = pd.DataFrame(
        [
            {
                "filas_referencia": len(reference_df),
                "filas_actual": len(current_df),
                "desviacion_conteo_pct": round(100 * (len(current_df) - len(reference_df)) / max(len(reference_df), 1), 2),
                "columnas_agregadas": len(added_columns),
                "columnas_eliminadas": len(removed_columns),
                "columnas_con_drift_severo": int(len(severe_drift)),
            }
        ]
    )

    interpretation_parts: list[str] = []
    if added_columns or removed_columns:
        interpretation_parts.append("Se detecto schema drift entre el corte historico y el actual.")
    if not severe_drift.empty:
        interpretation_parts.append(
            "Hay drift distribucional relevante en variables clave; valida cambio de captura, mezcla poblacional o estacionalidad antes de reutilizar el modelo."
        )
    if not interpretation_parts:
        interpretation_parts.append(
            "No se detecto drift estructural o distribucional severo bajo el umbral PSI definido."
        )
    interpretation = " ".join(interpretation_parts)

    _emit_interpretation(interpretation, verbose)
    return {
        "summary": summary,
        "schema_report": schema_report,
        "drift_report": drift_report,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "interpretation": interpretation,
    }


def report_pipeline_health(
    dataset_name: str,
    updated_at: Any,
    observed_at: Any | None = None,
    expected_rows: int | None = None,
    observed_rows: int | None = None,
    validation_failed_rows: int = 0,
    total_validated_rows: int | None = None,
    freshness_threshold_hours: float = 24.0,
    count_tolerance_pct: float = 20.0,
    critical_freshness_multiplier: float = 2.0,
    critical_count_tolerance_pct: float | None = None,
    validation_error_alert_pct: float = 5.0,
    validation_error_critical_pct: float = 10.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Construye una lectura de salud operativa del pipeline con foco DataOps.

    Entradas:
        dataset_name: Nombre del dataset o activo monitoreado.
        updated_at: Momento de ultima actualizacion valida.
        observed_at: Momento de observacion; si es None usa utcnow.
        expected_rows: Volumen esperado o historico de referencia.
        observed_rows: Volumen actual observado.
        validation_failed_rows: Registros que fallaron reglas de negocio.
        total_validated_rows: Total de registros validados.
        freshness_threshold_hours: Umbral de frescura en horas.
        count_tolerance_pct: Tolerancia maxima de desviacion de volumen.
        critical_freshness_multiplier: Multiplicador para convertir alerta de frescura en bloqueo.
        critical_count_tolerance_pct: Desviacion de conteo a partir de la cual se bloquea el modelado.
        validation_error_alert_pct: Tasa de error de validacion a partir de la cual la ejecucion pasa a alerta.
        validation_error_critical_pct: Tasa de error de validacion a partir de la cual se bloquea el modelado.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con resumen y banderas de estado.

    Pruebas ejecutadas:
        Control de frescura, desviacion de conteo y tasa de error de validacion.
    """
    # Traduce la observabilidad del pipeline a un semaforo legible por negocio y operaciones.
    refreshed_at = pd.Timestamp(updated_at)
    observed_timestamp = pd.Timestamp.utcnow() if observed_at is None else pd.Timestamp(observed_at)
    if refreshed_at.tzinfo is not None:
        refreshed_at = refreshed_at.tz_localize(None)
    if observed_timestamp.tzinfo is not None:
        observed_timestamp = observed_timestamp.tz_localize(None)
    freshness_hours = round(float((observed_timestamp - refreshed_at).total_seconds() / 3600), 2)

    if observed_rows is None:
        observed_rows = total_validated_rows or 0
    if total_validated_rows is None:
        total_validated_rows = max(observed_rows, 1)

    count_deviation_pct = np.nan
    if expected_rows is not None and expected_rows > 0:
        count_deviation_pct = round(100 * (observed_rows - expected_rows) / expected_rows, 2)

    validation_error_rate = round(100 * validation_failed_rows / max(total_validated_rows, 1), 2)
    critical_freshness_hours = freshness_threshold_hours * max(critical_freshness_multiplier, 1.0)
    critical_count_tolerance = (
        critical_count_tolerance_pct
        if critical_count_tolerance_pct is not None
        else max(count_tolerance_pct * 2, 40.0)
    )

    freshness_status = "critico" if freshness_hours > critical_freshness_hours else "alerta" if freshness_hours > freshness_threshold_hours else "ok"
    count_status = (
        "critico"
        if not np.isnan(count_deviation_pct) and abs(count_deviation_pct) > critical_count_tolerance
        else "alerta"
        if not np.isnan(count_deviation_pct) and abs(count_deviation_pct) > count_tolerance_pct
        else "ok"
    )
    validation_status = (
        "critico"
        if validation_error_rate > validation_error_critical_pct
        else "alerta"
        if validation_error_rate > validation_error_alert_pct
        else "ok"
    )
    active_alerts = [
        dimension
        for dimension, status in {
            "frescura": freshness_status,
            "conteo": count_status,
            "validacion": validation_status,
        }.items()
        if status != "ok"
    ]
    critical_issues = int(sum(status == "critico" for status in [freshness_status, count_status, validation_status]))
    if validation_status == "critico":
        decision_operativa = "blocked"
        severidad_global = "critico"
        accion_recomendada = "Bloquear modelado y recalibracion hasta corregir los fallos criticos de validacion del dato."
    elif active_alerts:
        decision_operativa = "degraded"
        severidad_global = "alerta"
        accion_recomendada = "Permitir ejecucion degradada solo para diagnostico; revisar alertas antes de promover decisiones."
    else:
        decision_operativa = "allow"
        severidad_global = "ok"
        accion_recomendada = "Permitir ejecucion completa del pipeline y seguimiento rutinario."

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "frescura_horas": freshness_hours,
                "desviacion_conteo_pct": count_deviation_pct,
                "validation_error_rate": validation_error_rate,
                "estado_frescura": freshness_status,
                "estado_conteo": count_status,
                "estado_validacion": validation_status,
                "alertas_activas": ", ".join(active_alerts) if active_alerts else "ninguna",
                "issues_criticos": critical_issues,
                "severidad_global": severidad_global,
                "decision_operativa": decision_operativa,
                "accion_recomendada": accion_recomendada,
            }
        ]
    )

    interpretation = (
        f"Salud del pipeline '{dataset_name}': frescura = {freshness_hours}h, "
        f"tasa de error de validacion = {validation_error_rate}%. "
        f"Decision operativa = {decision_operativa}. {accion_recomendada}"
    )
    _emit_interpretation(interpretation, verbose)
    return {"summary": summary, "interpretation": interpretation}


def _infer_semantic_family(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    normalized = series.dropna().astype(str).str.strip().str.lower()
    unique_values = set(normalized.unique())
    if unique_values and unique_values.issubset({"si", "no", "true", "false", "0", "1"}):
        return "binary_categorical"
    return "categorical"


def _sample_values(series: pd.Series, limit: int = 3) -> str:
    values = series.dropna().astype(str).head(limit).tolist()
    return " | ".join(values)


def audit_tabular_data_standards(
    df: pd.DataFrame,
    id_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
    null_markers: Sequence[str] = ("NA", "N/A", "NULL", "null", ""),
    verbose: bool = True,
) -> dict[str, Any]:
    """Audita estandares tabulares, metadata minima y consistencia de codificacion.

    Entradas:
        df: DataFrame a revisar.
        id_columns: Identificadores que deben ser unicos y estables.
        date_columns: Columnas temporales que deben parsearse de forma consistente.
        null_markers: Tokens textuales aceptados como nulos lógicos.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con auditorias de cabeceras, tipado potencial, nulos textuales,
        columnas fecha, identificadores y contrato tabular de exportacion.

    Pruebas ejecutadas:
        Reglas de nomenclatura, unicidad de identificadores, consistencia de tokens nulos,
        deteccion de columnas texto que deberian tiparse y parseo de fechas.
    """
    _ensure_dataframe(df)
    id_columns = list(id_columns or [])
    inferred_date_columns = [
        column
        for column in df.columns
        if "fecha" in str(column).lower() or "date" in str(column).lower()
    ]
    date_columns = list(date_columns or inferred_date_columns)
    _ensure_columns(df, [*id_columns, *date_columns])

    normalized_headers = list(normalizar_nombres_columnas(df.head(0)).columns)
    column_name_audit = pd.DataFrame(
        [
            {
                "columna_original": original,
                "columna_estandar": normalized,
                "cumple_nomenclatura": bool(str(original) == normalized),
                "requiere_estandarizacion": bool(str(original) != normalized),
            }
            for original, normalized in zip(df.columns, normalized_headers)
        ]
    )

    normalized_null_markers = {marker.strip().upper() for marker in null_markers if marker is not None}
    text_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    categorical_standard_rows: list[dict[str, Any]] = []
    numeric_like_rows: list[dict[str, Any]] = []
    for column in text_columns:
        values = df[column].dropna().astype(str).str.strip()
        if values.empty:
            continue

        normalized_values = values.str.casefold()
        null_tokens_detected = sorted({token for token in values.unique().tolist() if token.strip().upper() in normalized_null_markers})
        categorical_standard_rows.append(
            {
                "columna": column,
                "niveles_originales": int(values.nunique(dropna=True)),
                "niveles_normalizados": int(normalized_values.nunique(dropna=True)),
                "variantes_espacios_mayusculas": int(
                    max(values.nunique(dropna=True) - normalized_values.nunique(dropna=True), 0)
                ),
                "tokens_nulos_detectados": " | ".join(null_tokens_detected),
                "tokens_nulos_inconsistentes": bool(len(null_tokens_detected) > 1),
            }
        )

        numeric_probe = (
            values.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)
            .str.replace(r"(?<=\d),(?=\d)", ".", regex=True)
            .str.replace("%", "", regex=False)
        )
        numeric_rate = float(pd.to_numeric(numeric_probe, errors="coerce").notna().mean())
        if numeric_rate >= 0.85 and values.nunique(dropna=True) > 5:
            numeric_like_rows.append(
                {
                    "columna": column,
                    "pct_parseable_como_numerica": round(numeric_rate * 100, 2),
                    "accion_recomendada": "Revisar contrato de ingesta; la columna parece numerica almacenada como texto.",
                }
            )

    categorical_standard_audit = pd.DataFrame(categorical_standard_rows)
    numeric_like_report = pd.DataFrame(numeric_like_rows)

    date_rows: list[dict[str, Any]] = []
    for column in date_columns:
        raw = df[column]
        non_null = raw.notna().sum()
        parsed = pd.to_datetime(raw, errors="coerce")
        pattern_count = (
            raw.dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"\d", "9", regex=True)
            .nunique(dropna=True)
        )
        parse_rate = float(parsed.notna().sum() / max(non_null, 1))
        date_rows.append(
            {
                "columna": column,
                "pct_parseo_valido": round(parse_rate * 100, 2),
                "patrones_detectados": int(pattern_count),
                "requiere_normalizacion": bool(parse_rate < 0.98 or pattern_count > 1),
                "fecha_min": parsed.min(),
                "fecha_max": parsed.max(),
            }
        )
    date_audit = pd.DataFrame(date_rows)

    id_rows: list[dict[str, Any]] = []
    for column in id_columns:
        uniqueness_ratio = float(df[column].nunique(dropna=False) / max(len(df), 1))
        id_rows.append(
            {
                "columna": column,
                "pct_unicidad": round(uniqueness_ratio * 100, 2),
                "duplicados": int(df[column].duplicated().sum()),
                "cumple_identificador_unico": bool(df[column].duplicated().sum() == 0),
            }
        )
    id_audit = pd.DataFrame(id_rows)

    export_contract = pd.DataFrame(
        [
            {
                "codificacion_recomendada": "utf-8",
                "cabecera_unica": True,
                "una_observacion_por_fila": True,
                "nulo_estandar": "NA",
                "nomenclatura_columnas": "snake_case",
                "diccionario_datos_procesable": True,
                "metadatos_trazabilidad": True,
            }
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "columnas_fuera_nomenclatura": int(column_name_audit["requiere_estandarizacion"].sum()),
                "columnas_con_variantes_categoricas": int(
                    categorical_standard_audit["variantes_espacios_mayusculas"].gt(0).sum()
                )
                if not categorical_standard_audit.empty
                else 0,
                "columnas_con_tokens_nulos_inconsistentes": int(
                    categorical_standard_audit["tokens_nulos_inconsistentes"].sum()
                )
                if not categorical_standard_audit.empty
                else 0,
                "columnas_fecha_con_alerta": int(date_audit["requiere_normalizacion"].sum())
                if not date_audit.empty
                else 0,
                "identificadores_con_duplicados": int(id_audit["duplicados"].gt(0).sum())
                if not id_audit.empty
                else 0,
                "columnas_texto_que_parecen_numericas": int(len(numeric_like_report)),
            }
        ]
    )

    summary_row = summary.iloc[0]
    messages: list[str] = []
    if int(summary_row["columnas_fuera_nomenclatura"]) > 0:
        messages.append(
            "Hay columnas fuera de snake_case; estandarizarlas mejora consistencia, interoperabilidad y contrato CSV reutilizable."
        )
    if int(summary_row["columnas_con_tokens_nulos_inconsistentes"]) > 0:
        messages.append(
            "Se detectaron marcadores textuales de nulos inconsistentes; conviene consolidarlos bajo un unico token logico como 'NA'."
        )
    if int(summary_row["columnas_fecha_con_alerta"]) > 0:
        messages.append(
            "Al menos una columna fecha necesita normalizacion de formato o parseo antes de confiar en analisis temporales."
        )
    if int(summary_row["identificadores_con_duplicados"]) > 0:
        messages.append(
            "Algun identificador no es unico; revisa la unidad de analisis antes de joins, scorecards o exportaciones operativas."
        )
    if int(summary_row["columnas_texto_que_parecen_numericas"]) > 0:
        messages.append(
            "Se detectaron columnas texto que parecen numericas; tiparlas reduce ambiguedad en EDA, reglas y modelado."
        )
    if not messages:
        messages.append(
            "El dataset cumple razonablemente las reglas tabulares clave: cabeceras estables, tipado interpretable y contrato exportable reproducible."
        )

    interpretation = " ".join(messages)
    _emit_interpretation(interpretation, verbose)
    return {
        "summary": summary,
        "column_name_audit": column_name_audit,
        "categorical_standard_audit": categorical_standard_audit,
        "numeric_like_report": numeric_like_report,
        "date_audit": date_audit,
        "id_audit": id_audit,
        "export_contract": export_contract,
        "interpretation": interpretation,
    }


def build_dataset_data_dictionary(
    df: pd.DataFrame,
    dataset_name: str,
    descriptions: dict[str, str] | None = None,
    id_columns: Sequence[str] | None = None,
    target: str | None = None,
    date_columns: Sequence[str] | None = None,
    source_system: str = "internal_analytics",
    owner: str = "analytics_engineering",
    verbose: bool = True,
) -> dict[str, Any]:
    """Construye un diccionario de datos procesable y un esquema tabular reutilizable.

    Entradas:
        df: DataFrame fuente.
        dataset_name: Nombre logico del dataset.
        descriptions: Glosario opcional por columna.
        id_columns: Identificadores oficiales del dataset.
        target: Columna objetivo si existe.
        date_columns: Columnas temporales relevantes.
        source_system: Sistema o dominio de procedencia del dato.
        owner: Responsable logico del activo analitico.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con tabla de diccionario de datos, metadata tabular y esquema estilo CSVW.

    Pruebas ejecutadas:
        Perfilado columnar, asignacion de rol semantico y empaquetado de metadata procesable.
    """
    _ensure_dataframe(df)
    descriptions = descriptions or {}
    id_columns = list(id_columns or [])
    inferred_date_columns = [
        column
        for column in df.columns
        if "fecha" in str(column).lower() or "date" in str(column).lower()
    ]
    date_columns = list(date_columns or inferred_date_columns)
    _ensure_columns(df, [*id_columns, *date_columns])
    if target is not None:
        _ensure_columns(df, [target])

    normalized_headers = list(normalizar_nombres_columnas(df.head(0)).columns)
    dictionary_rows: list[dict[str, Any]] = []
    schema_columns: list[dict[str, Any]] = []
    for original_name, standardized_name in zip(df.columns, normalized_headers):
        series = df[original_name]
        semantic_family = _infer_semantic_family(series)
        if target is not None and original_name == target:
            role = "target"
        elif original_name in id_columns:
            role = "identifier"
        elif original_name in date_columns:
            role = "timestamp"
        else:
            role = "feature"

        unique_ratio = float(series.nunique(dropna=False) / max(len(df), 1))
        description = descriptions.get(
            original_name,
            f"Variable '{original_name}' del dataset {dataset_name}; requiere definicion funcional especifica si se expone fuera del pipeline.",
        )
        dictionary_rows.append(
            {
                "columna": original_name,
                "columna_estandar": standardized_name,
                "rol": role,
                "familia_semantica": semantic_family,
                "tipo_dato": str(series.dtype),
                "descripcion": description,
                "pct_nulos": round(float(series.isna().mean() * 100), 2),
                "valores_unicos": int(series.nunique(dropna=True)),
                "pct_unicidad": round(unique_ratio * 100, 2),
                "nullable": bool(series.isna().any()),
                "ejemplo_valores": _sample_values(series),
            }
        )
        schema_columns.append(
            {
                "name": standardized_name,
                "titles": original_name,
                "datatype": semantic_family,
                "description": description,
                "required": not bool(series.isna().any()),
                "unique": bool(original_name in id_columns or unique_ratio >= 0.98),
            }
        )

    metadata = {
        "dataset_name": dataset_name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "owner": owner,
        "source_system": source_system,
        "encoding": "utf-8",
        "null_token_standard": "NA",
        "naming_convention": "snake_case",
        "target": target,
        "id_columns": id_columns,
        "date_columns": date_columns,
    }
    schema = {
        "@context": "https://www.w3.org/ns/csvw",
        "url": f"{dataset_name}.csv",
        "tableSchema": {"columns": schema_columns},
        "dialect": {"encoding": "utf-8", "header": True},
    }
    metadata_table = pd.DataFrame([metadata])

    interpretation = (
        f"Se construyo un diccionario de datos procesable para '{dataset_name}' con {len(dictionary_rows)} columnas. "
        "La combinacion de glosario, roles semanticos y esquema tabular facilita trazabilidad, FAIR interno y exportaciones auditables."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "dictionary": pd.DataFrame(dictionary_rows),
        "metadata_table": metadata_table,
        "metadata": metadata,
        "schema": schema,
        "interpretation": interpretation,
    }


def audit_sampling_representativeness(
    df: pd.DataFrame,
    segment_columns: Sequence[str] | None = None,
    target: str | None = None,
    date_column: str | None = None,
    rare_share_threshold: float = 5.0,
    dominance_threshold: float = 60.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Audita cobertura, dominancia segmental y plan de muestreo recomendado.

    Entradas:
        df: DataFrame fuente.
        segment_columns: Dimensiones para revisar subrepresentacion o dominancia.
        target: Objetivo opcional para contrastar tasas por segmento.
        date_column: Columna temporal para revisar cobertura longitudinal.
        rare_share_threshold: Umbral minimo de peso porcentual para considerar un grupo estable.
        dominance_threshold: Umbral maximo a partir del cual un grupo domina el dataset.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con distribuciones por segmento, cobertura temporal, alertas y plan de muestreo.

    Pruebas ejecutadas:
        Conteos por segmento, deteccion de grupos raros/dominantes y recomendacion de
        muestreo aleatorio, estratificado o control temporal por bloques.
    """
    _ensure_dataframe(df)
    segment_columns = list(segment_columns or [])
    requested_columns = [column for column in segment_columns if column in df.columns]
    _ensure_columns(df, requested_columns)
    if target is not None:
        _ensure_columns(df, [target])
    if date_column is not None:
        _ensure_columns(df, [date_column])

    distributions: list[pd.DataFrame] = []
    alert_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []

    for column in requested_columns:
        distribution = (
            df[column]
            .fillna("NA")
            .astype(str)
            .value_counts(dropna=False, normalize=True)
            .mul(100)
            .round(2)
            .rename_axis("valor")
            .reset_index(name="pct_registros")
        )
        counts = (
            df[column]
            .fillna("NA")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("valor")
            .reset_index(name="n_registros")
        )
        distribution = distribution.merge(counts, on="valor", how="left")
        distribution.insert(0, "dimension", column)
        if target is not None:
            target_profile = (
                df.assign(_segment=df[column].fillna("NA").astype(str))
                .groupby("_segment", dropna=False)[target]
                .mean()
                .mul(100)
                .round(2)
                .rename_axis("valor")
                .reset_index(name="pct_objetivo")
            )
            distribution = distribution.merge(target_profile, on="valor", how="left")
        distributions.append(distribution)

        dominant_share = float(distribution["pct_registros"].max()) if not distribution.empty else 0.0
        low_coverage = distribution[distribution["pct_registros"] < rare_share_threshold]
        if dominant_share >= dominance_threshold:
            alert_rows.append(
                {
                    "dimension": column,
                    "hallazgo": "Dominancia segmental",
                    "detalle": f"Un grupo concentra {round(dominant_share, 2)}% de los registros.",
                    "severidad": "Seguimiento",
                }
            )
        if not low_coverage.empty:
            alert_rows.append(
                {
                    "dimension": column,
                    "hallazgo": "Baja cobertura",
                    "detalle": f"Hay {len(low_coverage)} grupos por debajo de {rare_share_threshold}% de representacion.",
                    "severidad": "Alerta",
                }
            )

        recommended_strategy = (
            "muestreo_estratificado"
            if dominant_share >= dominance_threshold or not low_coverage.empty
            else "muestreo_aleatorio_simple"
        )
        plan_rows.append(
            {
                "dimension": column,
                "estrategia_recomendada": recommended_strategy,
                "justificacion": (
                    "Preservar cobertura minima por subgrupo y evitar que segmentos minoritarios desaparezcan de la evaluacion."
                    if recommended_strategy == "muestreo_estratificado"
                    else "La distribucion observada no exige cuotas adicionales para un baseline inicial."
                ),
            }
        )

    segment_distribution = pd.concat(distributions, ignore_index=True) if distributions else pd.DataFrame()
    alerts = pd.DataFrame(alert_rows)

    temporal_coverage = pd.DataFrame()
    if date_column is not None:
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce").dropna().sort_values()
        if parsed_dates.empty:
            temporal_coverage = pd.DataFrame(
                [{"columna": date_column, "dias_cobertura": 0, "max_gap_dias": np.nan, "pct_parseo_valido": 0.0}]
            )
            alert_rows.append(
                {
                    "dimension": date_column,
                    "hallazgo": "Cobertura temporal invalida",
                    "detalle": "La columna temporal no pudo parsearse de forma confiable.",
                    "severidad": "Alerta",
                }
            )
        else:
            coverage_days = int((parsed_dates.max() - parsed_dates.min()).days)
            max_gap_days = int(parsed_dates.diff().dt.days.dropna().max()) if len(parsed_dates) > 1 else 0
            temporal_coverage = pd.DataFrame(
                [
                    {
                        "columna": date_column,
                        "dias_cobertura": coverage_days,
                        "max_gap_dias": max_gap_days,
                        "pct_parseo_valido": round(float(parsed_dates.shape[0] / max(df[date_column].notna().sum(), 1) * 100), 2),
                    }
                ]
            )
            if coverage_days < 180:
                alert_rows.append(
                    {
                        "dimension": date_column,
                        "hallazgo": "Cobertura temporal corta",
                        "detalle": f"Solo hay {coverage_days} dias cubiertos; conviene prudencia en generalizacion y drift.",
                        "severidad": "Seguimiento",
                    }
                )
            plan_rows.append(
                {
                    "dimension": date_column,
                    "estrategia_recomendada": "muestreo_por_bloques_temporales",
                    "justificacion": "Las evaluaciones fuera de muestra deben preservar orden temporal para evitar leakage retrospectivo.",
                }
            )

    alerts = pd.DataFrame(alert_rows)
    sampling_plan = pd.DataFrame(plan_rows)
    summary = pd.DataFrame(
        [
            {
                "dimensiones_segmentadas": int(len(requested_columns)),
                "alertas_segmentales": int(len(alerts[alerts["dimension"].isin(requested_columns)])) if not alerts.empty else 0,
                "estrategias_estratificadas": int(
                    sampling_plan["estrategia_recomendada"].eq("muestreo_estratificado").sum()
                )
                if not sampling_plan.empty
                else 0,
                "cobertura_temporal_dias": int(temporal_coverage.iloc[0]["dias_cobertura"])
                if not temporal_coverage.empty
                else np.nan,
            }
        ]
    )

    messages: list[str] = []
    if not alerts.empty:
        messages.append(
            "La cobertura no es totalmente homogenea entre segmentos; conviene preservar cuotas o estratos al construir muestras analiticas y validaciones."
        )
    else:
        messages.append(
            "No se detectaron sesgos de cobertura severos en las dimensiones auditadas para una lectura inicial del caso."
        )
    if date_column is not None and not temporal_coverage.empty:
        messages.append(
            "La validacion debe respetar orden temporal para no mezclar informacion futura con observaciones historicas."
        )

    interpretation = " ".join(messages)
    _emit_interpretation(interpretation, verbose)
    return {
        "summary": summary,
        "segment_distribution": segment_distribution,
        "temporal_coverage": temporal_coverage,
        "alerts": alerts,
        "sampling_plan": sampling_plan,
        "interpretation": interpretation,
    }


def detect_simpsons_paradox(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str,
    method: Literal["pearson", "spearman"] = "spearman",
    min_group_size: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    """Detecta si la tendencia agregada cambia de signo al segmentar por subgrupos.

    Entradas:
        df: DataFrame fuente.
        x_column: Variable explicativa numerica.
        y_column: Variable respuesta numerica.
        group_column: Segmentacion candidata a confundir la lectura agregada.
        method: Metodo de asociacion usado.
        min_group_size: Tamano minimo por grupo para considerar el resultado estable.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con correlacion agregada, correlaciones por grupo e interpretacion.

    Pruebas ejecutadas:
        Correlacion agregada y segmentada para detectar reversión de signo tipo Simpson.
    """
    # Evita vender una tendencia agregada cuando los subgrupos cuentan una historia opuesta.
    _ensure_dataframe(df)
    _ensure_columns(df, [x_column, y_column, group_column])
    working = df[[x_column, y_column, group_column]].dropna().copy()
    working[x_column] = pd.to_numeric(working[x_column], errors="coerce")
    working[y_column] = pd.to_numeric(working[y_column], errors="coerce")
    working = working.dropna(subset=[x_column, y_column])

    aggregate = analyze_correlation(working, x_column=x_column, y_column=y_column, method=method, verbose=False)
    aggregate_sign = np.sign(aggregate["statistic"])
    group_rows: list[dict[str, Any]] = []
    valid_groups = 0
    opposite_weight = 0

    for group_name, group_data in working.groupby(group_column):
        if len(group_data) < min_group_size:
            continue
        result = analyze_correlation(group_data, x_column=x_column, y_column=y_column, method=method, verbose=False)
        group_sign = np.sign(result["statistic"])
        if aggregate_sign != 0 and group_sign != 0 and group_sign != aggregate_sign:
            opposite_weight += len(group_data)
        valid_groups += 1
        group_rows.append(
            {
                "grupo": group_name,
                "n": len(group_data),
                "coeficiente": round(float(result["statistic"]), 4),
                "p_value": round(float(result["p_value"]), 4),
                "signo_opuesto_al_agregado": bool(aggregate_sign != 0 and group_sign != 0 and group_sign != aggregate_sign),
            }
        )

    group_report = pd.DataFrame(group_rows).sort_values("n", ascending=False) if group_rows else pd.DataFrame()
    paradox_detected = bool(valid_groups >= 2 and opposite_weight >= 0.5 * len(working) and abs(aggregate["statistic"]) >= 0.1)
    if paradox_detected:
        interpretation = (
            "Se detecta una señal compatible con paradoja de Simpson: la asociacion agregada cambia de signo o se debilita de forma sustantiva al segmentar. "
            "Antes de concluir, modela el segmento explicitamente o usa interacciones."
        )
    else:
        interpretation = (
            "No aparece una reversión fuerte de la tendencia agregada bajo la segmentacion analizada. "
            "Aun asi, conserva los graficos segmentados para descartar confundidores relevantes."
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "aggregate": pd.DataFrame([aggregate]),
        "group_report": group_report,
        "paradox_detected": paradox_detected,
        "interpretation": interpretation,
    }


def run_rfe_feature_selection(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str] | None = None,
    problem_type: ProblemType | Literal["auto"] = "auto",
    n_features_to_select: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta Recursive Feature Elimination para priorizar variables estables.

    Entradas:
        df: DataFrame fuente.
        target: Variable objetivo.
        features: Predictores opcionales.
        problem_type: Tipo de problema o deteccion automatica.
        n_features_to_select: Numero de variables finales a conservar.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con ranking de variables, seleccion final e interpretacion.

    Pruebas ejecutadas:
        Recursive Feature Elimination con regresion lineal o logistica segun el objetivo.
    """
    # Reduce ruido de predictores para mejorar generalizacion e interpretacion del bloque final.
    _ensure_dataframe(df)
    _ensure_columns(df, [target])
    feature_list = list(features) if features is not None else [column for column in df.columns if column != target]
    _ensure_columns(df, feature_list)

    working = df[feature_list + [target]].dropna(subset=[target]).copy()
    X = pd.get_dummies(working[feature_list], drop_first=True, dtype=float)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = working[target]
    selected_problem = _infer_problem_type(y) if problem_type == "auto" else problem_type

    if selected_problem == "classification":
        estimator = LogisticRegression(max_iter=5000, random_state=42, solver="liblinear")
    else:
        estimator = LinearRegression()

    n_features = min(max(1, n_features_to_select), X.shape[1])
    selector = RFE(estimator=estimator, n_features_to_select=n_features)
    selector.fit(X, y)

    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "ranking": selector.ranking_,
            "selected": selector.support_,
        }
    ).sort_values(["ranking", "feature"]).reset_index(drop=True)
    selected_features = ranking[ranking["selected"]]["feature"].tolist()
    interpretation = (
        f"RFE selecciono {len(selected_features)} variables como bloque mas parsimonioso para un problema de {selected_problem}. "
        "Usa este ranking como filtro metodologico, no como verdad definitiva aislada del contexto de negocio."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "ranking": ranking,
        "selected_features": selected_features,
        "problem_type": selected_problem,
        "interpretation": interpretation,
    }


def evaluate_probability_calibration(
    model_result: dict[str, Any],
    n_bins: int = 6,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua Brier score y calibracion de modelos de clasificacion binaria.

    Entradas:
        model_result: Salida de train_supervised_model.
        n_bins: Numero de bins para curva y tabla de calibracion.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con metricas, tabla por bin e interpretacion.

    Pruebas ejecutadas:
        Brier score y reliability table sobre probabilidades predichas.
    """
    # Separa ranking de probabilidad bien calibrada para decisiones con umbral y coste del error.
    if "predictions" not in model_result:
        raise ValueError("model_result no contiene predicciones.")

    predictions = model_result["predictions"].copy()
    required_columns = {"actual", "predicted_probability"}
    if not required_columns.issubset(predictions.columns):
        raise ValueError("La calibracion requiere una salida binaria con columna 'predicted_probability'.")
    if predictions["actual"].nunique(dropna=True) != 2:
        raise ValueError("La calibracion automatica esta implementada solo para clasificacion binaria.")

    actual = pd.to_numeric(predictions["actual"], errors="coerce").astype(int)
    probability = pd.to_numeric(predictions["predicted_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(actual, probability))
    raw_brier = np.nan
    calibration_method = None
    calibration_comparison = model_result.get("calibration_comparison")
    if "predicted_probability_raw" in predictions.columns:
        raw_probability = pd.to_numeric(predictions["predicted_probability_raw"], errors="coerce").clip(1e-6, 1 - 1e-6)
        raw_brier = float(brier_score_loss(actual, raw_probability))
    calibration_payload = model_result.get("probability_calibration")
    if calibration_payload is not None:
        calibration_method = calibration_payload.get("method")
    observed, estimated = calibration_curve(actual, probability, n_bins=n_bins)

    calibration_frame = pd.DataFrame({"estimated_probability": estimated, "observed_frequency": observed})
    working = pd.DataFrame({"actual": actual, "probability": probability})
    working["bin"] = pd.qcut(working["probability"], q=min(n_bins, working["probability"].nunique()), duplicates="drop")
    bin_summary = working.groupby("bin", observed=False).agg(
        n=("actual", "size"),
        mean_probability=("probability", "mean"),
        observed_rate=("actual", "mean"),
    ).reset_index()
    bin_summary["calibration_gap"] = (bin_summary["observed_rate"] - bin_summary["mean_probability"]).round(4)

    if not np.isnan(raw_brier):
        delta = raw_brier - brier
        if delta > 0.003:
            interpretation = (
                f"Brier score calibrado = {brier:.4f} frente a {raw_brier:.4f} en bruto. "
                f"La calibracion {calibration_method or 'post-hoc'} mejora materialmente la probabilidad operativa."
            )
        elif delta >= 0:
            interpretation = (
                f"Brier score calibrado = {brier:.4f} frente a {raw_brier:.4f} en bruto. "
                "La calibracion aporta una mejora ligera; sigue revisando bins extremos y estabilidad temporal."
            )
        else:
            interpretation = (
                f"Brier score calibrado = {brier:.4f} frente a {raw_brier:.4f} en bruto. "
                "La calibracion no mejora esta corrida; conviene revisar metodo, tamano del split o especificacion base."
            )
    elif brier <= 0.16:
        interpretation = (
            f"Brier score = {brier:.4f}. La calibracion probabilistica es razonable para decisiones con umbral, aunque siempre conviene revisar bins extremos."
        )
    else:
        interpretation = (
            f"Brier score = {brier:.4f}. La probabilidad aun necesita calibracion adicional o mejor especificacion del modelo."
        )

    _emit_interpretation(interpretation, verbose)
    metrics_payload: dict[str, Any] = {"brier_score": round(brier, 4), "n_bins": len(calibration_frame)}
    if not np.isnan(raw_brier):
        metrics_payload["brier_score_raw"] = round(raw_brier, 4)
        metrics_payload["brier_score_delta"] = round(raw_brier - brier, 4)
    if calibration_method is not None:
        metrics_payload["calibration_method"] = calibration_method
    return {
        "metrics": pd.DataFrame([metrics_payload]),
        "calibration_curve": calibration_frame,
        "bin_summary": bin_summary,
        "comparison": calibration_comparison if isinstance(calibration_comparison, pd.DataFrame) else pd.DataFrame(),
        "interpretation": interpretation,
    }


def _fit_probability_calibrator(
    actual: pd.Series,
    raw_probability: Sequence[float],
    method: Literal["sigmoid", "isotonic"],
    random_state: int,
) -> dict[str, Any] | None:
    """Ajusta un calibrador post-hoc sin reemplazar el pipeline base del modelo."""

    target = pd.to_numeric(actual, errors="coerce").astype(int)
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(target) < 40 or target.nunique(dropna=True) != 2:
        return None

    if method == "sigmoid":
        calibrator = LogisticRegression(max_iter=2000, random_state=random_state)
        calibrator.fit(probability.reshape(-1, 1), target)
        calibrated_probability = calibrator.predict_proba(probability.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        calibrator.fit(probability, target)
        calibrated_probability = calibrator.predict(probability)
    else:
        raise ValueError("Metodo de calibracion no soportado. Usa 'sigmoid' o 'isotonic'.")

    raw_brier = float(brier_score_loss(target, probability))
    calibrated_brier = float(brier_score_loss(target, calibrated_probability))
    return {
        "method": method,
        "estimator": calibrator,
        "calibration_rows": int(len(target)),
        "class_balance": round(float(target.mean() * 100), 2),
        "raw_brier_score": round(raw_brier, 4),
        "calibrated_brier_score": round(calibrated_brier, 4),
        "improvement": round(raw_brier - calibrated_brier, 4),
    }


def _apply_probability_calibrator(
    calibration_payload: dict[str, Any] | None,
    raw_probability: Sequence[float],
) -> np.ndarray:
    """Aplica el calibrador entrenado a un vector de probabilidades crudas."""

    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if calibration_payload is None or calibration_payload.get("estimator") is None:
        return probability

    estimator = calibration_payload["estimator"]
    method = calibration_payload.get("method")
    if method == "sigmoid":
        return np.clip(estimator.predict_proba(probability.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    if method == "isotonic":
        return np.clip(estimator.predict(probability), 1e-6, 1 - 1e-6)
    return probability


def plot_probability_calibration(
    calibration_result: dict[str, Any],
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica la curva de calibracion y el gap por bin.

    Entradas:
        calibration_result: Salida de evaluate_probability_calibration.

    Salidas:
        Figura y arreglo de ejes.

    Pruebas ejecutadas:
        Visualizacion de reliability diagram y gap de calibracion por bin.
    """
    # Deja visible si el modelo sobreconfia o infraconfia en distintos tramos de probabilidad.
    if "calibration_curve" not in calibration_result or "bin_summary" not in calibration_result:
        raise ValueError("calibration_result no tiene la estructura esperada.")

    curve = calibration_result["calibration_curve"]
    bins = calibration_result["bin_summary"]
    aplicar_tema_profesional()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = np.atleast_1d(axes)
    axes[0].plot(curve["estimated_probability"], curve["observed_frequency"], marker="o", color="#0F766E")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#EA580C")
    axes[0].set_title("Reliability diagram")
    axes[0].set_xlabel("Probabilidad predicha")
    axes[0].set_ylabel("Frecuencia observada")

    axes[1].bar(range(len(bins)), bins["calibration_gap"], color="#2563EB")
    axes[1].axhline(0, linestyle="--", color="#EA580C")
    axes[1].set_title("Gap de calibracion por bin")
    axes[1].set_xlabel("Bin")
    axes[1].set_ylabel("Observado - predicho")
    fig.tight_layout()
    return fig, axes


def run_multiverse_analysis(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str] | None = None,
    problem_type: ProblemType | Literal["auto"] = "auto",
    numeric_outlier_columns: Sequence[str] | None = None,
    specifications: Sequence[dict[str, Any]] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta varias especificaciones razonables para medir robustez metodologica.

    Entradas:
        df: DataFrame fuente.
        target: Variable objetivo.
        features: Predictores opcionales.
        problem_type: Tipo de problema o deteccion automatica.
        numeric_outlier_columns: Columnas numericas sobre las que controlar outliers si aplica.
        specifications: Lista de escenarios a ejecutar.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con resumen de escenarios, mejor especificacion e interpretacion.

    Pruebas ejecutadas:
        Reentrena multiples pipelines con decisiones razonables de outliers, transformacion y algoritmo.
    """
    # Fuerza a comprobar si la conclusion depende de una sola combinacion de limpieza y modelado.
    _ensure_dataframe(df)
    _ensure_columns(df, [target])
    feature_list = list(features) if features is not None else [column for column in df.columns if column != target]

    if specifications is None:
        specifications = [
            {
                "name": "logistic_baseline",
                "algorithm": "logistic",
                "numeric_imputer": "median",
                "scaler": "standard",
                "apply_power_transform": False,
                "outlier_method": "clip_iqr",
            },
            {
                "name": "logistic_power_knn",
                "algorithm": "logistic",
                "numeric_imputer": "knn",
                "scaler": "robust",
                "apply_power_transform": True,
                "outlier_method": "winsorize",
            },
            {
                "name": "random_forest_baseline",
                "algorithm": "random_forest",
                "numeric_imputer": "median",
                "scaler": "none",
                "apply_power_transform": False,
                "outlier_method": "clip_iqr",
            },
            {
                "name": "gradient_boosting_iterative",
                "algorithm": "gradient_boosting",
                "numeric_imputer": "iterative",
                "scaler": "none",
                "apply_power_transform": True,
                "outlier_method": "clip_iqr",
            },
        ]

    rows: list[dict[str, Any]] = []
    for specification in specifications:
        prepared_df = df.copy()
        outlier_method = specification.get("outlier_method")
        if outlier_method is not None and numeric_outlier_columns:
            prepared_df = handle_outliers(
                prepared_df,
                columns=numeric_outlier_columns,
                method=outlier_method,
                verbose=False,
            )["data"]

        model = train_supervised_model(
            prepared_df,
            target=target,
            problem_type=problem_type,
            algorithm=specification.get("algorithm", "auto"),
            features=feature_list,
            numeric_imputer=specification.get("numeric_imputer", "median"),
            categorical_imputer=specification.get("categorical_imputer", "most_frequent"),
            scaler=specification.get("scaler", "standard"),
            apply_power_transform=specification.get("apply_power_transform", False),
            power_method=specification.get("power_method", "yeo-johnson"),
            verbose=False,
        )

        metrics_row = model["metrics"].iloc[0].to_dict()
        primary_metric = (
            metrics_row.get("roc_auc")
            if "roc_auc" in metrics_row
            else metrics_row.get("r2", metrics_row.get("f1", metrics_row.get("mae")))
        )
        rows.append(
            {
                "scenario": specification.get("name", specification.get("algorithm", "scenario")),
                "algorithm": model["algorithm"],
                "numeric_imputer": specification.get("numeric_imputer", "median"),
                "outlier_method": outlier_method,
                "power_transform": specification.get("apply_power_transform", False),
                "primary_metric": round(float(primary_metric), 4) if primary_metric is not None else np.nan,
                **{key: round(float(value), 4) for key, value in metrics_row.items()},
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No se generaron escenarios validos en el multiverse analysis.")

    maximize_metric = "roc_auc" if "roc_auc" in summary.columns else "r2" if "r2" in summary.columns else "f1"
    ascending = maximize_metric not in {"roc_auc", "r2", "f1", "accuracy"}
    summary = summary.sort_values(maximize_metric, ascending=ascending).reset_index(drop=True)
    best_scenario = summary.iloc[0].to_dict()
    interpretation = (
        f"El multiverse analysis ejecuto {len(summary)} escenarios. "
        f"La mejor especificacion observada fue '{best_scenario['scenario']}' en terminos de {maximize_metric}. "
        "Si el ranking cambia mucho entre escenarios, la conclusion debe comunicarse como sensible a la especificacion."
    )
    _emit_interpretation(interpretation, verbose)
    return {"summary": summary, "best_scenario": best_scenario, "interpretation": interpretation}


def normalize_competitive_event_probabilities(
    scores: Sequence[float],
    groups: Sequence[Any],
    temperature: float = 1.0,
) -> pd.Series:
    """Normaliza scores dentro de cada evento para obtener probabilidades competitivas.

    Entradas:
        scores: Scores continuos de cada competidor; pueden ser logits, confianza o probabilidad cruda.
        groups: Identificador del evento o carrera al que pertenece cada fila.
        temperature: Temperatura del softmax. Valores mayores suavizan; menores endurecen.

    Salidas:
        Serie de probabilidades que suma 1 dentro de cada grupo.

    Pruebas ejecutadas:
        No ejecuta un contraste estadistico; aplica normalizacion softmax estable por grupo.
    """
    # Reescala scores por evento para que la prediccion respete que solo un competidor puede ganar.
    if temperature <= 0:
        raise ValueError("temperature debe ser mayor que 0 para normalizar probabilidades competitivas.")

    working = pd.DataFrame({"score": pd.to_numeric(scores, errors="coerce"), "group": groups}).copy()
    if working["score"].isna().any():
        raise ValueError("scores contiene valores no numericos o nulos tras la conversion.")
    if working["group"].isna().any():
        raise ValueError("groups contiene valores nulos; cada fila debe pertenecer a un evento valido.")

    probabilities = pd.Series(np.zeros(len(working), dtype=float), index=working.index)
    for _, index in working.groupby("group").groups.items():
        event_scores = working.loc[index, "score"].to_numpy(dtype=float) / temperature
        event_scores = event_scores - np.max(event_scores)
        exp_scores = np.exp(event_scores)
        denominator = exp_scores.sum()
        if denominator <= 0 or np.isnan(denominator):
            probabilities.loc[index] = 1.0 / len(index)
        else:
            probabilities.loc[index] = exp_scores / denominator

    return probabilities


def evaluate_competitive_event_predictions(
    predictions: pd.DataFrame,
    group_column: str,
    target_column: str = "actual",
    probability_column: str = "competitive_probability",
    top_k: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua predicciones competitivas a nivel evento con metricas de ranking y probabilidad.

    Entradas:
        predictions: DataFrame con una fila por competidor y columnas de grupo, objetivo y probabilidad.
        group_column: Identificador de evento o carrera.
        target_column: Columna binaria donde 1 representa al ganador real.
        probability_column: Probabilidad competitiva del competidor dentro del evento.
        top_k: Profundidad para medir hit rate y NDCG truncado.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con metricas agregadas, reporte por evento, predicciones ordenadas e interpretacion.

    Pruebas ejecutadas:
        Top-1 accuracy, hit rate top-k, MRR, NDCG@k, log-loss multicategoria por evento y Brier score.
    """
    # Resume el rendimiento donde importa en deportes competitivos: orden correcto y probabilidad bien calibrada por carrera.
    _ensure_dataframe(predictions)
    _ensure_columns(predictions, [group_column, target_column, probability_column])
    if top_k < 1:
        raise ValueError("top_k debe ser al menos 1.")

    working = predictions[[group_column, target_column, probability_column]].copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working[probability_column] = pd.to_numeric(working[probability_column], errors="coerce")
    working = working.dropna(subset=[group_column, target_column, probability_column])
    working[target_column] = working[target_column].astype(int)

    event_rows: list[dict[str, Any]] = []
    sorted_predictions = predictions.copy()
    sorted_predictions["predicted_rank"] = (
        sorted_predictions.groupby(group_column)[probability_column]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    sorted_predictions["is_top_pick"] = (sorted_predictions["predicted_rank"] == 1).astype(int)

    for event_id, event in sorted_predictions.groupby(group_column):
        ordered = event.sort_values(probability_column, ascending=False).reset_index(drop=True)
        actual = pd.to_numeric(ordered[target_column], errors="coerce").fillna(0).astype(int).to_numpy()
        probabilities = np.clip(
            pd.to_numeric(ordered[probability_column], errors="coerce").fillna(0).to_numpy(dtype=float),
            1e-12,
            1.0,
        )
        if len(ordered) < 2 or actual.sum() != 1:
            continue

        winner_rank = int(np.flatnonzero(actual == 1)[0] + 1)
        winner_probability = float(probabilities[winner_rank - 1])
        field_size = int(len(ordered))
        k_effective = min(top_k, field_size)
        ndcg_value = float(ndcg_score(actual.reshape(1, -1), probabilities.reshape(1, -1), k=k_effective))
        brier_value = float(brier_score_loss(actual, probabilities))

        event_rows.append(
            {
                group_column: event_id,
                "field_size": field_size,
                "winner_rank": winner_rank,
                "winner_probability": round(winner_probability, 6),
                "top1_hit": float(winner_rank == 1),
                f"top{k_effective}_hit": float(winner_rank <= k_effective),
                "reciprocal_rank": round(1.0 / winner_rank, 6),
                "event_log_loss": round(float(-np.log(winner_probability)), 6),
                "event_brier": round(brier_value, 6),
                f"ndcg@{k_effective}": round(ndcg_value, 6),
            }
        )

    if not event_rows:
        raise ValueError(
            "No hay eventos validos para evaluar. Revisa que cada grupo tenga al menos dos competidores y exactamente un ganador."
        )

    event_report = pd.DataFrame(event_rows)
    k_metric = f"top{min(top_k, int(event_report['field_size'].max()))}_hit"
    ndcg_metric = f"ndcg@{min(top_k, int(event_report['field_size'].max()))}"
    metrics = {
        "n_events": int(len(event_report)),
        "mean_field_size": round(float(event_report["field_size"].mean()), 3),
        "top1_accuracy": round(float(event_report["top1_hit"].mean()), 4),
        k_metric: round(float(event_report[k_metric].mean()), 4),
        "mean_reciprocal_rank": round(float(event_report["reciprocal_rank"].mean()), 4),
        ndcg_metric: round(float(event_report[ndcg_metric].mean()), 4),
        "winner_log_loss": round(float(event_report["event_log_loss"].mean()), 4),
        "winner_brier": round(float(event_report["event_brier"].mean()), 4),
        "mean_winner_probability": round(float(event_report["winner_probability"].mean()), 4),
    }

    if metrics["top1_accuracy"] >= 0.4 and metrics[ndcg_metric] >= 0.8:
        interpretation = (
            "El modelo competitivo ordena bien los participantes y asigna probabilidad util al ganador real. "
            "La metodologia por evento parece accionable para ranking y priorizacion pre-carrera."
        )
    elif metrics["top1_accuracy"] >= 0.25:
        interpretation = (
            "El ranking por evento aporta senal util, pero todavia conviene reforzar variables, calibracion o estabilidad entre carreras."
        )
    else:
        interpretation = (
            "La capacidad de ordenar ganadores por carrera aun es debil. Revisa leakage, representacion del evento y comparacion contra baseline de mercado."
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "metrics": pd.DataFrame([metrics]),
        "event_report": event_report.sort_values(["winner_rank", "winner_probability"], ascending=[True, False]),
        "predictions": sorted_predictions.sort_values([group_column, "predicted_rank"]).reset_index(drop=True),
        "interpretation": interpretation,
    }


def build_competitive_event_prediction_board(
    predictions: pd.DataFrame,
    group_column: str,
    competitor_column: str,
    probability_column: str = "competitive_probability",
    actual_position_column: str | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Construye una tabla operativa por evento con ganador, exacta y podio predichos.

    Entradas:
        predictions: DataFrame con una fila por competidor y probabilidad competitiva por evento.
        group_column: Identificador de carrera o evento.
        competitor_column: Identificador o nombre del competidor que se mostrara en la salida.
        probability_column: Columna con la probabilidad competitiva del modelo.
        actual_position_column: Columna opcional con la posicion real de llegada para validar exacta/podio.
        top_k: Numero maximo de posiciones a resumir en el tablero.

    Salidas:
        DataFrame con una fila por evento y columnas listas para leer predicciones tipo ganador,
        exacta y podio, junto con aciertos cuando existe el orden real.

    Pruebas ejecutadas:
        No ejecuta un test estadistico; resume el ranking predicho y, si existe, el ranking real.
    """
    # Convierte el ranking por competidor en una vista de negocio por carrera, lista para decisiones pre-evento.
    _ensure_dataframe(predictions)
    required_columns = [group_column, competitor_column, probability_column]
    if actual_position_column is not None:
        required_columns.append(actual_position_column)
    _ensure_columns(predictions, required_columns)
    if top_k < 2:
        raise ValueError("top_k debe ser al menos 2 para resumir ganador y exacta.")

    working = predictions.copy()
    working[probability_column] = pd.to_numeric(working[probability_column], errors="coerce")
    working = working.dropna(subset=[group_column, competitor_column, probability_column]).copy()
    if working.empty:
        raise ValueError("No quedaron filas validas tras limpiar columnas requeridas del tablero competitivo.")

    if "predicted_rank" not in working.columns:
        working["predicted_rank"] = (
            working.groupby(group_column)[probability_column]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    if actual_position_column is not None:
        working[actual_position_column] = pd.to_numeric(working[actual_position_column], errors="coerce")

    board_rows: list[dict[str, Any]] = []
    for event_id, event in working.groupby(group_column):
        ordered = event.sort_values(["predicted_rank", probability_column], ascending=[True, False]).reset_index(drop=True)
        predicted_names = ordered[competitor_column].astype(str).tolist()
        predicted_probabilities = ordered[probability_column].astype(float).tolist()
        top_predicted = predicted_names[:top_k]
        top_probabilities = predicted_probabilities[:top_k]

        row: dict[str, Any] = {
            group_column: event_id,
            "field_size": int(len(ordered)),
            "predicted_winner": top_predicted[0] if top_predicted else None,
            "predicted_winner_probability": round(float(top_probabilities[0]), 6) if top_probabilities else np.nan,
            "predicted_exacta": " > ".join(top_predicted[:2]) if len(top_predicted) >= 2 else None,
            "predicted_podium": " > ".join(top_predicted[:3]) if len(top_predicted) >= 3 else None,
            "predicted_top_k": top_predicted,
            "predicted_top_k_probabilities": [round(float(value), 4) for value in top_probabilities],
        }

        if actual_position_column is not None:
            actual_order = (
                ordered.dropna(subset=[actual_position_column])
                .query(f"{actual_position_column} > 0")
                .sort_values(actual_position_column)
                .reset_index(drop=True)
            )
            actual_names = actual_order[competitor_column].astype(str).tolist()
            actual_top2 = actual_names[:2]
            actual_top3 = actual_names[:3]

            row.update(
                {
                    "actual_winner": actual_names[0] if actual_names else None,
                    "actual_exacta": " > ".join(actual_top2) if len(actual_top2) == 2 else None,
                    "actual_podium": " > ".join(actual_top3) if len(actual_top3) == 3 else None,
                    "winner_hit": float(bool(actual_names) and top_predicted[:1] == actual_names[:1]),
                    "exacta_ordered_hit": float(len(actual_top2) == 2 and top_predicted[:2] == actual_top2),
                    "exacta_box_hit": float(len(actual_top2) == 2 and set(top_predicted[:2]) == set(actual_top2)),
                    "podium_ordered_hit": float(len(actual_top3) == 3 and top_predicted[:3] == actual_top3),
                    "podium_box_hit": float(len(actual_top3) == 3 and set(top_predicted[:3]) == set(actual_top3)),
                }
            )

        board_rows.append(row)

    return pd.DataFrame(board_rows).sort_values(group_column).reset_index(drop=True)


def evaluate_competitive_event_tickets(
    predictions: pd.DataFrame,
    group_column: str,
    competitor_column: str,
    actual_position_column: str,
    probability_column: str = "competitive_probability",
    top_k: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua acierto de ganador, exacta y podio a partir del ranking competitivo.

    Entradas:
        predictions: DataFrame con ranking probabilistico por competidor.
        group_column: Identificador del evento o carrera.
        competitor_column: Identificador o nombre del competidor.
        actual_position_column: Posicion real de llegada para validar el top-k predicho.
        probability_column: Probabilidad competitiva generada por el modelo.
        top_k: Profundidad maxima a resumir en el tablero de predicciones.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con metricas agregadas y tablero por evento.

    Pruebas ejecutadas:
        Hit rate del ganador, exacta exacta, exacta box, podio exacto y podio box.
    """
    # Traducir el ranking a exacta y podio permite medir si el modelo sirve para decisiones top-k y combinadas.
    board = build_competitive_event_prediction_board(
        predictions=predictions,
        group_column=group_column,
        competitor_column=competitor_column,
        probability_column=probability_column,
        actual_position_column=actual_position_column,
        top_k=top_k,
    )

    required_metrics = [
        "winner_hit",
        "exacta_ordered_hit",
        "exacta_box_hit",
        "podium_ordered_hit",
        "podium_box_hit",
    ]
    missing_metrics = [column for column in required_metrics if column not in board.columns]
    if missing_metrics:
        raise ValueError(
            "No se pudieron calcular las metricas de exacta/podio. Revisa que actual_position_column tenga posiciones reales validas."
        )

    metrics = {
        "n_events": int(len(board)),
        "winner_hit_rate": round(float(board["winner_hit"].mean()), 4),
        "exacta_ordered_hit": round(float(board["exacta_ordered_hit"].mean()), 4),
        "exacta_box_hit": round(float(board["exacta_box_hit"].mean()), 4),
        "podium_ordered_hit": round(float(board["podium_ordered_hit"].mean()), 4),
        "podium_box_hit": round(float(board["podium_box_hit"].mean()), 4),
        "mean_top1_probability": round(float(board["predicted_winner_probability"].mean()), 4),
    }

    if metrics["exacta_box_hit"] >= 0.4 and metrics["podium_box_hit"] >= 0.65:
        interpretation = (
            "El ranking competitivo ya no solo identifica ganadores: tambien deja una senal util para construir exactas y podios con sentido operativo."
        )
    elif metrics["winner_hit_rate"] >= 0.35:
        interpretation = (
            "La senal para ganador es util, pero la combinatoria de exacta/podio aun requiere mas contexto historico o calibracion adicional."
        )
    else:
        interpretation = (
            "El tablero competitivo todavia no sostiene decisiones de exacta o podio con suficiente consistencia. Conviene reforzar features historicas y validacion temporal."
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "metrics": pd.DataFrame([metrics]),
        "event_board": board,
        "interpretation": interpretation,
    }


def train_competitive_event_model(
    df: pd.DataFrame,
    target: str,
    group_column: str,
    features: Sequence[str] | None = None,
    algorithm: Literal["logistic", "random_forest"] = "logistic",
    test_size: float = 0.25,
    random_state: int = 42,
    numeric_imputer: Literal["mean", "median"] = "median",
    categorical_imputer: Literal["most_frequent", "constant"] = "most_frequent",
    scaler: Literal["standard", "robust", "none"] = "standard",
    apply_power_transform: bool = False,
    power_method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
    probability_temperature: float = 1.0,
    min_group_size: int = 2,
    top_k: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Entrena un modelo competitivo por evento usando separacion agrupada y metricas de ranking.

    Entradas:
        df: DataFrame con una fila por competidor.
        target: Columna binaria que indica el ganador real.
        group_column: Identificador del evento o carrera.
        features: Predictores a usar. Si es None, usa todas salvo target y grupo.
        algorithm: Algoritmo base de clasificacion binaria.
        test_size: Proporcion de grupos para test.
        random_state: Semilla reproducible.
        numeric_imputer: Estrategia de imputacion numerica.
        categorical_imputer: Estrategia de imputacion categorica.
        scaler: Tipo de escalado para numericas.
        apply_power_transform: Si aplica transformacion de potencia a numericas.
        power_method: Metodo de potencia solicitado.
        probability_temperature: Temperatura del softmax por evento.
        min_group_size: Minimo de competidores por evento valido.
        top_k: Profundidad usada en ranking y hit rate.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con pipeline, metricas binarias y por evento, predicciones, importancias y reportes de grupo.

    Pruebas ejecutadas:
        GroupShuffleSplit por evento, ROC-AUC binario, log-loss binario, top-1 accuracy, MRR, NDCG@k,
        Brier y log-loss del ganador a nivel evento.
    """
    # Separa carreras completas entre train y test para evitar fuga entre competidores del mismo evento.
    _ensure_dataframe(df)
    _ensure_columns(df, [target, group_column])
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size debe estar entre 0 y 1 para separar grupos completos.")

    feature_list = list(features) if features is not None else [
        column for column in df.columns if column not in {target, group_column}
    ]
    feature_list = [column for column in feature_list if column not in {target, group_column}]
    if not feature_list:
        raise ValueError("No quedaron features validas tras excluir target y group_column del modelado competitivo.")
    _ensure_columns(df, feature_list)

    working = df[feature_list + [target, group_column]].dropna(subset=[target, group_column]).copy()
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working[working[target].isin([0, 1])].copy()
    working[target] = working[target].astype(int)

    group_quality = (
        working.groupby(group_column)[target]
        .agg(group_size="size", winners="sum")
        .reset_index()
    )
    valid_groups = group_quality[
        (group_quality["group_size"] >= min_group_size) & (group_quality["winners"] == 1)
    ][group_column]
    filtered = working[working[group_column].isin(valid_groups)].copy()
    if filtered.empty:
        raise ValueError(
            "No quedaron eventos validos tras filtrar por tamano minimo y ganador unico."
        )

    X = filtered[feature_list].copy()
    y = filtered[target].copy()
    groups = filtered[group_column].copy()
    pipeline_info = build_preprocessing_pipeline(
        X,
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        scaler=scaler,
        apply_power_transform=apply_power_transform,
        power_method=power_method,
        verbose=False,
    )

    model_map: dict[str, Any] = {
        "logistic": LogisticRegression(max_iter=5000, random_state=random_state, solver="liblinear"),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
        ),
    }
    if algorithm not in model_map:
        raise ValueError("algorithm no soportado. Usa 'logistic' o 'random_forest'.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(X, y, groups=groups))
    X_train = X.iloc[train_index].copy()
    X_test = X.iloc[test_index].copy()
    y_train = y.iloc[train_index].reset_index(drop=True)
    y_test = y.iloc[test_index].reset_index(drop=True)
    groups_test = groups.iloc[test_index].reset_index(drop=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", pipeline_info["preprocessor"]),
            ("model", model_map[algorithm]),
        ]
    )
    pipeline.fit(X_train, y_train)
    predicted_labels = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        predicted_probability = pipeline.predict_proba(X_test)[:, 1]
    else:
        predicted_probability = predicted_labels.astype(float)

    if hasattr(pipeline.named_steps["model"], "decision_function"):
        raw_scores = pipeline.decision_function(X_test)
    else:
        clipped = np.clip(predicted_probability, 1e-6, 1 - 1e-6)
        raw_scores = np.log(clipped / (1 - clipped))

    competitive_probability = normalize_competitive_event_probabilities(
        raw_scores,
        groups=groups_test,
        temperature=probability_temperature,
    )

    prediction_frame = pd.DataFrame(
        {
            group_column: groups_test.to_numpy(),
            "actual": y_test,
            "predicted": predicted_labels,
            "predicted_probability": predicted_probability,
            "raw_score": raw_scores,
            "competitive_probability": competitive_probability.to_numpy(),
        }
    )
    prediction_frame["field_size"] = prediction_frame.groupby(group_column)[group_column].transform("size")

    binary_metrics: dict[str, float] = {
        "accuracy": round(float(accuracy_score(y_test, predicted_labels)), 4),
        "precision": round(float(precision_score(y_test, predicted_labels, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted_labels, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predicted_labels, zero_division=0)), 4),
    }
    if y_test.nunique(dropna=True) == 2:
        binary_metrics["roc_auc"] = round(float(roc_auc_score(y_test, predicted_probability)), 4)
        binary_metrics["log_loss"] = round(float(log_loss(y_test, predicted_probability)), 4)

    scoring = "roc_auc" if y_test.nunique(dropna=True) == 2 else "accuracy"
    importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring=scoring,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    evaluation = evaluate_competitive_event_predictions(
        prediction_frame,
        group_column=group_column,
        target_column="actual",
        probability_column="competitive_probability",
        top_k=top_k,
        verbose=False,
    )
    event_metrics = evaluation["metrics"].copy()
    top1 = float(event_metrics.iloc[0]["top1_accuracy"])
    ndcg_column = next(column for column in event_metrics.columns if column.startswith("ndcg@"))
    ndcg_value = float(event_metrics.iloc[0][ndcg_column])
    interpretation = (
        f"Se entreno un modelo competitivo '{algorithm}' con separacion agrupada por {group_column}. "
        f"Top-1 accuracy = {top1:.3f}, {ndcg_column} = {ndcg_value:.3f}. "
        "Esta evaluacion evita contaminar train y test con competidores de la misma carrera y mide ranking util a nivel evento."
    )

    _emit_interpretation(interpretation, verbose)
    return {
        "pipeline": pipeline,
        "problem_type": "competitive_event_classification",
        "algorithm": algorithm,
        "group_column": group_column,
        "metrics": event_metrics,
        "binary_metrics": pd.DataFrame([binary_metrics]),
        "predictions": evaluation["predictions"],
        "event_report": evaluation["event_report"],
        "feature_importance": feature_importance,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test,
        "group_quality": group_quality,
        "valid_group_count": int(filtered[group_column].nunique()),
        "interpretation": interpretation,
        "preprocessing": pipeline_info,
    }


def plot_competitive_event_diagnostics(
    model_result: dict[str, Any],
    n_bins: int = 6,
) -> tuple[plt.Figure, np.ndarray]:
    """Genera diagnosticos visuales para un problema competitivo por evento.

    Entradas:
        model_result: Salida de train_competitive_event_model o de evaluate_competitive_event_predictions empaquetada en un dict.
        n_bins: Numero de bins para la curva de calibracion de los top picks.

    Salidas:
        Figura y arreglo de ejes.

    Pruebas ejecutadas:
        Curva de calibracion de la seleccion top-1, distribucion del rango real del ganador y
        distribucion de la probabilidad asignada al ganador verdadero.
    """
    # Convierte la validacion competitiva en tres chequeos visuales: calibracion, ranking y confianza sobre el ganador.
    if "predictions" not in model_result or "event_report" not in model_result:
        raise ValueError("model_result no tiene la estructura esperada de train_competitive_event_model.")

    predictions = model_result["predictions"].copy()
    event_report = model_result["event_report"].copy()
    aplicar_tema_profesional()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes = np.atleast_1d(axes)

    top_picks = predictions[predictions["predicted_rank"] == 1].copy()
    if len(top_picks) >= 3 and top_picks["actual"].nunique() > 1:
        observed, estimated = calibration_curve(top_picks["actual"], top_picks["competitive_probability"], n_bins=n_bins)
        axes[0].plot(estimated, observed, marker="o", color="#0F766E")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="#EA580C")
        axes[0].set_title("Calibracion de la seleccion top-1")
        axes[0].set_xlabel("Probabilidad predicha")
        axes[0].set_ylabel("Frecuencia observada")
    else:
        axes[0].text(0.5, 0.5, "No hay suficientes top picks para una curva de calibracion estable.", ha="center", va="center")
        axes[0].set_axis_off()

    sns.histplot(event_report["winner_rank"], bins=min(12, int(event_report["winner_rank"].max())), ax=axes[1], color="#2563EB")
    axes[1].set_title("Rango real del ganador")
    axes[1].set_xlabel("Posicion del ganador en el ranking predicho")
    axes[1].set_ylabel("Carreras")

    sns.histplot(event_report["winner_probability"], bins=12, kde=True, ax=axes[2], color="#EA580C")
    axes[2].set_title("Probabilidad asignada al ganador")
    axes[2].set_xlabel("Probabilidad competitiva")
    axes[2].set_ylabel("Carreras")

    fig.tight_layout()
    return fig, axes


def audit_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    id_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
    segment_columns: Sequence[str] | None = None,
    missing_threshold: float = 0.3,
    leakage_threshold: float = 0.98,
    verbose: bool = True,
) -> dict[str, Any]:
    """Audita calidad, completitud y riesgos basicos de sesgo del dataset.

    Entradas:
        df: DataFrame a revisar.
        target: Nombre opcional de la variable objetivo para buscar leakage potencial.
        id_columns: Columnas identificadoras que se deben revisar por unicidad.
        date_columns: Columnas fecha relevantes para trazabilidad y cobertura.
        segment_columns: Dimensiones de negocio para revisar representatividad.
        missing_threshold: Umbral de nulos para alertas de completitud.
        leakage_threshold: Correlacion absoluta o coincidencia exacta considerada sospechosa.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con reporte de calidad, nulos, columnas constantes, riesgo de leakage,
        control de identificadores, contrato tabular, representatividad e interpretacion.

    Pruebas ejecutadas:
        Reglas de auditoria de completitud, duplicados, unicidad, estandares tabulares,
        representatividad y correlaciones extremas contra la variable objetivo cuando esta disponible.
    """
    # Valida la estructura base antes de calcular indicadores de calidad y sesgo.
    _ensure_dataframe(df)
    if target is not None:
        _ensure_columns(df, [target])

    # Resume completitud, duplicados y señales basicas de estructura del dataset.
    quality_report = reporte_calidad_datos(df)
    missing_report = resumir_nulos(df)
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round(df.duplicated().mean() * 100, 2)
    constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]
    high_missing_columns = (
        missing_report[missing_report["pct_nulos"] >= missing_threshold * 100]
        .reset_index()
        .rename(columns={"index": "columna"})
    )

    # Revisa columnas identificadoras para detectar claves mal definidas o registros repetidos.
    id_columns = list(id_columns or [])
    date_columns = list(date_columns or [])
    segment_columns = list(segment_columns or [])
    id_report_rows: list[dict[str, Any]] = []
    for column in id_columns:
        _ensure_columns(df, [column])
        uniqueness_ratio = df[column].nunique(dropna=False) / len(df)
        id_report_rows.append(
            {
                "columna": column,
                "pct_unicidad": round(uniqueness_ratio * 100, 2),
                "duplicados": int(df[column].duplicated().sum()),
            }
        )
    id_report = pd.DataFrame(id_report_rows)

    # Busca relaciones sospechosamente cercanas al objetivo como primera alarma de leakage.
    leakage_candidates = pd.DataFrame(columns=["columna", "tipo", "fuerza"])
    if target is not None:
        candidates: list[dict[str, Any]] = []
        y = df[target]

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if target in numeric_columns:
            correlations = (
                df[numeric_columns]
                .corr(numeric_only=True)[target]
                .drop(labels=[target])
                .dropna()
                .abs()
                .sort_values(ascending=False)
            )
            for column, strength in correlations.items():
                if strength >= leakage_threshold:
                    candidates.append(
                        {
                            "columna": column,
                            "tipo": "correlacion_extrema",
                            "fuerza": round(float(strength), 4),
                        }
                    )

        target_as_text = y.astype(str).fillna("<NA>")
        for column in df.columns:
            if column == target:
                continue
            exact_match = (df[column].astype(str).fillna("<NA>") == target_as_text).mean()
            if exact_match >= leakage_threshold:
                candidates.append(
                    {
                        "columna": column,
                        "tipo": "coincidencia_objetivo",
                        "fuerza": round(float(exact_match), 4),
                    }
                )

        if candidates:
            leakage_candidates = pd.DataFrame(candidates).drop_duplicates().sort_values(
                "fuerza", ascending=False
            )

    tabular_standards = audit_tabular_data_standards(
        df,
        id_columns=id_columns,
        date_columns=date_columns,
        verbose=False,
    )
    sampling_audit = audit_sampling_representativeness(
        df,
        segment_columns=segment_columns,
        target=target,
        date_column=date_columns[0] if date_columns else None,
        verbose=False,
    )

    executive_summary = pd.DataFrame(
        [
            {
                "filas": len(df),
                "columnas": df.shape[1],
                "pct_duplicados_fila": duplicate_pct,
                "columnas_constantes": len(constant_columns),
                "columnas_con_nulos_altos": len(high_missing_columns),
                "candidatos_leakage": len(leakage_candidates),
                "alertas_tabulares": int(tabular_standards["summary"].iloc[0].sum()),
                "alertas_representatividad": int(len(sampling_audit["alerts"])),
            }
        ]
    )

    # Compone una conclusion legible que sintetiza los riesgos principales de auditoria.
    messages: list[str] = []
    if duplicate_count > 0:
        messages.append(
            f"Se detectaron {duplicate_count} filas duplicadas ({duplicate_pct}%). "
            "Revisa si provienen de joins incorrectos o de una unidad de analisis mal definida."
        )
    else:
        messages.append("No se detectaron filas completamente duplicadas en esta vista del dataset.")

    if not high_missing_columns.empty:
        messages.append(
            "Hay columnas con nulos relevantes. Antes de imputar, valida si el mecanismo parece MCAR, MAR o MNAR."
        )
    else:
        messages.append("No hay columnas que superen el umbral de nulos criticos definido para la auditoria.")

    if not leakage_candidates.empty:
        messages.append(
            "Se encontraron variables con senales sospechosamente cercanas al objetivo. "
            "Valida origen temporal y definicion de negocio antes de modelar."
        )
    else:
        messages.append("No aparecieron candidatos de leakage evidentes con las reglas automaticas aplicadas.")

    if int(tabular_standards["summary"].iloc[0].sum()) > 0:
        messages.append(tabular_standards["interpretation"])
    if not sampling_audit["alerts"].empty:
        messages.append(sampling_audit["interpretation"])

    interpretation = " ".join(messages)
    _emit_interpretation(interpretation, verbose)

    return {
        "summary": executive_summary,
        "quality_report": quality_report,
        "missing_report": missing_report,
        "high_missing_columns": high_missing_columns,
        "constant_columns": constant_columns,
        "id_report": id_report,
        "leakage_candidates": leakage_candidates,
        "tabular_standards": tabular_standards,
        "sampling_audit": sampling_audit,
        "interpretation": interpretation,
    }


def plot_missingness_heatmap(
    df: pd.DataFrame,
    max_columns: int = 25,
) -> tuple[plt.Figure, plt.Axes]:
    """Genera un heatmap de faltantes para validar visualmente la auditoria.

    Entradas:
        df: DataFrame analizado.
        max_columns: Numero maximo de columnas a mostrar, priorizando las de mayor missingness.

    Salidas:
        Figura y eje de Matplotlib.

    Pruebas ejecutadas:
        No ejecuta una prueba estadistica; visualiza patron de ausencias por observacion.
    """
    # Prioriza visualmente las columnas donde el patron de ausencia puede ser informativo.
    _ensure_dataframe(df)
    aplicar_tema_profesional()

    missing_order = df.isna().mean().sort_values(ascending=False)
    selected_columns = missing_order.head(max_columns).index.tolist()
    matrix = df[selected_columns].isna().astype(int)

    fig, ax = plt.subplots(figsize=(12, max(4, len(selected_columns) * 0.35)))
    if matrix.sum().sum() == 0:
        ax.text(0.5, 0.5, "No se detectaron faltantes en las columnas seleccionadas.", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    sns.heatmap(matrix.T, cmap="rocket_r", cbar=False, ax=ax)
    ax.set_title("Patron de valores faltantes")
    ax.set_xlabel("Observaciones")
    ax.set_ylabel("Columnas")
    fig.tight_layout()
    return fig, ax


def handle_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: Literal["clip_iqr", "remove_iqr", "winsorize"] = "clip_iqr",
    iqr_factor: float = 1.5,
    winsorize_limits: tuple[float, float] = (0.01, 0.01),
    verbose: bool = True,
) -> dict[str, Any]:
    """Trata outliers numericos con reglas reproducibles basadas en cuantiles e IQR.

    Entradas:
        df: DataFrame original.
        columns: Columnas numericas a tratar. Si es None, usa todas las numericas.
        method: Estrategia: recorte por IQR, eliminacion o winsorizacion por percentiles.
        iqr_factor: Factor multiplicador del IQR para definir limites.
        winsorize_limits: Percentiles inferiores y superiores para winsorizacion.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con dataframe transformado, reporte de outliers e interpretacion.

    Pruebas ejecutadas:
        Regla de outliers por IQR y cuantiles de winsorizacion.
    """
    # Aplica una regla homogénea por columna para no tratar extremos de forma ad hoc.
    _ensure_dataframe(df)
    result = df.copy()
    target_columns = list(columns) if columns is not None else list(df.select_dtypes(include=np.number).columns)
    _ensure_columns(df, target_columns)

    reports: list[dict[str, Any]] = []
    rows_before = len(result)

    # Calcula limites robustos y ejecuta la estrategia elegida sobre cada variable numerica.
    for column in target_columns:
        series = _numeric_series(result[column], name=column)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        mask = result[column].between(lower, upper) | result[column].isna()

        if method == "clip_iqr":
            result[column] = pd.to_numeric(result[column], errors="coerce").clip(lower, upper)
        elif method == "remove_iqr":
            result = result[mask].copy()
        elif method == "winsorize":
            lower_q = series.quantile(winsorize_limits[0])
            upper_q = series.quantile(1 - winsorize_limits[1])
            result[column] = pd.to_numeric(result[column], errors="coerce").clip(lower_q, upper_q)
            lower, upper = lower_q, upper_q
        else:
            raise ValueError("Metodo no soportado. Usa 'clip_iqr', 'remove_iqr' o 'winsorize'.")

        reports.append(
            {
                "columna": column,
                "limite_inferior": round(float(lower), 4),
                "limite_superior": round(float(upper), 4),
                "pct_outliers_original": round(float((~mask).mean() * 100), 2),
                "metodo": method,
            }
        )

    report = pd.DataFrame(reports).sort_values("pct_outliers_original", ascending=False)
    rows_after = len(result)

    # Traduce el tratamiento aplicado a una recomendacion operativa legible.
    if method == "remove_iqr":
        interpretation = (
            f"Se eliminaron {rows_before - rows_after} filas al aplicar la regla IQR. "
            "Usa esta estrategia solo si los extremos son errores o si su influencia distorsiona el analisis."
        )
    else:
        interpretation = (
            f"Se aplico el metodo '{method}' sobre {len(target_columns)} columnas numericas. "
            "La muestra se conserva, pero la magnitud de los extremos queda controlada para EDA y modelado."
        )

    _emit_interpretation(interpretation, verbose)
    return {"data": result, "report": report, "interpretation": interpretation}


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    numeric_imputer: Literal["mean", "median", "knn", "iterative"] = "median",
    categorical_imputer: Literal["most_frequent", "constant"] = "most_frequent",
    scaler: Literal["standard", "robust", "none"] = "standard",
    apply_power_transform: bool = False,
    power_method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
    verbose: bool = True,
) -> dict[str, Any]:
    """Construye un preprocesador reproducible con ColumnTransformer de scikit-learn.

    Entradas:
        X: Matriz de predictores en formato DataFrame.
        numeric_imputer: Estrategia de imputacion numerica.
        categorical_imputer: Estrategia de imputacion categorica.
        scaler: Tipo de escalado numerico.
        apply_power_transform: Si aplica transformaciones de potencia a numericas.
        power_method: Metodo de potencia, Box-Cox o Yeo-Johnson.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con el ColumnTransformer, listas de columnas e interpretacion.

    Pruebas ejecutadas:
        No ejecuta un test de hipotesis; construye un pipeline reproducible con opcion
        de PowerTransformer para Box-Cox o Yeo-Johnson.
    """
    # Separa columnas numericas y categoricas para construir un pipeline reproducible.
    _ensure_dataframe(X)
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    chosen_power_method = power_method
    # Box-Cox exige positividad estricta; si no se cumple, se degrada a Yeo-Johnson.
    if apply_power_transform and power_method == "box-cox":
        numeric_data = X[numeric_columns]
        if numeric_data.empty or (numeric_data.min(numeric_only=True) <= 0).any():
            chosen_power_method = "yeo-johnson"

    numeric_steps: list[tuple[str, Any]] = [("imputer", _resolve_numeric_imputer(numeric_imputer))]
    if apply_power_transform and numeric_columns:
        numeric_steps.append(
            ("power", PowerTransformer(method=chosen_power_method, standardize=False))
        )
    numeric_steps.append(("scaler", _resolve_scaler(scaler)))

    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=categorical_imputer, fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    # Encapsula imputacion, potencia, escalado y codificacion en un solo objeto reusable.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", Pipeline(steps=categorical_steps), categorical_columns),
        ],
        remainder="drop",
    )
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

    if apply_power_transform:
        if chosen_power_method != power_method:
            interpretation = (
                "Se solicito Box-Cox, pero se detectaron ceros o negativos. "
                "El pipeline usa Yeo-Johnson para mantener validez estadistica."
            )
        else:
            interpretation = (
                f"El pipeline aplica {chosen_power_method} a variables numericas antes del escalado."
            )
    else:
        interpretation = "El pipeline usa imputacion y escalado sin transformacion de potencia."

    _emit_interpretation(interpretation, verbose)
    return {
        "preprocessor": preprocessor,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "power_method_used": chosen_power_method if apply_power_transform else None,
        "interpretation": interpretation,
    }


def compare_power_transformations(
    series: pd.Series,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compara transformaciones de potencia y recomienda la mas estable.

    Entradas:
        series: Serie numerica a transformar.
        verbose: Si es True, imprime la recomendacion principal.

    Salidas:
        Diccionario con datos transformados, resumen de asimetria y recomendacion.

    Pruebas ejecutadas:
        Aplica Yeo-Johnson siempre y Box-Cox solo cuando la variable es estrictamente positiva.
    """
    # Compara la variable original frente a transformaciones candidatas en la misma escala analitica.
    numeric = _numeric_series(series)
    transformed = pd.DataFrame({"original": numeric.reset_index(drop=True)})

    yj = PowerTransformer(method="yeo-johnson", standardize=False)
    transformed["yeo_johnson"] = yj.fit_transform(numeric.to_frame()).ravel()

    if (numeric > 0).all():
        box_cox = PowerTransformer(method="box-cox", standardize=False)
        transformed["box_cox"] = box_cox.fit_transform(numeric.to_frame()).ravel()

    # Ordena las opciones por reduccion de asimetria y curtosis para recomendar la mas estable.
    summary_rows: list[dict[str, Any]] = []
    for column in transformed.columns:
        summary_rows.append(
            {
                "transformacion": column,
                "asimetria_abs": round(abs(float(stats.skew(transformed[column], bias=False))), 4),
                "curtosis_abs": round(abs(float(stats.kurtosis(transformed[column], fisher=True, bias=False))), 4),
                "desviacion": round(float(transformed[column].std(ddof=1)), 4),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["asimetria_abs", "curtosis_abs"])
    recommended = str(summary.iloc[0]["transformacion"])
    interpretation = (
        f"La transformacion recomendada es '{recommended}' porque reduce mas la asimetria absoluta. "
        "Confirma la decision con histogramas y Q-Q plots antes de asumir normalidad aproximada."
    )

    _emit_interpretation(interpretation, verbose)
    return {
        "data": transformed,
        "summary": summary,
        "recommended": recommended,
        "interpretation": interpretation,
    }


def plot_power_transformations(
    series: pd.Series,
) -> tuple[plt.Figure, np.ndarray]:
    """Dibuja distribuciones antes y despues de Box-Cox o Yeo-Johnson.

    Entradas:
        series: Serie numerica original.

    Salidas:
        Figura y arreglo de ejes con histogramas comparables.

    Pruebas ejecutadas:
        Visualizacion descriptiva de las transformaciones de potencia calculadas.
    """
    # Reutiliza el comparador para que el grafico y la recomendacion partan del mismo calculo.
    comparison = compare_power_transformations(series, verbose=False)
    aplicar_tema_profesional()

    data = comparison["data"]
    fig, axes = plt.subplots(1, len(data.columns), figsize=(5 * len(data.columns), 4))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, data.columns):
        sns.histplot(data[column], kde=True, ax=ax, color="#2563EB")
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel(column)
    fig.tight_layout()
    return fig, axes


def check_normality(
    series: pd.Series,
    alpha: float = 0.05,
    max_shapiro_size: int = 5000,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua normalidad estructural con varias pruebas y una lectura consolidada.

    Entradas:
        series: Serie numerica.
        alpha: Nivel de significacion.
        max_shapiro_size: Tamano maximo a evaluar directamente con Shapiro-Wilk.
        verbose: Si es True, imprime interpretacion automatizada.

    Salidas:
        Diccionario con prueba primaria, auditoria multi-test, resumen de forma e interpretacion.

    Pruebas ejecutadas:
        Shapiro-Wilk, Anderson-Darling, Lilliefors y Jarque-Bera, mas resumen de asimetria/curtosis.
    """
    # Combina pruebas complementarias porque un solo p-valor puede ser enganoso en pipelines reales.
    numeric = _numeric_series(series)
    if len(numeric) < 3:
        raise ValueError("Shapiro-Wilk requiere al menos 3 observaciones validas.")

    evaluated = numeric.sample(max_shapiro_size, random_state=42) if len(numeric) > max_shapiro_size else numeric
    shapiro_statistic, shapiro_p_value = stats.shapiro(evaluated)

    try:
        anderson_result = stats.anderson(numeric, dist="norm", method="interpolate")
        ad_p_value = float(anderson_result.pvalue)
        ad_critical_5 = float("nan")
        ad_significance_bound = float("nan")
        ad_rejects = bool(ad_p_value < alpha)
    except TypeError:
        anderson_result = stats.anderson(numeric, dist="norm")
        ad_significance_levels = np.asarray(anderson_result.significance_level, dtype=float) / 100.0
        ad_critical_values = np.asarray(anderson_result.critical_values, dtype=float)
        ad_critical_5 = float(
            np.interp(alpha, ad_significance_levels[::-1], ad_critical_values[::-1])
            if alpha >= ad_significance_levels.min() and alpha <= ad_significance_levels.max()
            else ad_critical_values[np.argmin(np.abs(ad_significance_levels - alpha))]
        )
        ad_rejects = bool(anderson_result.statistic > ad_critical_5)
        exceeded_levels = ad_significance_levels[anderson_result.statistic > ad_critical_values]
        ad_significance_bound = float(exceeded_levels.min()) if exceeded_levels.size else float(ad_significance_levels.max())
        ad_p_value = float("nan")

    lilliefors_statistic = np.nan
    lilliefors_p_value = np.nan
    if len(numeric) >= 5:
        lilliefors_statistic, lilliefors_p_value = lilliefors(numeric.to_numpy(dtype=float), dist="norm")

    jarque_result = stats.jarque_bera(numeric.to_numpy(dtype=float))
    skewness = float(stats.skew(numeric, bias=False)) if len(numeric) >= 3 else float("nan")
    kurtosis = float(stats.kurtosis(numeric, fisher=True, bias=False)) if len(numeric) >= 4 else float("nan")

    tests_table = pd.DataFrame(
        [
            {
                "prueba": "Shapiro-Wilk",
                "estadistico": float(shapiro_statistic),
                "p_value": float(shapiro_p_value),
                "alpha": alpha,
                "rechaza_normalidad": bool(shapiro_p_value < alpha),
                "foco": "Desviaciones generales de forma en muestras pequenas o moderadas",
                "muestra_evaluada": int(len(evaluated)),
            },
            {
                "prueba": "Anderson-Darling",
                "estadistico": float(anderson_result.statistic),
                "p_value": ad_p_value,
                "alpha": alpha,
                "rechaza_normalidad": ad_rejects,
                "foco": "Desviaciones en colas y riesgo extremo",
                "muestra_evaluada": int(len(numeric)),
            },
            {
                "prueba": "Lilliefors",
                "estadistico": float(lilliefors_statistic) if not np.isnan(lilliefors_statistic) else np.nan,
                "p_value": float(lilliefors_p_value) if not np.isnan(lilliefors_p_value) else np.nan,
                "alpha": alpha,
                "rechaza_normalidad": bool(lilliefors_p_value < alpha) if not np.isnan(lilliefors_p_value) else np.nan,
                "foco": "Grandes muestras con parametros poblacionales estimados",
                "muestra_evaluada": int(len(numeric)),
            },
            {
                "prueba": "Jarque-Bera",
                "estadistico": float(jarque_result.statistic),
                "p_value": float(jarque_result.pvalue),
                "alpha": alpha,
                "rechaza_normalidad": bool(jarque_result.pvalue < alpha),
                "foco": "Asimetria estructural y exceso de curtosis",
                "muestra_evaluada": int(len(numeric)),
            },
        ]
    )

    primary_method = "Shapiro-Wilk" if len(numeric) <= max_shapiro_size else "Lilliefors"
    primary_row = tests_table.loc[tests_table["prueba"] == primary_method].iloc[0]
    reject_votes = tests_table["rechaza_normalidad"].dropna().astype(bool)
    is_normal = bool((~reject_votes).sum() >= reject_votes.sum())
    consensus = "compatible con normalidad" if is_normal else "evidencia de no normalidad"

    transform_hint = "Box-Cox o log" if float(numeric.min()) > 0 else "Yeo-Johnson"
    if is_normal:
        interpretation = (
            f"La auditoria estructural sugiere una distribucion {consensus}. "
            f"La prueba primaria fue {primary_method} y no ofrece evidencia suficiente para rechazar normalidad al nivel {alpha:.2f}. "
            "Aun asi, el protocolo exige confirmar con Q-Q plot e histograma antes de defender supuestos parametricos."
        )
    else:
        interpretation = (
            f"La auditoria estructural sugiere {consensus}. La prueba primaria fue {primary_method} y al menos dos contrastes "
            "apuntan a asimetria, curtosis anomala o desviaciones de cola. "
            f"Antes de seguir con metodos parametricos conviene evaluar {transform_hint}, revisar el Q-Q plot y considerar metodos robustos o no parametricos."
        )

    if len(numeric) > max_shapiro_size:
        interpretation += (
            " En muestras grandes Shapiro-Wilk se reporta sobre una submuestra controlada para evitar hipersensibilidad; "
            "por eso el peso interpretativo recae mas en Lilliefors, Anderson-Darling y la inspeccion visual."
        )

    if abs(skewness) >= 1 or abs(kurtosis) >= 1:
        interpretation += (
            " La forma observada refuerza prudencia: la variable muestra sesgo o curtosis material, algo frecuente en variables financieras y monetarias."
        )

    recommended_action = (
        "Mantener ruta parametrica con auditoria visual documentada."
        if is_normal
        else f"Aplicar {transform_hint}, revisar colas con Anderson-Darling y, si la desviacion persiste, migrar a tecnicas robustas o no parametricas."
    )

    shape_summary = pd.DataFrame(
        [
            {
                "n_observaciones": int(len(numeric)),
                "asimetria": round(skewness, 4),
                "curtosis_exceso": round(kurtosis, 4),
                "prueba_primaria": primary_method,
                "consenso": consensus,
                "accion_recomendada": recommended_action,
                "anderson_critical_value_alpha": round(ad_critical_5, 4),
                "anderson_significance_bound": round(ad_significance_bound, 4),
            }
        ]
    )

    _emit_interpretation(interpretation, verbose)
    return {
        "method": primary_method,
        "statistic": float(primary_row["estadistico"]),
        "p_value": float(primary_row["p_value"]) if pd.notna(primary_row["p_value"]) else float("nan"),
        "alpha": alpha,
        "evaluated_n": int(len(evaluated)),
        "is_normal": is_normal,
        "tests_table": tests_table,
        "shape_summary": shape_summary,
        "recommended_action": recommended_action,
        "shapiro_sampled": bool(len(numeric) > max_shapiro_size),
        "interpretation": interpretation,
    }


def plot_qq_diagnostic(
    series: pd.Series,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Genera un Q-Q plot para validar visualmente la normalidad aproximada.

    Entradas:
        series: Serie numerica a inspeccionar.
        title: Titulo opcional de la figura.

    Salidas:
        Figura y eje de Matplotlib.

    Pruebas ejecutadas:
        No ejecuta un test; visualiza cuantiles observados frente a cuantiles teoricos normales.
    """
    # Estandariza la serie para que la lectura dependa de la forma y no de la magnitud monetaria.
    numeric = _numeric_series(series)
    mean_value = float(numeric.mean())
    std_value = float(numeric.std(ddof=0))

    if np.isfinite(std_value) and std_value > 0:
        plotted = (numeric - mean_value) / std_value
        y_label = "Cuantiles muestrales estandarizados"
        scale_note = "Serie escalada en z-scores para comparar forma contra N(0,1)."
    else:
        plotted = numeric - mean_value
        y_label = "Cuantiles muestrales centrados"
        scale_note = "Serie centrada; no fue posible estandarizar por varianza nula."

    (theoretical_q, sample_q), (slope, intercept, corr) = stats.probplot(plotted, dist="norm", fit=True)
    line_x = np.linspace(float(np.min(theoretical_q)), float(np.max(theoretical_q)), 200)
    line_y = slope * line_x + intercept
    axis_min = float(min(np.min(theoretical_q), np.min(sample_q), np.min(line_y)))
    axis_max = float(max(np.max(theoretical_q), np.max(sample_q), np.max(line_y)))
    axis_padding = max((axis_max - axis_min) * 0.08, 0.15)

    aplicar_tema_profesional()
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(theoretical_q, sample_q, alpha=0.65, s=28, color="#0f766e", edgecolor="none")
    ax.plot(line_x, line_y, color="#c62828", linewidth=2)
    ax.set_xlim(axis_min - axis_padding, axis_max + axis_padding)
    ax.set_ylim(axis_min - axis_padding, axis_max + axis_padding)
    ax.set_xlabel("Cuantiles teoricos normales")
    ax.set_ylabel(y_label)
    ax.set_title(title or f"Q-Q plot: {series.name or 'variable'}")
    ax.text(
        0.03,
        0.97,
        f"n = {len(plotted)} | r = {corr:.3f}\n{scale_note}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    fig.tight_layout()
    return fig, ax


def check_variance_homogeneity(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    alpha: float = 0.05,
    center: Literal["mean", "median", "trimmed"] = "median",
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua homocedasticidad entre grupos con Levene o Brown-Forsythe.

    Entradas:
        df: DataFrame fuente.
        value_column: Variable numerica a comparar.
        group_column: Variable de agrupacion.
        alpha: Nivel de significacion.
        center: Centro usado por Levene. 'median' equivale a Brown-Forsythe.
        verbose: Si es True, imprime interpretacion automatizada.

    Salidas:
        Diccionario con estadistico, p-valor, grupos y conclusion.

    Pruebas ejecutadas:
        Test de Levene; con center='median' se interpreta como variante Brown-Forsythe.
    """
    # Prepara los grupos como listas numericas independientes para el contraste de dispersion.
    _ensure_dataframe(df)
    _ensure_columns(df, [value_column, group_column])

    working = df[[value_column, group_column]].dropna().copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    grouped = [group[value_column] for _, group in working.groupby(group_column)]

    if len(grouped) < 2:
        raise ValueError("Se requieren al menos dos grupos con datos numericos validos.")

    # center='median' activa la lectura tipo Brown-Forsythe para mayor robustez a asimetria.
    statistic, p_value = stats.levene(*grouped, center=center)
    equal_variance = bool(p_value >= alpha)
    test_name = "Brown-Forsythe" if center == "median" else "Levene"

    if equal_variance:
        interpretation = (
            f"P-valor = {p_value:.4f} >= {alpha}. No se rechaza igualdad de varianzas. "
            "La comparacion parametrica clasica es razonable si la normalidad tambien aguanta."
        )
    else:
        interpretation = (
            f"P-valor = {p_value:.4f} < {alpha}. Se rechaza homogeneidad de varianzas. "
            "Conviene usar Welch, errores robustos o rutas no parametricas segun el caso."
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "method": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "equal_variance": equal_variance,
        "groups": int(len(grouped)),
        "interpretation": interpretation,
    }


def _build_structural_dispersion_audit(
    model: Any,
    design: pd.DataFrame,
    y: pd.Series,
    influence: Any,
    group_series: pd.Series | None = None,
    group_column: str | None = None,
    alpha: float = 0.05,
    robust_cov: Literal["HC0", "HC1", "HC2", "HC3"] = "HC3",
) -> dict[str, Any]:
    # Combina contrastes auxiliares y patron de residuos para evitar una lectura binaria ingenua.
    residuals = pd.Series(model.resid, index=y.index, name="residual")
    fitted = pd.Series(model.fittedvalues, index=y.index, name="fitted")
    standardized = pd.Series(
        influence.resid_studentized_internal,
        index=y.index,
        name="standardized_residual",
    )
    abs_residuals = residuals.abs()
    sqrt_abs_standardized = np.sqrt(np.abs(standardized))

    tests_rows: list[dict[str, Any]] = []

    # Breusch-Pagan en su variante tipo Koenker-Bassett reduce dependencia de normalidad exacta.
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, design, robust=True)
    tests_rows.append(
        {
            "prueba": "Breusch-Pagan / Koenker",
            "estadistico": float(bp_lm),
            "p_value": float(bp_lm_p),
            "alpha": alpha,
            "rechaza_homocedasticidad": bool(bp_lm_p < alpha),
            "foco": "Varianza del error linealmente asociada a los predictores",
            "detalle": f"F auxiliar = {bp_f:.4f}; p-valor F = {bp_f_p:.4f}",
        }
    )

    # White amplia la regresion auxiliar con cuadrados e interacciones para formas no lineales.
    white_lm, white_lm_p, white_f, white_f_p = het_white(residuals, design)
    tests_rows.append(
        {
            "prueba": "White",
            "estadistico": float(white_lm),
            "p_value": float(white_lm_p),
            "alpha": alpha,
            "rechaza_homocedasticidad": bool(white_lm_p < alpha),
            "foco": "Heterocedasticidad general y no lineal",
            "detalle": f"F auxiliar = {white_f:.4f}; p-valor F = {white_f_p:.4f}",
        }
    )

    # Goldfeld-Quandt ordena por el predictor mas alineado con la magnitud residual para buscar patron embudo.
    sort_feature = None
    gq_stat = float("nan")
    gq_p = float("nan")
    if design.shape[1] > 1 and len(y) > max(40, design.shape[1] * 6):
        sort_candidates = design.drop(columns="const", errors="ignore")
        strength_by_feature = (
            sort_candidates.apply(lambda column: abs(column.corr(abs_residuals, method="spearman")))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        sort_feature = str(strength_by_feature.sort_values(ascending=False).index[0])
        sort_position = int(sort_candidates.columns.get_loc(sort_feature))
        max_drop = max(len(y) - (2 * (sort_candidates.shape[1] + 5)), 0)
        drop_n = min(max(int(len(y) * 0.2), 1), max_drop) if max_drop > 0 else 0
        try:
            gq_stat, gq_p, _ = het_goldfeldquandt(
                y,
                sort_candidates,
                idx=sort_position,
                drop=drop_n if drop_n > 0 else None,
                alternative="two-sided",
            )
        except ValueError:
            gq_stat, gq_p = float("nan"), float("nan")

    tests_rows.append(
        {
            "prueba": "Goldfeld-Quandt",
            "estadistico": float(gq_stat),
            "p_value": float(gq_p),
            "alpha": alpha,
            "rechaza_homocedasticidad": bool(pd.notna(gq_p) and gq_p < alpha),
            "foco": "Cambios monotonos de varianza a lo largo del predictor ordenado",
            "detalle": (
                f"Variable de orden = {sort_feature}"
                if sort_feature is not None
                else "Sin variable de orden fiable por tamano o estructura de muestra"
            ),
        }
    )

    group_dispersion = None
    if group_series is not None:
        aligned_group = pd.Series(group_series, index=y.index, name=group_column or "grupo")
        grouped = pd.DataFrame(
            {
                "abs_residual": abs_residuals,
                "grupo": aligned_group,
            }
        ).dropna()
        if grouped["grupo"].nunique() >= 2:
            group_dispersion = check_variance_homogeneity(
                grouped,
                value_column="abs_residual",
                group_column="grupo",
                alpha=alpha,
                center="median",
                verbose=False,
            )
            tests_rows.append(
                {
                    "prueba": group_dispersion["method"],
                    "estadistico": float(group_dispersion["statistic"]),
                    "p_value": float(group_dispersion["p_value"]),
                    "alpha": alpha,
                    "rechaza_homocedasticidad": bool(not group_dispersion["equal_variance"]),
                    "foco": f"Dispersion residual entre grupos de {group_column or 'segmentacion'}",
                    "detalle": "Usa residuos absolutos para mantener robustez cuando la forma no es gaussiana.",
                }
            )

    tests_table = pd.DataFrame(tests_rows)

    # La magnitud importa: se estima si el patron residual cambia materialmente entre zonas bajas y altas del ajuste.
    residual_frame = pd.DataFrame(
        {
            "observation": y.index,
            "fitted": fitted,
            "residual": residuals,
            "abs_residual": abs_residuals,
            "standardized_residual": standardized,
            "sqrt_abs_standardized_residual": sqrt_abs_standardized,
        }
    )
    if group_series is not None:
        residual_frame[group_column or "grupo"] = pd.Series(group_series, index=y.index)

    lower_band = residual_frame[residual_frame["fitted"] <= residual_frame["fitted"].quantile(0.25)]["abs_residual"]
    upper_band = residual_frame[residual_frame["fitted"] >= residual_frame["fitted"].quantile(0.75)]["abs_residual"]
    spread_ratio = float("nan")
    if not lower_band.empty and not upper_band.empty:
        spread_ratio = float(upper_band.median() / max(lower_band.median(), 1e-8))

    abs_resid_corr = stats.spearmanr(
        residual_frame["fitted"],
        residual_frame["abs_residual"],
        nan_policy="omit",
    )
    abs_resid_fitted_spearman = float(abs_resid_corr.statistic)

    rejected_tests = int(tests_table["rechaza_homocedasticidad"].sum())
    visual_alert = bool(
        (pd.notna(spread_ratio) and spread_ratio >= 1.25)
        or (pd.notna(abs_resid_fitted_spearman) and abs(abs_resid_fitted_spearman) >= 0.2)
    )

    if rejected_tests == 0 and not visual_alert:
        consensus = "sin evidencia relevante de heterocedasticidad"
        practical_relevance = "baja"
        recommended_action = (
            f"Mantener errores robustos {robust_cov} como practica preventiva y continuar con monitoreo visual rutinario."
        )
        interpretation = (
            "Los contrastes de dispersión y el patrón de residuos no muestran una desviación material de homocedasticidad. "
            "La inferencia robusta sigue siendo recomendable como estándar industrial, pero no hay una alerta que obligue a reespecificar el modelo."
        )
    elif rejected_tests >= 2 and visual_alert:
        consensus = "evidencia consistente de heterocedasticidad"
        practical_relevance = "alta" if pd.notna(spread_ratio) and spread_ratio >= 1.5 else "media"
        recommended_action = (
            f"Conservar {robust_cov} o HC3/Huber-White para toda la inferencia, evitar t y F clasicos, "
            "y considerar reespecificacion, transformaciones o WLS solo si mejoran estabilidad sin destruir la escala de negocio."
        )
        interpretation = (
            "Las pruebas auxiliares y el análisis de residuos apuntan a heterocedasticidad con relevancia operativa. "
            "Los coeficientes OLS siguen siendo utilizables bajo exogeneidad, pero los errores estándar clásicos dejan de ser defendibles y la lectura paramétrica debe descansar en covarianzas robustas."
        )
    elif rejected_tests >= 1 and not visual_alert:
        consensus = "senal estadistica leve con relevancia practica baja"
        practical_relevance = "baja"
        recommended_action = (
            f"Documentar la desviacion, mantener {robust_cov} y evitar transformar variables solo por un p-valor marginal si el patron residual no altera decisiones."
        )
        interpretation = (
            "Aparece alguna señal estadística de heterocedasticidad, pero la magnitud del cambio en dispersión es pequeña. "
            "La respuesta correcta es robustecer la inferencia y dejar constancia metodológica, no forzar transformaciones que compliquen la lectura de negocio sin una ganancia real."
        )
    else:
        consensus = "seguimiento por dispersion potencial"
        practical_relevance = "media"
        recommended_action = (
            f"Mantener {robust_cov}, revisar el grafico de residuos en cada refresco del modelo y priorizar rutas robustas o no lineales si el patron tipo embudo se vuelve persistente."
        )
        interpretation = (
            "La evidencia no es totalmente concluyente, pero el patrón residual merece seguimiento. "
            "Antes de asumir homocedasticidad por comodidad conviene observar si la dispersión crece con el ajuste o se concentra en segmentos específicos."
        )

    shape_summary = pd.DataFrame(
        [
            {
                "n_observaciones": int(len(residual_frame)),
                "error_robusto": robust_cov,
                "pruebas_con_alerta": rejected_tests,
                "ratio_dispersion_q4_q1": round(spread_ratio, 4) if pd.notna(spread_ratio) else np.nan,
                "spearman_abs_resid_ajuste": round(abs_resid_fitted_spearman, 4),
                "variable_orden_goldfeld_quandt": sort_feature,
                "consenso": consensus,
                "relevancia_practica": practical_relevance,
                "accion_recomendada": recommended_action,
            }
        ]
    )

    return {
        "heteroscedasticity_detected": bool(rejected_tests >= 2 or (rejected_tests >= 1 and visual_alert)),
        "tests_table": tests_table,
        "shape_summary": shape_summary,
        "group_test": group_dispersion,
        "residual_frame": residual_frame,
        "rejected_tests": rejected_tests,
        "spread_ratio": spread_ratio,
        "abs_resid_fitted_spearman": abs_resid_fitted_spearman,
        "goldfeld_quandt_sort_feature": sort_feature,
        "practical_relevance": practical_relevance,
        "recommended_action": recommended_action,
        "interpretation": interpretation,
        "consensus": consensus,
    }


def analyze_correlation(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    method: Literal["auto", "pearson", "spearman", "kendall"] = "auto",
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict[str, Any]:
    """Mide asociacion entre dos variables y explica si conviene Pearson o Spearman.

    Entradas:
        df: DataFrame fuente.
        x_column: Primera variable numerica.
        y_column: Segunda variable numerica.
        method: Metodo de correlacion o seleccion automatica.
        alpha: Nivel de significacion para la interpretacion.
        verbose: Si es True, imprime conclusion automatizada.

    Salidas:
        Diccionario con metodo elegido, coeficiente, p-valor e interpretacion.

    Pruebas ejecutadas:
        Pearson, Spearman o Kendall; en modo auto revisa normalidad aproximada con Shapiro-Wilk.
    """
    # Limpia el par analitico antes de decidir si conviene una asociacion lineal o monotona.
    _ensure_dataframe(df)
    _ensure_columns(df, [x_column, y_column])

    pair = df[[x_column, y_column]].dropna().copy()
    pair[x_column] = pd.to_numeric(pair[x_column], errors="coerce")
    pair[y_column] = pd.to_numeric(pair[y_column], errors="coerce")
    pair = pair.dropna()
    if len(pair) < 3:
        raise ValueError("Se requieren al menos 3 pares validos para estimar correlacion.")

    selected_method = method
    # En modo automatico usa normalidad aproximada para inclinarse por Pearson o Spearman.
    if method == "auto":
        x_normal = check_normality(pair[x_column], alpha=alpha, verbose=False)["is_normal"]
        y_normal = check_normality(pair[y_column], alpha=alpha, verbose=False)["is_normal"]
        selected_method = "pearson" if x_normal and y_normal else "spearman"

    if selected_method == "pearson":
        statistic, p_value = stats.pearsonr(pair[x_column], pair[y_column])
    elif selected_method == "spearman":
        statistic, p_value = stats.spearmanr(pair[x_column], pair[y_column])
    elif selected_method == "kendall":
        statistic, p_value = stats.kendalltau(pair[x_column], pair[y_column])
    else:
        raise ValueError("Metodo no soportado. Usa 'auto', 'pearson', 'spearman' o 'kendall'.")

    direction = "positiva" if statistic >= 0 else "negativa"
    strength = abs(float(statistic))
    if strength >= 0.7:
        magnitude = "fuerte"
    elif strength >= 0.4:
        magnitude = "moderada"
    elif strength >= 0.2:
        magnitude = "debil"
    else:
        magnitude = "muy debil"

    # Devuelve una lectura que combina significacion, direccion y magnitud del coeficiente.
    if p_value < alpha:
        interpretation = (
            f"P-valor = {p_value:.4f} < {alpha}. Hay evidencia de asociacion {direction} {magnitude}. "
            "Recuerda que correlacion no implica causalidad y que los confundidores pueden cambiar la historia."
        )
    else:
        interpretation = (
            f"P-valor = {p_value:.4f} >= {alpha}. No hay evidencia suficiente de asociacion monotona o lineal consistente."
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "method": selected_method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_pairs": int(len(pair)),
        "interpretation": interpretation,
    }


def _label_vif_severity(vif: float, moderate_threshold: float = 5.0, critical_threshold: float = 10.0) -> str:
    # Traduce el VIF a una banda de riesgo util para leer reportes sin recalcular reglas mentales.
    if vif > critical_threshold:
        return "intervencion_critica"
    if vif >= moderate_threshold:
        return "alerta_moderada"
    return "estable"


def _label_condition_severity(condition_index: float) -> str:
    # Sigue la heuristica industrial habitual para resumir estabilidad numerica del sistema.
    if condition_index > 30:
        return "intervencion_critica"
    if condition_index >= 10:
        return "alerta_moderada"
    return "estable"


def _build_vif_report(
    working: pd.DataFrame,
    vif_threshold: float,
) -> pd.DataFrame:
    # Construye la tabla VIF estandarizada para reutilizarla en chequeos y mitigacion iterativa.
    design = sm.add_constant(working, has_constant="add")
    rows: list[dict[str, Any]] = []
    for position, column in enumerate(design.columns[1:], start=1):
        vif_value = float(variance_inflation_factor(design.values, position))
        rows.append(
            {
                "columna": column,
                "vif": round(vif_value, 4),
                "nivel_alerta": _label_vif_severity(vif_value, moderate_threshold=vif_threshold),
            }
        )

    return pd.DataFrame(rows).sort_values(["vif", "columna"], ascending=[False, True]).reset_index(drop=True)


def _build_belsley_diagnostics(
    working: pd.DataFrame,
    condition_threshold: float,
    variance_share_threshold: float,
) -> dict[str, Any]:
    # Estandariza el diseno sin constante para evitar falsas alarmas por escala antes del analisis SVD.
    centered = working - working.mean(axis=0)
    scaled = centered / working.std(axis=0, ddof=0).replace(0, np.nan)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

    if scaled.shape[1] < 2:
        empty_components = pd.DataFrame(
            columns=["componente", "singular_value", "eigenvalue", "condition_index", "nivel_alerta"]
        )
        empty_variance = pd.DataFrame(columns=["componente", "condition_index", "feature", "variance_proportion"])
        empty_critical = pd.DataFrame(
            columns=[
                "componente",
                "condition_index",
                "features_comprometidas",
                "n_features_comprometidas",
                "max_variance_proportion",
            ]
        )
        return {
            "scaled_design": scaled,
            "components": empty_components,
            "variance_decomposition": empty_variance,
            "critical_components": empty_critical,
            "scaled_condition_number": float("nan"),
            "max_condition_index": float("nan"),
        }

    matrix = scaled.to_numpy(dtype=float, copy=True)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    eigenvalues = singular_values**2
    safe_singular_values = np.where(singular_values <= 1e-12, np.nan, singular_values)
    safe_eigenvalues = np.where(eigenvalues <= 1e-12, np.nan, eigenvalues)
    max_singular = float(np.nanmax(safe_singular_values))
    condition_indices = np.where(np.isnan(safe_singular_values), np.inf, max_singular / safe_singular_values)

    components = pd.DataFrame(
        {
            "componente": np.arange(1, len(singular_values) + 1),
            "singular_value": np.round(singular_values, 6),
            "eigenvalue": np.round(eigenvalues, 6),
            "condition_index": np.round(condition_indices, 4),
        }
    )
    components["nivel_alerta"] = components["condition_index"].apply(_label_condition_severity)

    raw_proportions = (vh.T**2) / safe_eigenvalues
    row_sums = np.nansum(raw_proportions, axis=1, keepdims=True)
    variance_proportions = np.divide(
        raw_proportions,
        row_sums,
        out=np.zeros_like(raw_proportions),
        where=row_sums > 0,
    )

    variance_rows: list[dict[str, Any]] = []
    critical_rows: list[dict[str, Any]] = []
    features = scaled.columns.tolist()
    for component_position, component_row in components.iterrows():
        implicated_features: list[str] = []
        implicated_proportions: list[float] = []
        for feature_position, feature_name in enumerate(features):
            proportion = float(variance_proportions[feature_position, component_position])
            variance_rows.append(
                {
                    "componente": int(component_row["componente"]),
                    "condition_index": float(component_row["condition_index"]),
                    "feature": feature_name,
                    "variance_proportion": round(proportion, 4),
                }
            )
            if proportion >= variance_share_threshold:
                implicated_features.append(feature_name)
                implicated_proportions.append(proportion)

        if float(component_row["condition_index"]) >= condition_threshold and len(implicated_features) >= 2:
            critical_rows.append(
                {
                    "componente": int(component_row["componente"]),
                    "condition_index": round(float(component_row["condition_index"]), 4),
                    "features_comprometidas": ", ".join(implicated_features),
                    "n_features_comprometidas": len(implicated_features),
                    "max_variance_proportion": round(max(implicated_proportions), 4),
                }
            )

    return {
        "scaled_design": scaled,
        "components": components,
        "variance_decomposition": pd.DataFrame(variance_rows),
        "critical_components": pd.DataFrame(critical_rows),
        "scaled_condition_number": round(float(np.nanmax(condition_indices)), 4),
        "max_condition_index": round(float(np.nanmax(condition_indices)), 4),
    }


def _build_iterative_mitigation_path(
    working: pd.DataFrame,
    vif_threshold: float,
    condition_threshold: float,
    variance_share_threshold: float,
) -> dict[str, Any]:
    # Purga iterativamente la variable con mayor VIF hasta recuperar un espacio interpretable.
    remaining_columns = list(working.columns)
    steps: list[dict[str, Any]] = []

    while len(remaining_columns) >= 2:
        current = working[remaining_columns]
        vif_report = _build_vif_report(current, vif_threshold=vif_threshold)
        spectral = _build_belsley_diagnostics(
            current,
            condition_threshold=condition_threshold,
            variance_share_threshold=variance_share_threshold,
        )
        max_vif = float(vif_report["vif"].max())
        critical_components = spectral["critical_components"]
        severe_belsley = not critical_components.empty
        if max_vif < vif_threshold and not severe_belsley:
            break

        drop_row = vif_report.iloc[0]
        steps.append(
            {
                "iteracion": len(steps) + 1,
                "columna_eliminada": drop_row["columna"],
                "vif_eliminado": float(drop_row["vif"]),
                "condition_index_previo": spectral["scaled_condition_number"],
                "n_componentes_criticos": int(len(critical_components)),
                "columnas_restantes_previo": ", ".join(remaining_columns),
            }
        )
        remaining_columns.remove(str(drop_row["columna"]))

    final_report = _build_vif_report(working[remaining_columns], vif_threshold=vif_threshold)
    return {
        "steps": pd.DataFrame(steps),
        "recommended_features": remaining_columns,
        "final_report": final_report,
    }


def calculate_vif(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    vif_threshold: float = 5.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Calcula VIF para diagnosticar multicolinealidad entre predictores numericos.

    Entradas:
        df: DataFrame fuente.
        columns: Columnas numericas a evaluar. Si es None, usa todas las numericas.
        vif_threshold: Umbral de alerta.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con la tabla VIF e interpretacion.

    Pruebas ejecutadas:
        Variance Inflation Factor sobre la matriz de predictores numericos, mas analisis
        espectral tipo Belsley sobre la matriz centrada y escalada para evitar falsas alarmas
        por diferencia de escala o por la presencia de la constante.
    """
    # Construye una matriz numerica limpia para medir redundancia explicativa entre predictores.
    _ensure_dataframe(df)
    selected_columns = list(columns) if columns is not None else list(df.select_dtypes(include=np.number).columns)
    _ensure_columns(df, selected_columns)
    working = df[selected_columns].apply(pd.to_numeric, errors="coerce").dropna()

    constant_columns = working.columns[working.var(ddof=0) <= 1e-12].tolist()
    if constant_columns:
        working = working.drop(columns=constant_columns)

    if working.shape[1] < 2:
        raise ValueError("VIF requiere al menos dos columnas numericas sin nulos tras el filtrado.")

    vif_report = _build_vif_report(working, vif_threshold=vif_threshold)
    belsley = _build_belsley_diagnostics(
        working,
        condition_threshold=30.0,
        variance_share_threshold=0.5,
    )
    mitigation = _build_iterative_mitigation_path(
        working,
        vif_threshold=vif_threshold,
        condition_threshold=30.0,
        variance_share_threshold=0.5,
    )

    problematic = vif_report[vif_report["vif"] >= vif_threshold]
    critical_components = belsley["critical_components"]
    if problematic.empty and critical_components.empty:
        interpretation = (
            f"No se detectaron predictores con VIF >= {vif_threshold}. El numero de condicion escalado es "
            f"{belsley['scaled_condition_number']:.4f} y no aparecen componentes Belsley criticos. "
            "La interpretacion de coeficientes es razonablemente estable."
        )
    else:
        top_columns = ", ".join(problematic.head(5)["columna"].tolist())
        top_columns = top_columns or ", ".join(critical_components.head(1)["features_comprometidas"].tolist())
        belsley_message = (
            "No se identificaron componentes criticos en el analisis de Belsley."
            if critical_components.empty
            else (
                f"El analisis de Belsley encontro {len(critical_components)} componente(s) critico(s), con "
                f"dependencia compartida en {critical_components.iloc[0]['features_comprometidas']}."
            )
        )
        mitigation_features = ", ".join(mitigation["recommended_features"])
        interpretation = (
            f"Se detecto multicolinealidad relevante en: {top_columns}. "
            f"El numero de condicion escalado es {belsley['scaled_condition_number']:.4f}. {belsley_message} "
            "Considera eliminar redundancias, combinar variables, regularizar con Ridge o rotar el espacio con PCA. "
            f"La purga iterativa por VIF sugiere conservar: {mitigation_features}."
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "report": vif_report,
        "interpretation": interpretation,
        "scaled_condition_number": belsley["scaled_condition_number"],
        "spectral_report": belsley["components"],
        "variance_decomposition": belsley["variance_decomposition"],
        "critical_components": critical_components,
        "mitigation_path": mitigation["steps"],
        "recommended_features": mitigation["recommended_features"],
        "post_mitigation_report": mitigation["final_report"],
        "dropped_constant_columns": constant_columns,
    }


def train_supervised_model(
    df: pd.DataFrame,
    target: str,
    problem_type: ProblemType | Literal["auto"] = "auto",
    algorithm: Literal[
        "auto",
        "logistic",
        "random_forest",
        "gradient_boosting",
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "knn",
        "xgboost",
        "lightgbm",
        "catboost",
        "gam",
        "mars",
        "mlp",
        "arima",
        "prophet",
        "lstm",
        "neural_network",
    ] = "auto",
    features: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    numeric_imputer: Literal["mean", "median", "knn", "iterative"] = "median",
    categorical_imputer: Literal["most_frequent", "constant"] = "most_frequent",
    scaler: Literal["standard", "robust", "none"] = "standard",
    apply_power_transform: bool = False,
    power_method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
    date_column: str | None = None,
    business_case: str | None = None,
    forecast_lags: int = 6,
    forecast_horizon: int | None = None,
    seasonality_period: int = 12,
    probability_calibration: Literal["none", "sigmoid", "isotonic", "auto"] = "auto",
    calibration_size: float = 0.2,
    verbose: bool = True,
) -> dict[str, Any]:
    """Entrena un modelo supervisado con pipeline reproducible e interpretacion automatica.

    Entradas:
        df: DataFrame fuente.
        target: Nombre de la variable objetivo.
        problem_type: Tipo de problema o deteccion automatica.
        algorithm: Algoritmo a usar. En modo auto elige un baseline interpretable.
        features: Subconjunto opcional de predictores.
        test_size: Proporcion del conjunto de prueba.
        random_state: Semilla de reproducibilidad.
        numeric_imputer: Estrategia de imputacion numerica.
        categorical_imputer: Estrategia de imputacion categorica.
        scaler: Tipo de escalado.
        apply_power_transform: Si aplica transformaciones de potencia en numericas.
        power_method: Metodo de potencia solicitado.
        date_column: Columna temporal obligatoria en forecasting y recomendada para validacion temporal.
        business_case: Caso de negocio que puede fijar benchmark por defecto desde el contrato YAML.
        forecast_lags: Numero de rezagos del objetivo para el pipeline temporal.
        forecast_horizon: Horizonte minimo del holdout temporal.
        seasonality_period: Periodicidad estacional usada para generar lags estacionales.
        probability_calibration: Metodo post-hoc para calibrar probabilidades binarias.
            En modo auto compara sigmoid e isotonic y solo aplica calibracion si mejora el holdout.
        calibration_size: Proporcion del train usada para calibracion interna.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con pipeline o modelo temporal, metricas, predicciones, importancia de variables,
        calibracion probabilistica e interpretacion.

    Pruebas ejecutadas:
        Metricas supervisadas de clasificacion, regresion o forecasting, calibracion probabilistica
        post-hoc para binaria cuando se solicita, y permutation importance como alternativa
        model-agnostic a SHAP cuando se trabaja solo con librerias estandar.
    """
    # Prepara la matriz supervisada y separa la variable objetivo antes de modelar.
    _ensure_dataframe(df)
    _ensure_columns(df, [target])

    feature_list = list(features) if features is not None else [column for column in df.columns if column != target]
    _ensure_columns(df, feature_list)
    working = df[feature_list + [target]].dropna(subset=[target]).copy()
    X = working[feature_list].copy()
    y = working[target].copy()

    requested_algorithm = _normalize_algorithm_name(algorithm)
    if requested_algorithm == "auto" and business_case is not None:
        requested_algorithm = resolve_business_case_benchmark_models(business_case)[0]

    # Decide automaticamente la familia del problema si el usuario no la fija de antemano.
    if problem_type == "auto":
        selected_problem = "forecasting" if requested_algorithm in FORECASTING_ALGORITHMS else _infer_problem_type(y)
    else:
        selected_problem = problem_type

    if selected_problem == "forecasting" or requested_algorithm in FORECASTING_ALGORITHMS:
        selected_algorithm = requested_algorithm if requested_algorithm != "auto" else "arima"
        return _train_temporal_forecasting_model(
            df=df,
            target=target,
            algorithm=selected_algorithm,
            features=feature_list,
            test_size=test_size,
            date_column=date_column,
            random_state=random_state,
            lag_count=forecast_lags,
            seasonality_period=seasonality_period,
            forecast_horizon=forecast_horizon,
            verbose=verbose,
        )

    if selected_problem == "classification":
        selected_algorithm = requested_algorithm if requested_algorithm != "auto" else "logistic"
        stratify = y if y.nunique(dropna=True) <= 20 else None
    else:
        selected_algorithm = requested_algorithm if requested_algorithm != "auto" else "linear"
        stratify = None

    effective_scaler = _resolve_effective_scaler(selected_algorithm, scaler)
    pipeline_info = build_preprocessing_pipeline(
        X,
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        scaler=effective_scaler,
        apply_power_transform=apply_power_transform,
        power_method=power_method,
        verbose=False,
    )
    estimator = _build_supervised_estimator(selected_problem, selected_algorithm, random_state)

    # Separa train y test antes de ajustar para evitar fuga en la evaluacion.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Encadena preprocesado y modelo en un unico objeto reproducible y desplegable.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", pipeline_info["preprocessor"]),
            ("model", estimator),
        ]
    )

    calibration_payload: dict[str, Any] | None = None
    calibration_message = ""
    fit_X = X_train
    fit_y = y_train
    calibration_X: pd.DataFrame | None = None
    calibration_y: pd.Series | None = None
    can_calibrate = (
        selected_problem == "classification"
        and y.nunique(dropna=True) == 2
        and hasattr(estimator, "predict_proba")
        and probability_calibration != "none"
        and calibration_size > 0
        and len(X_train) >= 80
        and y_train.value_counts().min() >= 10
    )
    if can_calibrate:
        fit_X, calibration_X, fit_y, calibration_y = train_test_split(
            X_train,
            y_train,
            test_size=calibration_size,
            random_state=random_state,
            stratify=y_train,
        )

    pipeline.fit(fit_X, fit_y)
    predictions = pipeline.predict(X_test)

    metrics: dict[str, float]
    probabilities: np.ndarray | None = None
    calibration_comparison = pd.DataFrame()
    # Calcula metricas alineadas con la naturaleza del objetivo y el tipo de salida disponible.
    if selected_problem == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        }
        if y_test.nunique(dropna=True) == 2 and hasattr(pipeline.named_steps["model"], "predict_proba"):
            raw_probabilities = pipeline.predict_proba(X_test)[:, 1]
            raw_brier_test = float(brier_score_loss(y_test, raw_probabilities))
            calibration_comparison_rows: list[dict[str, Any]] = [
                {
                    "method": "raw",
                    "brier_score": round(raw_brier_test, 4),
                    "brier_score_delta": 0.0,
                    "applied": True,
                    "selected_for_scoring": True,
                    "calibration_rows": 0,
                    "note": "Probabilidad cruda del modelo base.",
                }
            ]
            if calibration_X is not None and calibration_y is not None:
                calibration_probabilities = pipeline.predict_proba(calibration_X)[:, 1]
                candidate_methods = ["sigmoid", "isotonic"] if probability_calibration == "auto" else [probability_calibration]
                candidate_payloads: list[dict[str, Any]] = []
                for candidate_method in candidate_methods:
                    candidate_payload = _fit_probability_calibrator(
                        calibration_y.reset_index(drop=True),
                        calibration_probabilities,
                        method=candidate_method,
                        random_state=random_state,
                    )
                    if candidate_payload is None:
                        continue
                    candidate_probabilities = _apply_probability_calibrator(candidate_payload, raw_probabilities)
                    candidate_brier_test = float(brier_score_loss(y_test, candidate_probabilities))
                    candidate_payload["holdout_report"] = pd.DataFrame(
                        [
                            {
                                "brier_score_raw": round(raw_brier_test, 4),
                                "brier_score_calibrated": round(candidate_brier_test, 4),
                                "brier_score_delta": round(raw_brier_test - candidate_brier_test, 4),
                            }
                        ]
                    )
                    candidate_payload["candidate_probabilities"] = candidate_probabilities
                    candidate_payloads.append(candidate_payload)
                    calibration_comparison_rows.append(
                        {
                            "method": candidate_method,
                            "brier_score": round(candidate_brier_test, 4),
                            "brier_score_delta": round(raw_brier_test - candidate_brier_test, 4),
                            "applied": False,
                            "selected_for_scoring": False,
                            "calibration_rows": int(candidate_payload.get("calibration_rows", 0)),
                            "note": f"Calibrador {candidate_method} evaluado sobre el holdout.",
                        }
                    )

                if candidate_payloads:
                    best_payload = min(
                        candidate_payloads,
                        key=lambda payload: float(payload["holdout_report"].iloc[0]["brier_score_calibrated"]),
                    )
                    best_brier_test = float(best_payload["holdout_report"].iloc[0]["brier_score_calibrated"])
                    if best_brier_test + 1e-6 < raw_brier_test:
                        probabilities = best_payload["candidate_probabilities"]
                        calibration_payload = {key: value for key, value in best_payload.items() if key != "candidate_probabilities"}
                        calibration_payload["applied"] = True
                        for row in calibration_comparison_rows:
                            row["applied"] = row["method"] == calibration_payload["method"]
                            row["selected_for_scoring"] = row["method"] == calibration_payload["method"]
                        calibration_message = (
                            f" Se aplico calibracion {calibration_payload['method']} con un split interno de {len(calibration_y)} filas; "
                            f"el Brier en holdout paso de {raw_brier_test:.4f} a {best_brier_test:.4f}."
                        )
                    else:
                        calibration_payload = {
                            "method": "none",
                            "applied": False,
                            "reason": "Ningun calibrador mejoro el Brier del holdout frente a la probabilidad cruda.",
                            "holdout_report": pd.DataFrame(
                                [
                                    {
                                        "brier_score_raw": round(raw_brier_test, 4),
                                        "brier_score_calibrated": round(raw_brier_test, 4),
                                        "brier_score_delta": 0.0,
                                    }
                                ]
                            ),
                        }
                        probabilities = raw_probabilities
                        for row in calibration_comparison_rows:
                            row["applied"] = row["method"] == "raw"
                            row["selected_for_scoring"] = row["method"] == "raw"
                        calibration_message = (
                            " Se evaluaron calibradores post-hoc, pero se mantuvo la probabilidad cruda porque ninguna opcion mejoro el holdout."
                        )
                else:
                    probabilities = raw_probabilities
                    calibration_comparison_rows[0]["note"] = "No fue posible ajustar calibradores candidatos con el split interno disponible."
            else:
                probabilities = raw_probabilities
                calibration_comparison_rows[0]["note"] = "No se abrio split interno de calibracion; se conserva la probabilidad cruda."
            calibration_comparison = pd.DataFrame(calibration_comparison_rows)
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, probabilities)), 4)
            metrics["log_loss"] = round(float(log_loss(y_test, probabilities)), 4)
        interpretation_key = "roc_auc" if "roc_auc" in metrics else "f1"
    else:
        residuals = y_test - predictions
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
            "r2": round(float(r2_score(y_test, predictions)), 4),
            "error_medio": round(float(np.mean(residuals)), 4),
        }
        interpretation_key = "r2"

    scoring = "roc_auc" if selected_problem == "classification" and probabilities is not None else "f1_weighted" if selected_problem == "classification" else "r2"

    try:
        # Usa permutation importance para una lectura model-agnostic comparable entre algoritmos.
        importance = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=random_state,
            scoring=scoring,
        )
        feature_importance = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    except Exception:
        feature_importance = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": np.zeros(len(X_test.columns), dtype=float),
                "importance_std": np.zeros(len(X_test.columns), dtype=float),
            }
        )

    prediction_frame = pd.DataFrame({"actual": y_test.reset_index(drop=True), "predicted": predictions})
    if probabilities is not None:
        prediction_frame["predicted_probability"] = probabilities
        if selected_problem == "classification" and 'raw_probabilities' in locals():
            prediction_frame["predicted_probability_raw"] = raw_probabilities

    scaler_message = ""
    if effective_scaler != scaler:
        scaler_message = (
            f" El escalado solicitado se ajusto automaticamente a '{effective_scaler}' porque '{selected_algorithm}' es sensible a distancia."
        )
    interpretation = (
        f"Se entreno un modelo '{selected_algorithm}' para un problema de {selected_problem}. "
        f"{_summarize_metric(interpretation_key, metrics[interpretation_key])} "
        f"Usa permutation importance como baseline global rapido y complementalo con SHAP cuando necesites transparencia aditiva local-global y auditoria fina de contribuciones.{calibration_message}{scaler_message}"
    )
    _emit_interpretation(interpretation, verbose)

    return {
        "pipeline": pipeline,
        "problem_type": selected_problem,
        "algorithm": selected_algorithm,
        "metrics": pd.DataFrame([metrics]),
        "predictions": prediction_frame,
        "feature_importance": feature_importance,
        "X_test": X_test,
        "y_test": y_test.reset_index(drop=True),
        "probability_calibration": calibration_payload,
        "calibration_comparison": calibration_comparison,
        "interpretation": interpretation,
        "preprocessing": pipeline_info,
        "requested_scaler": scaler,
        "effective_scaler": effective_scaler,
        "business_case": business_case,
    }


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
) -> tuple[plt.Figure, plt.Axes]:
    """Grafica importancia de variables para validar la interpretacion del modelo.

    Entradas:
        feature_importance: Tabla con columnas feature, importance_mean e importance_std.
        top_n: Numero de variables a mostrar.

    Salidas:
        Figura y eje de Matplotlib.

    Pruebas ejecutadas:
        Visualizacion de permutation importance ya calculada.
    """
    # Ordena y acota la visualizacion para mostrar solo las variables mas influyentes.
    if feature_importance.empty:
        raise ValueError("La tabla de importancia esta vacia.")

    aplicar_tema_profesional()
    top = feature_importance.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#0F766E")
    ax.set_title("Importancia de variables")
    ax.set_xlabel("Permutation importance media")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig, ax


def plot_model_diagnostics(
    model_result: dict[str, Any],
) -> tuple[plt.Figure, np.ndarray]:
    """Genera graficos criticos para validar un modelo de clasificacion o regresion.

    Entradas:
        model_result: Salida de train_supervised_model.

    Salidas:
        Figura y arreglo de ejes.

    Pruebas ejecutadas:
        Residuales y predicho vs observado en regresion, o matriz de confusion y calibracion
        en clasificacion binaria cuando hay probabilidades disponibles.
    """
    # Decide automaticamente el panel diagnostico segun sea un problema de regresion o clasificacion.
    if "predictions" not in model_result or "problem_type" not in model_result:
        raise ValueError("model_result no tiene la estructura esperada de train_supervised_model.")

    predictions = model_result["predictions"].copy()
    problem_type = model_result["problem_type"]
    aplicar_tema_profesional()

    # En regresion interesa comparar ajuste y estructura del error; en clasificacion, separacion y calibracion.
    if problem_type == "regression":
        residuals = predictions["actual"] - predictions["predicted"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = np.atleast_1d(axes)

        axes[0].scatter(predictions["actual"], predictions["predicted"], alpha=0.7, color="#2563EB")
        line_min = min(predictions["actual"].min(), predictions["predicted"].min())
        line_max = max(predictions["actual"].max(), predictions["predicted"].max())
        axes[0].plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#EA580C")
        axes[0].set_title("Observado vs predicho")
        axes[0].set_xlabel("Valor real")
        axes[0].set_ylabel("Prediccion")

        axes[1].scatter(predictions["predicted"], residuals, alpha=0.7, color="#0F766E")
        axes[1].axhline(0, linestyle="--", color="#EA580C")
        axes[1].set_title("Residuos vs prediccion")
        axes[1].set_xlabel("Prediccion")
        axes[1].set_ylabel("Residuo")
        fig.tight_layout()
        return fig, axes

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = np.atleast_1d(axes)
    matrix = confusion_matrix(predictions["actual"], predictions["predicted"])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("Matriz de confusion")
    axes[0].set_xlabel("Predicho")
    axes[0].set_ylabel("Real")

    if "predicted_probability" in predictions.columns and predictions["actual"].nunique() == 2:
        observed, estimated = calibration_curve(
            predictions["actual"],
            predictions["predicted_probability"],
            n_bins=6,
        )
        axes[1].plot(estimated, observed, marker="o", color="#0F766E")
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="#EA580C")
        axes[1].set_title("Curva de calibracion")
        axes[1].set_xlabel("Probabilidad predicha")
        axes[1].set_ylabel("Frecuencia observada")
    else:
        axes[1].hist(predictions["predicted"], color="#2563EB")
        axes[1].set_title("Distribucion de predicciones")
        axes[1].set_xlabel("Clase predicha")
        axes[1].set_ylabel("Conteo")

    fig.tight_layout()
    return fig, axes


def compare_groups(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict[str, Any]:
    """Selecciona automaticamente el contraste adecuado para comparar grupos.

    Entradas:
        df: DataFrame fuente.
        value_column: Variable numerica a comparar.
        group_column: Variable de agrupacion.
        alpha: Nivel de significacion.
        verbose: Si es True, imprime conclusion automatizada.

    Salidas:
        Diccionario con el test elegido, diagnosticos previos, p-valor, tamano del efecto e interpretacion.

    Pruebas ejecutadas:
        Shapiro-Wilk por grupo, Levene/Brown-Forsythe, t de Student, Welch, Mann-Whitney,
        ANOVA clasico, ANOVA de Welch o Kruskal-Wallis segun corresponda.
    """
    # Construye los grupos comparables y recoge los diagnosticos previos al contraste principal.
    _ensure_dataframe(df)
    _ensure_columns(df, [value_column, group_column])

    working = df[[value_column, group_column]].dropna().copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    grouped_series = [group[value_column].reset_index(drop=True) for _, group in working.groupby(group_column)]
    group_names = working[group_column].dropna().unique().tolist()

    if len(grouped_series) < 2:
        raise ValueError("Se requieren al menos dos grupos validos para comparar.")

    # Evalua normalidad por grupo para saber si la via parametrica es defendible.
    normality_rows: list[dict[str, Any]] = []
    all_groups_normal = True
    for name, values in zip(group_names, grouped_series):
        if len(values) < 3:
            all_groups_normal = False
            normality_rows.append({"grupo": name, "n": len(values), "p_value": np.nan, "is_normal": False})
            continue
        result = check_normality(values, alpha=alpha, verbose=False)
        normality_rows.append(
            {
                "grupo": name,
                "n": len(values),
                "p_value": round(result["p_value"], 4),
                "is_normal": result["is_normal"],
            }
        )
        all_groups_normal = all_groups_normal and result["is_normal"]

    homogeneity = check_variance_homogeneity(
        working,
        value_column=value_column,
        group_column=group_column,
        alpha=alpha,
        center="median",
        verbose=False,
    )

    statistic: float
    p_value: float
    effect_size: float
    # Elige automaticamente el test mas coherente con numero de grupos y supuestos observados.
    if len(grouped_series) == 2:
        first_group, second_group = grouped_series
        if all_groups_normal and homogeneity["equal_variance"]:
            statistic, p_value = stats.ttest_ind(first_group, second_group, equal_var=True)
            selected_test = "Student t-test"
        elif all_groups_normal:
            statistic, p_value = stats.ttest_ind(first_group, second_group, equal_var=False)
            selected_test = "Welch t-test"
        else:
            statistic, p_value = stats.mannwhitneyu(first_group, second_group, alternative="two-sided")
            selected_test = "Mann-Whitney U"
        effect_size = _cohen_d(first_group, second_group)
    else:
        if all_groups_normal and homogeneity["equal_variance"]:
            statistic, p_value = stats.f_oneway(*grouped_series)
            selected_test = "ANOVA"
        elif all_groups_normal:
            welch_result = anova_oneway(grouped_series, use_var="unequal")
            statistic = float(welch_result.statistic)
            p_value = float(welch_result.pvalue)
            selected_test = "Welch ANOVA"
        else:
            statistic, p_value = stats.kruskal(*grouped_series)
            selected_test = "Kruskal-Wallis"
        effect_size = _eta_squared(grouped_series)

    if p_value < alpha:
        interpretation = (
            f"P-valor = {p_value:.4f} < {alpha}. Se rechaza la hipotesis nula y hay diferencias entre grupos segun {selected_test}. "
            "Acompanalo con tamano del efecto e inspeccion visual antes de convertirlo en una conclusion de negocio."
        )
    else:
        interpretation = (
            f"P-valor = {p_value:.4f} >= {alpha}. No hay evidencia suficiente para afirmar diferencias entre grupos con {selected_test}."
        )

    _emit_interpretation(interpretation, verbose)
    return {
        "test": selected_test,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "effect_size": round(float(effect_size), 4) if not np.isnan(effect_size) else np.nan,
        "group_normality": pd.DataFrame(normality_rows),
        "variance_homogeneity": homogeneity,
        "interpretation": interpretation,
    }


def fit_ols_inference(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    group_column: str | None = None,
    robust_cov: Literal["HC0", "HC1", "HC2", "HC3"] = "HC3",
    verbose: bool = True,
) -> dict[str, Any]:
    """Ajusta una regresion OLS con inferencia robusta y resumen interpretable.

    Entradas:
        df: DataFrame fuente.
        target: Variable objetivo continua.
        features: Predictores del modelo; las categoricas se expanden con dummies.
        group_column: Variable opcional para revisar dispersion residual entre segmentos.
        robust_cov: Tipo de errores robustos heterocedasticidad-consistentes.
        verbose: Si es True, imprime una conclusion ejecutiva.

    Salidas:
        Diccionario con el modelo statsmodels, coeficientes, intervalos, VIF, dispersion residual y diagnosticos.

    Pruebas ejecutadas:
        OLS con covarianza robusta HC3, VIF, Breusch-Pagan, White, Goldfeld-Quandt,
        Brown-Forsythe opcional por grupos, Durbin-Watson y diagnostico de influencia tipo Cook.
    """
    # Convierte el dataset a una matriz totalmente numerica compatible con statsmodels.
    _ensure_dataframe(df)
    required_columns = list(dict.fromkeys([target, *features, *( [group_column] if group_column else [])]))
    _ensure_columns(df, required_columns)

    working = df[required_columns].dropna().copy()
    y = pd.to_numeric(working[target], errors="coerce").astype(float)
    X = pd.get_dummies(working[list(features)], drop_first=True, dtype=float)
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    valid_mask = y.notna() & X.notna().all(axis=1)
    y = y.loc[valid_mask]
    X = X.loc[valid_mask]
    group_values = working.loc[valid_mask, group_column] if group_column else None

    if X.empty:
        raise ValueError("No quedaron observaciones validas para ajustar el modelo OLS.")

    # Ajusta OLS con errores robustos para no depender de homocedasticidad perfecta.
    design = sm.add_constant(X, has_constant="add").astype(float)
    model = sm.OLS(y, design).fit(cov_type=robust_cov)
    influence = model.get_influence()
    dispersion_audit = _build_structural_dispersion_audit(
        model=model,
        design=design,
        y=y,
        influence=influence,
        group_series=group_values,
        group_column=group_column,
        robust_cov=robust_cov,
    )
    summary_table = pd.DataFrame(
        {
            "coef": model.params,
            "std_error": model.bse,
            "p_value": model.pvalues,
            "ci_lower": model.conf_int()[0],
            "ci_upper": model.conf_int()[1],
        }
    ).round(4)

    # Complementa la inferencia con un chequeo de multicolinealidad sobre el mismo diseno.
    multicollinearity = calculate_vif(X, columns=X.columns.tolist(), verbose=False)
    vif_report = multicollinearity["report"].rename(columns={"columna": "feature"})
    critical_components = multicollinearity["critical_components"]
    max_vif = float(vif_report["vif"].max()) if not vif_report.empty else float("nan")

    cooks_distance = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    influence_report = pd.DataFrame(
        {
            "observation": y.index,
            "cooks_distance": cooks_distance,
            "leverage": leverage,
            "standard_residual": influence.resid_studentized_internal,
        }
    ).sort_values("cooks_distance", ascending=False)
    diagnostics = pd.DataFrame(
        [
            {
                "durbin_watson": round(float(durbin_watson(model.resid)), 4),
                "condition_number": round(float(model.condition_number), 4),
                "condition_number_scaled": multicollinearity["scaled_condition_number"],
                "max_vif": round(max_vif, 4) if pd.notna(max_vif) else np.nan,
                "belsley_componentes_criticos": int(len(critical_components)),
                "max_cooks_distance": round(float(np.max(cooks_distance)), 4),
                "pct_influential_points": round(float((cooks_distance > (4 / max(len(y), 1))).mean() * 100), 2),
                "heterocedasticidad_alertas": int(dispersion_audit["rejected_tests"]),
                "heterocedasticidad_consenso": dispersion_audit["consensus"],
                "heterocedasticidad_relevancia": dispersion_audit["practical_relevance"],
                "ratio_dispersion_q4_q1": round(float(dispersion_audit["spread_ratio"]), 4)
                if pd.notna(dispersion_audit["spread_ratio"])
                else np.nan,
            }
        ]
    )

    significant_predictors = summary_table.drop(index="const", errors="ignore")
    significant_predictors = significant_predictors[significant_predictors["p_value"] < 0.05]
    if significant_predictors.empty:
        interpretation = (
            "No se detectaron coeficientes claramente distintos de cero al 5% con errores robustos HC3. "
            "Esto no implica ausencia de senal, sino que la evidencia lineal condicionada es limitada o inestable."
        )
    else:
        top_features = ", ".join(significant_predictors.index[:5].tolist())
        interpretation = (
            f"Los predictores con evidencia mas clara al 5% son: {top_features}. "
            "Interpreta cada coeficiente como efecto marginal condicionado al resto y revisa el VIF antes de dar lectura causal o explicativa fuerte."
        )

    if not np.isnan(max_vif) and (max_vif >= 5 or not critical_components.empty):
        interpretation = (
            f"{interpretation} El diseno escalado presenta numero de condicion = {multicollinearity['scaled_condition_number']:.4f} "
            f"y {len(critical_components)} componente(s) Belsley critico(s); evita presentar estos coeficientes como efectos puros sin mitigacion adicional."
        )
    else:
        interpretation = (
            f"{interpretation} El diseno escalado no muestra alertas severas de multicolinealidad segun VIF y diagnostico espectral."
        )

    interpretation = f"{interpretation} {dispersion_audit['interpretation']}"

    _emit_interpretation(interpretation, verbose)
    return {
        "model": model,
        "summary_table": summary_table,
        "vif_report": vif_report,
        "multicollinearity": multicollinearity,
        "diagnostics": diagnostics,
        "influence_report": influence_report,
        "dispersion_audit": dispersion_audit,
        "residual_diagnostics": dispersion_audit["residual_frame"],
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "interpretation": interpretation,
    }


def audit_structural_dispersion(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    group_column: str | None = None,
    robust_cov: Literal["HC0", "HC1", "HC2", "HC3"] = "HC3",
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta solo la auditoria de heterocedasticidad y analisis de residuos.

    Entradas:
        df: DataFrame fuente.
        target: Variable objetivo continua.
        features: Predictores del ajuste OLS.
        group_column: Segmentacion opcional para Brown-Forsythe sobre residuos absolutos.
        robust_cov: Familia de errores robustos para el ajuste base.
        verbose: Si es True, imprime la lectura ejecutiva de dispersion.

    Salidas:
        Diccionario con contrastes de Breusch-Pagan, White, Goldfeld-Quandt,
        dispersion por grupos y tabla de residuos.

    Pruebas ejecutadas:
        Reutiliza fit_ols_inference para asegurar el mismo diseno, la misma covarianza robusta
        y el mismo contrato de salida que usa el resto del ecosistema.
    """
    result = fit_ols_inference(
        df,
        target=target,
        features=features,
        group_column=group_column,
        robust_cov=robust_cov,
        verbose=False,
    )
    dispersion = result["dispersion_audit"]
    _emit_interpretation(dispersion["interpretation"], verbose)
    return dispersion


def plot_structural_dispersion_diagnostics(
    ols_result: dict[str, Any],
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica patron residual para detectar embudos, megafonos o dispersion irregular.

    Entradas:
        ols_result: Salida de fit_ols_inference.

    Salidas:
        Figura y arreglo de ejes.

    Pruebas ejecutadas:
        No ejecuta nuevos contrastes; visualiza residuos estandarizados y scale-location.
    """
    # Hace visible si la varianza residual crece con el valor ajustado aunque el modelo siga rankeando bien.
    if "residual_diagnostics" not in ols_result:
        raise ValueError("ols_result no contiene residual_diagnostics; usa la salida de fit_ols_inference.")

    residual_frame = ols_result["residual_diagnostics"].copy()
    aplicar_tema_profesional()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = np.atleast_1d(axes)

    axes[0].scatter(
        residual_frame["fitted"],
        residual_frame["standardized_residual"],
        alpha=0.65,
        color="#2563EB",
    )
    sns.regplot(
        data=residual_frame,
        x="fitted",
        y="standardized_residual",
        lowess=True,
        scatter=False,
        ax=axes[0],
        line_kws={"color": "#EA580C", "linewidth": 2},
    )
    axes[0].axhline(0, linestyle="--", color="#0F172A", linewidth=1)
    axes[0].axhline(2, linestyle=":", color="#94A3B8", linewidth=1)
    axes[0].axhline(-2, linestyle=":", color="#94A3B8", linewidth=1)
    axes[0].set_title("Residuos estandarizados vs ajustados")
    axes[0].set_xlabel("Valor ajustado")
    axes[0].set_ylabel("Residuo estandarizado")

    axes[1].scatter(
        residual_frame["fitted"],
        residual_frame["sqrt_abs_standardized_residual"],
        alpha=0.65,
        color="#0F766E",
    )
    sns.regplot(
        data=residual_frame,
        x="fitted",
        y="sqrt_abs_standardized_residual",
        lowess=True,
        scatter=False,
        ax=axes[1],
        line_kws={"color": "#DC2626", "linewidth": 2},
    )
    axes[1].set_title("Scale-location")
    axes[1].set_xlabel("Valor ajustado")
    axes[1].set_ylabel("Raiz(|residuo estandarizado|)")

    fig.tight_layout()
    return fig, axes


def plot_ols_influence_diagnostics(
    ols_result: dict[str, Any],
    top_n: int = 15,
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica residuos y puntos influyentes de un ajuste OLS.

    Entradas:
        ols_result: Salida de fit_ols_inference.
        top_n: Numero de observaciones influyentes a mostrar.

    Salidas:
        Figura y arreglo de ejes.

    Pruebas ejecutadas:
        Visualizacion de residuos vs ajustados y ranking de Cook's distance.
    """
    # Hace visible si el ajuste lineal esta dominado por pocos puntos o residuos estructurados.
    if "model" not in ols_result or "influence_report" not in ols_result:
        raise ValueError("ols_result no tiene la estructura esperada de fit_ols_inference.")

    model = ols_result["model"]
    influence_report = ols_result["influence_report"].head(top_n).sort_values("cooks_distance")
    aplicar_tema_profesional()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = np.atleast_1d(axes)
    axes[0].scatter(model.fittedvalues, model.resid, alpha=0.7, color="#2563EB")
    axes[0].axhline(0, linestyle="--", color="#EA580C")
    axes[0].set_title("Residuos vs ajustados")
    axes[0].set_xlabel("Valor ajustado")
    axes[0].set_ylabel("Residuo")

    axes[1].barh(influence_report["observation"].astype(str), influence_report["cooks_distance"], color="#0F766E")
    axes[1].set_title("Top Cook distance")
    axes[1].set_xlabel("Cook's distance")
    axes[1].set_ylabel("Observacion")
    fig.tight_layout()
    return fig, axes


def plot_group_distributions(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Dibuja boxplots y violin plots para validar diferencias de grupos.

    Entradas:
        df: DataFrame fuente.
        value_column: Variable numerica.
        group_column: Variable de agrupacion.

    Salidas:
        Figura y arreglo de ejes.

    Pruebas ejecutadas:
        No ejecuta una prueba; visualiza dispersion, asimetria y valores extremos por grupo.
    """
    # Combina boxplot y violin plot para ver dispersion, asimetria y extremos por segmento.
    _ensure_dataframe(df)
    _ensure_columns(df, [value_column, group_column])
    aplicar_tema_profesional()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = np.atleast_1d(axes)
    sns.boxplot(data=df, x=group_column, y=value_column, ax=axes[0], color="#93C5FD")
    axes[0].set_title("Boxplot por grupo")
    axes[0].tick_params(axis="x", rotation=20)

    sns.violinplot(data=df, x=group_column, y=value_column, ax=axes[1], color="#86EFAC")
    axes[1].set_title("Violin plot por grupo")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig, axes


# Extensiones v3: memoria temporal, parsimonia e incertidumbre operativa.
def fractional_difference(
    series: pd.Series,
    d: float = 0.4,
    weight_threshold: float = 1e-4,
    max_window: int = 48,
) -> pd.Series:
    """Aplica diferenciacion fraccional a una serie preservando memoria parcial."""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    weights = [1.0]
    order = 1
    # Corta la recursion cuando el peso incremental deja de aportar senal o la ventana ya no es util.
    while True:
        next_weight = -weights[-1] * (d - order + 1) / order
        if abs(next_weight) < weight_threshold or order >= max_window:
            break
        weights.append(float(next_weight))
        order += 1

    width = len(weights)
    reversed_weights = np.array(weights[::-1], dtype=float)
    values = numeric.to_numpy(dtype=float)
    transformed = np.full(len(values), np.nan, dtype=float)
    # Las primeras filas quedan en NaN por construccion: todavia no existe suficiente historia para aplicar el operador.
    for position in range(width - 1, len(values)):
        window = values[position - width + 1 : position + 1]
        if np.isnan(window).any():
            continue
        transformed[position] = float(np.dot(reversed_weights, window))
    return pd.Series(transformed, index=series.index, name=f"{series.name}_fracdiff")


def _build_classifier(algorithm: str, random_state: int) -> Any:
    """Devuelve el clasificador homologado para las rutinas avanzadas v3."""
    # Se apoya en la misma fabrica del entrenamiento supervisado para no desalinear benchmark y validacion avanzada.
    return _build_supervised_estimator("classification", algorithm, random_state)


def _fit_classifier_with_explicit_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    algorithm: Literal[
        "logistic",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
        "xgboost",
        "catboost",
        "knn",
        "mlp",
        "neural_network",
    ],
    random_state: int,
) -> dict[str, Any]:
    """Entrena un pipeline sobre un split ya definido y devuelve metricas homogeneas."""
    # Reutiliza el mismo contrato de preprocesado para que todas las extensiones v3 hablen el mismo formato.
    preprocessor_info = build_preprocessing_pipeline(X_train, verbose=False)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_info["preprocessor"]),
            ("model", _build_classifier(algorithm, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    # El umbral 0.5 se usa solo para derivar etiquetas auxiliares; la comparacion principal vive en probabilidades y scoring rules.
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = float(roc_auc_score(y_test, probabilities)) if y_test.nunique(dropna=True) == 2 else float("nan")
    return {
        "pipeline": pipeline,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "log_loss": round(float(log_loss(y_test, probabilities, labels=[0, 1])), 4),
            "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        },
    }


def _fit_logit_information_criteria(
    preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float | str]:
    """Calcula AIC/BIC sobre el espacio transformado del modelo logistico."""
    # statsmodels necesita una matriz de diseno de rango completo; el preprocesado de sklearn puede incluir dummies redundantes y columnas casi constantes.
    transformed = np.asarray(preprocessor.transform(X_train), dtype=float)
    if transformed.ndim == 1:
        transformed = transformed.reshape(-1, 1)

    finite_columns = np.isfinite(transformed).all(axis=0)
    filtered = transformed[:, finite_columns]
    variable_columns = np.nanstd(filtered, axis=0) > 1e-10 if filtered.size else np.array([], dtype=bool)
    filtered = filtered[:, variable_columns] if filtered.size else filtered
    if filtered.size == 0:
        return {
            "aic": float("nan"),
            "bic": float("nan"),
            "log_likelihood": float("nan"),
            "n_parameters": 0,
            "waic_status": "not_computed: empty_design_after_filtering",
        }

    design_matrix = sm.tools.tools.fullrank(filtered)
    design = sm.add_constant(design_matrix, has_constant="add")
    if design.shape[0] <= design.shape[1]:
        return {
            "aic": float("nan"),
            "bic": float("nan"),
            "log_likelihood": float("nan"),
            "n_parameters": int(design.shape[1]),
            "waic_status": "not_computed: insufficient_observations_for_full_rank_design",
        }

    last_exception: Exception | None = None
    for method in ("lbfgs", "bfgs", "newton"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=HessianInversionWarning)
                warnings.simplefilter("ignore", category=StatsmodelsConvergenceWarning)
                fit = sm.Logit(y_train.astype(float).to_numpy(), np.asarray(design, dtype=float)).fit(
                    disp=False,
                    method=method,
                    maxiter=300,
                )
            return {
                "aic": round(float(fit.aic), 4),
                "bic": round(float(fit.bic), 4),
                "log_likelihood": round(float(fit.llf), 4),
                "n_parameters": int(design.shape[1]),
                "waic_status": "not_applicable_without_bayesian_posterior",
            }
        except Exception as exc:
            last_exception = exc

    # Si el ajuste sigue fallando tras limpiar rango y probar varios optimizadores, se conserva el fallo como metadata auditable.
    return {
        "aic": float("nan"),
        "bic": float("nan"),
        "log_likelihood": float("nan"),
        "n_parameters": int(design.shape[1]),
        "waic_status": f"not_computed: {type(last_exception).__name__}",
    }


def run_logistic_parsimony_study(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    feature_steps: Sequence[int],
    test_size: float = 0.25,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua modelos logisticos anidados con AIC/BIC y pesos de Akaike."""
    _ensure_dataframe(df)
    _ensure_columns(df, [target, *features])
    working = df[list(dict.fromkeys([*features, target]))].dropna(subset=[target]).copy()
    X = working[list(features)].copy()
    y = working[target].copy()
    stratify = y if y.nunique(dropna=True) == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Fuerza una familia de modelos anidados sobre el prefijo de variables para que AIC/BIC comparen complejidad real.
    candidate_sizes = sorted({min(max(2, int(size)), len(features)) for size in feature_steps} | {len(features)})
    rows: list[dict[str, Any]] = []
    probability_matrix: list[np.ndarray] = []
    subsets: list[list[str]] = []
    for size in candidate_sizes:
        subset = list(features[:size])
        fitted = _fit_classifier_with_explicit_split(
            X_train[subset],
            y_train,
            X_test[subset],
            y_test,
            algorithm="logistic",
            random_state=random_state,
        )
        information = _fit_logit_information_criteria(
            fitted["pipeline"].named_steps["preprocessor"],
            X_train[subset],
            y_train,
        )
        probability_matrix.append(fitted["probabilities"])
        subsets.append(subset)
        rows.append(
            {
                "candidate": f"logistic_top_{size}",
                "n_features": size,
                "selected_features": ", ".join(subset),
                "roc_auc": fitted["metrics"]["roc_auc"],
                "log_loss": fitted["metrics"]["log_loss"],
                "brier_score": fitted["metrics"]["brier_score"],
                "aic": information["aic"],
                "bic": information["bic"],
                "log_likelihood": information["log_likelihood"],
                "n_parameters": information["n_parameters"],
                "waic_status": information["waic_status"],
            }
        )

    summary = pd.DataFrame(rows)
    summary["akaike_weight"] = 0.0
    valid_mask = summary["aic"].notna()
    if valid_mask.any():
        min_aic = float(summary.loc[valid_mask, "aic"].min())
        delta = summary.loc[valid_mask, "aic"] - min_aic
        weights = np.exp(-0.5 * delta)
        weights = weights / weights.sum()
        summary.loc[valid_mask, "akaike_weight"] = weights.round(6)

    # La recomendacion operativa prioriza BIC para penalizar complejidad de forma mas conservadora que AIC.
    if summary["bic"].notna().any():
        recommended_position = int(summary.sort_values(["bic", "roc_auc", "log_loss"], ascending=[True, False, True]).index[0])
    else:
        recommended_position = int(summary.sort_values(["roc_auc", "log_loss"], ascending=[False, True]).index[0])
    recommended_features = subsets[recommended_position]

    if summary["akaike_weight"].sum() > 0:
        # El consenso ponderado suaviza la eleccion puntual cuando varios candidatos quedan tecnicamente muy cerca.
        weight_vector = summary["akaike_weight"].to_numpy(dtype=float)
        stacked = np.vstack(probability_matrix)
        ensemble_probability = (weight_vector[:, None] * stacked).sum(axis=0)
    else:
        ensemble_probability = probability_matrix[recommended_position]

    ensemble_metrics = pd.DataFrame(
        [
            {
                "roc_auc": round(float(roc_auc_score(y_test, ensemble_probability)), 4),
                "log_loss": round(float(log_loss(y_test, ensemble_probability, labels=[0, 1])), 4),
                "brier_score": round(float(brier_score_loss(y_test, ensemble_probability)), 4),
                "waic_status": "not_applicable_without_bayesian_posterior",
            }
        ]
    )
    interpretation = (
        "La parsimonia v3 compara modelos logisticos anidados donde AIC y BIC si son comparables de forma rigurosa. "
        "El criterio operativo privilegia BIC para sesgar hacia especificaciones mas simples y usa pesos de Akaike para un consenso estable. "
        "WAIC se declara como extension bayesiana y no se calcula sobre estimadores frequentistas de scikit-learn."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "summary": summary.sort_values(["bic", "roc_auc", "log_loss"], ascending=[True, False, True]).reset_index(drop=True),
        "ensemble_metrics": ensemble_metrics,
        "ensemble_predictions": pd.DataFrame(
            {
                "actual": y_test.reset_index(drop=True),
                "ensemble_probability": ensemble_probability,
                "source_index": y_test.index.to_numpy(),
            }
        ),
        "recommended_features": recommended_features,
        "interpretation": interpretation,
    }


def run_purged_temporal_validation(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    date_column: str,
    algorithm: Literal[
        "logistic",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
        "xgboost",
        "catboost",
        "knn",
        "mlp",
        "neural_network",
    ],
    n_splits: int,
    purge_gap_days: int,
    embargo_gap_days: int,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta una validacion temporal conservadora con purga y embargo."""
    _ensure_dataframe(df)
    _ensure_columns(df, [target, date_column, *features])
    working = df[list(dict.fromkeys([date_column, target, *features]))].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column, target]).sort_values(date_column).reset_index(drop=True)
    unique_dates = np.array(sorted(working[date_column].dropna().unique()))
    # Cada bloque temporal actua una vez como holdout para respetar causalidad cronologica.
    date_blocks = [block for block in np.array_split(unique_dates, max(2, n_splits)) if len(block) > 0]

    fold_rows: list[dict[str, Any]] = []
    for fold_number, test_dates in enumerate(date_blocks, start=1):
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])
        train_cutoff = test_start - pd.Timedelta(days=purge_gap_days)
        embargo_end = test_end + pd.Timedelta(days=embargo_gap_days)

        # La purga retira historia demasiado cercana al corte y el embargo evita recontaminar el entrenamiento con informacion casi futura.
        train_mask = working[date_column] < train_cutoff
        test_mask = working[date_column].between(test_start, test_end)
        purge_mask = working[date_column].between(train_cutoff, test_start, inclusive="left")
        embargo_mask = working[date_column].between(test_end, embargo_end, inclusive="right")

        train_df = working.loc[train_mask].copy()
        test_df = working.loc[test_mask].copy()
        if len(train_df) < 80 or len(test_df) < 20 or train_df[target].nunique(dropna=True) < 2 or test_df[target].nunique(dropna=True) < 2:
            continue

        fitted = _fit_classifier_with_explicit_split(
            train_df[list(features)],
            train_df[target],
            test_df[list(features)],
            test_df[target],
            algorithm=algorithm,
            random_state=random_state + fold_number,
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "train_end": train_df[date_column].max(),
                "test_start": test_start,
                "test_end": test_end,
                "embargo_end": embargo_end,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "purged_rows": int(purge_mask.sum()),
                "embargoed_rows": int(embargo_mask.sum()),
                "roc_auc": fitted["metrics"]["roc_auc"],
                "log_loss": fitted["metrics"]["log_loss"],
                "brier_score": fitted["metrics"]["brier_score"],
            }
        )

    fold_report = pd.DataFrame(fold_rows)
    if fold_report.empty:
        summary = pd.DataFrame(
            [{
                "n_folds": 0,
                "mean_roc_auc": np.nan,
                "std_roc_auc": np.nan,
                "mean_log_loss": np.nan,
                "mean_brier_score": np.nan,
            }]
        )
        interpretation = "No hubo suficientes ventanas temporales validas para ejecutar la validacion purgada con embargo."
    else:
        summary = pd.DataFrame(
            [{
                "n_folds": int(len(fold_report)),
                "mean_roc_auc": round(float(fold_report["roc_auc"].mean()), 4),
                "std_roc_auc": round(float(fold_report["roc_auc"].std(ddof=0)), 4),
                "mean_log_loss": round(float(fold_report["log_loss"].mean()), 4),
                "mean_brier_score": round(float(fold_report["brier_score"].mean()), 4),
            }]
        )
        interpretation = (
            "La validacion temporal v3 entrena solo con historia previa al bloque de prueba y elimina observaciones cercanas al corte para reducir leakage serial. "
            "Un desvio bajo entre folds sugiere estabilidad de regimen; un desvio alto advierte sensibilidad temporal."
        )
    _emit_interpretation(interpretation, verbose)
    return {
        "summary": summary,
        "fold_report": fold_report,
        "interpretation": interpretation,
    }


def run_bootstrap_prediction_intervals(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    date_column: str,
    algorithm: Literal[
        "logistic",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
        "xgboost",
        "catboost",
        "knn",
        "mlp",
        "neural_network",
    ],
    n_iterations: int,
    alpha: float,
    random_state: int = 42,
    id_column: str = "cliente_id",
    verbose: bool = True,
) -> dict[str, Any]:
    """Genera intervalos de pronostico mediante bootstrap multivariable."""
    _ensure_dataframe(df)
    _ensure_columns(df, [target, date_column, *features])
    working = (
        df[list(dict.fromkeys([id_column, date_column, target, *features]))].copy()
        if id_column in df.columns
        else df[list(dict.fromkeys([date_column, target, *features]))].copy()
    )
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column, target]).sort_values(date_column).reset_index(drop=True)
    # Se reserva el tramo mas reciente como referencia para medir incertidumbre fuera de muestra, no sobre la misma historia reamuestrada.
    reference_cut = working[date_column].quantile(0.65)
    train_df = working[working[date_column] <= reference_cut].copy()
    reference_df = working[working[date_column] > reference_cut].copy()
    if train_df.empty or reference_df.empty:
        midpoint = len(working) // 2
        train_df = working.iloc[:midpoint].copy()
        reference_df = working.iloc[midpoint:].copy()

    base_fit = _fit_classifier_with_explicit_split(
        train_df[list(features)],
        train_df[target],
        reference_df[list(features)],
        reference_df[target],
        algorithm=algorithm,
        random_state=random_state,
    )
    bootstrap_samples: list[np.ndarray] = []
    for iteration in range(int(n_iterations)):
        # El bootstrap remuestrea solo train para approximar sensibilidad del score ante cambios plausibles en la muestra historica.
        sample = train_df.sample(n=len(train_df), replace=True, random_state=random_state + iteration)
        if sample[target].nunique(dropna=True) < 2:
            continue
        fitted = _fit_classifier_with_explicit_split(
            sample[list(features)],
            sample[target],
            reference_df[list(features)],
            reference_df[target],
            algorithm=algorithm,
            random_state=random_state + iteration + 1,
        )
        bootstrap_samples.append(fitted["probabilities"])

    if bootstrap_samples:
        stacked = np.vstack(bootstrap_samples)
        lower = np.quantile(stacked, alpha / 2, axis=0)
        median = np.quantile(stacked, 0.5, axis=0)
        upper = np.quantile(stacked, 1 - alpha / 2, axis=0)
    else:
        lower = median = upper = base_fit["probabilities"]

    intervals = pd.DataFrame(
        {
            "actual": reference_df[target].reset_index(drop=True),
            "probability_base": base_fit["probabilities"],
            "prediction_interval_p05": np.round(lower, 4),
            "prediction_interval_p50": np.round(median, 4),
            "prediction_interval_p95": np.round(upper, 4),
        }
    )
    if id_column in reference_df.columns:
        intervals[id_column] = reference_df[id_column].reset_index(drop=True)
    intervals["interval_width"] = (intervals["prediction_interval_p95"] - intervals["prediction_interval_p05"]).round(4)
    portfolio_summary = pd.DataFrame(
        [{
            "bootstrap_iterations_effective": int(len(bootstrap_samples)),
            "mean_probability_base": round(float(intervals["probability_base"].mean()), 4),
            "mean_interval_width": round(float(intervals["interval_width"].mean()), 4),
            "high_risk_share_p05": round(float((intervals["prediction_interval_p05"] >= 0.5).mean()), 4),
            "high_risk_share_p95": round(float((intervals["prediction_interval_p95"] >= 0.5).mean()), 4),
        }]
    )
    interpretation = (
        "La v3 reporta intervalos de pronostico y no solo un promedio puntual. "
        "Un intervalo ancho indica inestabilidad predictiva sobre un cliente concreto y conviene leerlo como margen de duda operativa, no como error del promedio muestral."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "prediction_intervals": intervals.sort_values("interval_width", ascending=False).reset_index(drop=True),
        "portfolio_summary": portfolio_summary,
        "interpretation": interpretation,
    }


def _safe_rate(numerator: float, denominator: float) -> float:
    """Evita divisiones por cero en metricas de grupo."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


# Extensiones v3: equidad, activacion y gobierno.
def run_fairness_audit(
    reference_df: pd.DataFrame,
    model_result: dict[str, Any] | None,
    target: str,
    sensitive_columns: Sequence[str],
    age_bins: Sequence[int],
    verbose: bool = True,
) -> dict[str, Any]:
    """Evalua brechas de equidad en seleccion, error y calibracion por grupo."""
    if model_result is None or "X_test" not in model_result or "predictions" not in model_result:
        empty_summary = pd.DataFrame([{"sensitive_feature": "not_run", "status": "not_run"}])
        return {
            "summary": empty_summary,
            "group_metrics": pd.DataFrame(),
            "interpretation": "La auditoria de equidad no se ejecuto porque no hubo un modelo disponible.",
        }

    # La auditoria se monta sobre el mismo holdout del modelo para no mezclar grupos con distribuciones distintas a las evaluadas.
    base = model_result["X_test"].copy().reset_index(drop=True)
    predictions = model_result["predictions"].copy().reset_index(drop=True)
    base[target] = model_result["y_test"].reset_index(drop=True)
    base["predicted_label"] = predictions["predicted"].astype(int)
    probability_column = "predicted_probability" if "predicted_probability" in predictions.columns else None
    base["predicted_probability"] = predictions[probability_column] if probability_column is not None else predictions["predicted"].astype(float)

    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for sensitive_feature in sensitive_columns:
        if sensitive_feature == "edad" and "edad" in base.columns:
            # Edad se convierte en tramos auditables para evitar comparar cada ano como si fuese un grupo independiente.
            buckets = [-np.inf, *[int(item) for item in age_bins], np.inf]
            labels = ["hasta_30", "31_45", "46_60", "61_plus"]
            group_series = pd.cut(pd.to_numeric(base["edad"], errors="coerce"), bins=buckets, labels=labels)
            feature_label = "edad_bucket"
        elif sensitive_feature in base.columns:
            group_series = base[sensitive_feature].astype(str).fillna("<NA>")
            feature_label = sensitive_feature
        else:
            summary_rows.append(
                {
                    "sensitive_feature": sensitive_feature,
                    "status": "missing_in_dataset",
                    "demographic_parity_difference": np.nan,
                    "equal_opportunity_difference": np.nan,
                    "equalized_odds_difference": np.nan,
                }
            )
            continue

        audit_frame = base.copy()
        audit_frame[feature_label] = group_series.astype(str).fillna("<NA>")
        by_group = audit_frame.groupby(feature_label, dropna=False)
        selection_rates: list[float] = []
        true_positive_rates: list[float] = []
        false_positive_rates: list[float] = []
        for group_name, group in by_group:
            # Se separan seleccion, TPR y FPR para no colapsar equidad de acceso y equidad de error en una sola cifra.
            actual = group[target].astype(int)
            predicted = group["predicted_label"].astype(int)
            positives = int((actual == 1).sum())
            negatives = int((actual == 0).sum())
            tp = int(((predicted == 1) & (actual == 1)).sum())
            fp = int(((predicted == 1) & (actual == 0)).sum())
            selection_rate = float(predicted.mean()) if len(group) else float("nan")
            tpr = _safe_rate(tp, positives)
            fpr = _safe_rate(fp, negatives)
            selection_rates.append(selection_rate)
            if not np.isnan(tpr):
                true_positive_rates.append(tpr)
            if not np.isnan(fpr):
                false_positive_rates.append(fpr)
            metric_rows.append(
                {
                    "sensitive_feature": feature_label,
                    "group": group_name,
                    "support": int(len(group)),
                    "selection_rate": round(selection_rate, 4),
                    "actual_rate": round(float(actual.mean()), 4),
                    "mean_probability": round(float(group["predicted_probability"].mean()), 4),
                    "tpr": round(tpr, 4) if not np.isnan(tpr) else np.nan,
                    "fpr": round(fpr, 4) if not np.isnan(fpr) else np.nan,
                    "calibration_gap": round(float(abs(group["predicted_probability"].mean() - actual.mean())), 4),
                }
            )

        dp_diff = max(selection_rates) - min(selection_rates) if selection_rates else float("nan")
        dp_ratio = min(selection_rates) / max(selection_rates) if selection_rates and max(selection_rates) > 0 else float("nan")
        eo_diff = max(true_positive_rates) - min(true_positive_rates) if len(true_positive_rates) >= 2 else float("nan")
        fpr_diff = max(false_positive_rates) - min(false_positive_rates) if len(false_positive_rates) >= 2 else float("nan")
        eod_diff = max(value for value in [eo_diff, fpr_diff] if not np.isnan(value)) if any(not np.isnan(value) for value in [eo_diff, fpr_diff]) else float("nan")
        summary_rows.append(
            {
                "sensitive_feature": feature_label,
                "status": "evaluated",
                "demographic_parity_difference": round(dp_diff, 4) if not np.isnan(dp_diff) else np.nan,
                "demographic_parity_ratio": round(dp_ratio, 4) if not np.isnan(dp_ratio) else np.nan,
                "equal_opportunity_difference": round(eo_diff, 4) if not np.isnan(eo_diff) else np.nan,
                "equalized_odds_difference": round(eod_diff, 4) if not np.isnan(eod_diff) else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    interpretation = (
        "La auditoria de equidad v3 separa paridad de seleccion de calidad de servicio. "
        "Una brecha elevada no implica automaticamente injusticia regulatoria, pero si exige revision de contexto, cobertura y dano potencial por grupo."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "summary": summary,
        "group_metrics": pd.DataFrame(metric_rows),
        "interpretation": interpretation,
    }


def build_consensus_gap_report(
    scorecard: pd.DataFrame,
    group_column: str = "macro_segmento",
    score_column: str = "probabilidad_abandono",
    value_column: str = "valor_esperado_contacto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Mide la brecha de senal del modelo frente al consenso segmental."""
    enriched = scorecard.copy()
    if group_column not in enriched.columns:
        enriched[group_column] = "consenso_global"
    # Primero se calcula la referencia promedio del segmento; luego se mide cuanto se aparta cada cliente de esa linea base.
    group_baseline = (
        enriched.groupby(group_column)
        .agg(
            consenso_score=(score_column, "mean"),
            consenso_valor=(value_column, "mean"),
        )
        .reset_index()
    )
    enriched = enriched.merge(group_baseline, on=group_column, how="left")
    enriched["brecha_score_vs_consenso"] = (enriched[score_column] - enriched["consenso_score"]).round(4)
    enriched["brecha_valor_vs_consenso"] = (enriched[value_column] - enriched["consenso_valor"]).round(2)
    summary = (
        enriched.groupby(group_column)
        .agg(
            clientes=(score_column, "size"),
            brecha_score_media=("brecha_score_vs_consenso", "mean"),
            brecha_valor_media=("brecha_valor_vs_consenso", "mean"),
            score_medio=(score_column, "mean"),
            valor_medio=(value_column, "mean"),
        )
        .round(4)
        .reset_index()
    )
    interpretation = (
        "La brecha contra consenso resume donde el modelo detecta un riesgo o valor que la media del segmento no esta capturando. "
        "Una brecha grande y positiva sugiere oportunidad diferencial; una brecha cercana a cero indica senal muy alineada con el consenso."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "scorecard": enriched,
        "summary": summary,
        "interpretation": interpretation,
    }


def build_bandit_policy(
    scorecard: pd.DataFrame,
    action_column: str = "next_best_offer",
    reward_column: str = "valor_esperado_contacto",
    epsilon: float = 0.10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Construye una politica inicial de epsilon-greedy y Thompson Sampling."""
    if action_column not in scorecard.columns or reward_column not in scorecard.columns:
        empty = pd.DataFrame([{"status": "not_available"}])
        return {
            "policy_table": empty,
            "summary": empty,
            "interpretation": "No hubo suficientes columnas para construir la politica bandit.",
        }

    working = scorecard[[action_column, reward_column]].copy()
    working[action_column] = working[action_column].astype(str).fillna("sin_accion")
    working[reward_column] = pd.to_numeric(working[reward_column], errors="coerce").fillna(0.0)
    policy_table = (
        working.groupby(action_column)
        .agg(
            n=(reward_column, "size"),
            mean_reward=(reward_column, "mean"),
            median_reward=(reward_column, "median"),
            positive_reward_rate=(reward_column, lambda values: float((pd.Series(values) > 0).mean())),
        )
        .reset_index()
    )
    # alpha/beta sintetizan evidencia positiva y negativa para dejar una priorizacion bayesiana ligera por brazo.
    positive_counts = (policy_table["positive_reward_rate"] * policy_table["n"]).round().astype(int)
    policy_table["alpha"] = 1 + positive_counts
    policy_table["beta"] = 1 + policy_table["n"] - positive_counts
    policy_table["thompson_posterior_mean"] = policy_table["alpha"] / (policy_table["alpha"] + policy_table["beta"])
    reward_min = float(policy_table["mean_reward"].min())
    reward_range = float(policy_table["mean_reward"].max() - reward_min)
    if reward_range == 0:
        normalized_reward = np.ones(len(policy_table))
    else:
        normalized_reward = (policy_table["mean_reward"] - reward_min) / reward_range
    # epsilon controla el porcentaje de exploracion y el score Thompson prioriza brazos con mejor media posterior y recompensa observada.
    policy_table["epsilon_greedy_score"] = ((1 - epsilon) * normalized_reward + (epsilon / max(len(policy_table), 1))).round(4)
    policy_table["thompson_score"] = (policy_table["thompson_posterior_mean"] * np.maximum(normalized_reward, 0.05)).round(4)
    policy_table = policy_table.sort_values(["thompson_score", "mean_reward"], ascending=[False, False]).reset_index(drop=True)
    summary = pd.DataFrame(
        [{
            "recommended_arm": str(policy_table.iloc[0][action_column]),
            "epsilon": round(float(epsilon), 4),
            "n_arms": int(len(policy_table)),
            "max_thompson_score": round(float(policy_table.iloc[0]["thompson_score"]), 4),
        }]
    )
    interpretation = (
        "La politica bandit v3 no sustituye evidencia causal historica, pero deja lista una regla de exploracion-explotacion para operar cuando empiecen a entrar recompensas reales por accion."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "policy_table": policy_table,
        "summary": summary,
        "interpretation": interpretation,
    }


def build_deep_learning_governance_report(
    deep_learning_enabled: bool,
    require_dropout: bool,
    require_batch_norm: bool,
    require_early_stopping: bool,
    require_bayesian_optimization: bool,
    require_mc_dropout: bool,
    active_algorithms: Sequence[str],
    verbose: bool = True,
) -> dict[str, Any]:
    """Resume que controles DL estan implementados y cuales quedan como brecha."""
    normalized_algorithms = {_normalize_algorithm_name(algorithm) for algorithm in active_algorithms}
    mlp_active = any(algorithm in {"mlp", "neural_network"} for algorithm in normalized_algorithms)
    lstm_active = "lstm" in normalized_algorithms
    early_stopping_implemented = mlp_active or lstm_active
    dropout_implemented = lstm_active
    # Esta tabla es deliberadamente una compuerta de gobierno: marca brechas aunque la stack tabular actual no implemente aun cada control DL.
    checklist = pd.DataFrame(
        [
            {
                "control": "early_stopping",
                "required": require_early_stopping,
                "implemented": early_stopping_implemented,
                "status": "implemented" if early_stopping_implemented and require_early_stopping else "gap" if require_early_stopping else "optional",
            },
            {
                "control": "dropout",
                "required": require_dropout,
                "implemented": dropout_implemented,
                "status": "implemented" if dropout_implemented and require_dropout else "gap" if require_dropout else "optional",
            },
            {
                "control": "batch_norm",
                "required": require_batch_norm,
                "implemented": False,
                "status": "gap" if require_batch_norm else "optional",
            },
            {
                "control": "bayesian_hpo",
                "required": require_bayesian_optimization,
                "implemented": False,
                "status": "gap" if require_bayesian_optimization else "optional",
            },
            {
                "control": "mc_dropout_in_production",
                "required": require_mc_dropout,
                "implemented": False,
                "status": "gap" if require_mc_dropout else "optional",
            },
        ]
    )
    if not deep_learning_enabled or not (mlp_active or lstm_active):
        checklist["status"] = "not_applicable"
    required_gap_count = int((checklist["status"] == "gap").sum()) if deep_learning_enabled and (mlp_active or lstm_active) else 0
    interpretation = (
        "No se activaron modelos deep learning en la corrida actual; la compuerta de gobierno queda como no aplicable."
        if not (mlp_active or lstm_active)
        else "La v4 deja una compuerta explicita para deep learning: MLP cubre early stopping en la stack tabular y LSTM eleva el nivel de control porque exige dropout y monitoreo adicional antes de promover secuencias a produccion."
    )
    _emit_interpretation(interpretation, verbose)
    return {
        "checklist": checklist,
        "required_gap_count": required_gap_count,
        "interpretation": interpretation,
    }


# Alias en espanol y superficie publica para integrarse con el resto del repositorio.
auditar_dataset = audit_dataset
auditar_faltantes = audit_missingness_mechanism
auditar_dispersion_estructural = audit_structural_dispersion
comparar_transformaciones_potencia = compare_power_transformations
comprobar_homocedasticidad = check_variance_homogeneity
comprobar_normalidad = check_normality
construir_pipeline_preprocesado = build_preprocessing_pipeline
detectar_paradoja_simpson = detect_simpsons_paradox
correlacion_analitica = analyze_correlation
entrenar_modelo_supervisado = train_supervised_model
evaluar_calibracion_probabilidades = evaluate_probability_calibration
evaluar_drift_dataset = evaluate_dataset_drift
ejecutar_rfe = run_rfe_feature_selection
ejecutar_multiverse_analysis = run_multiverse_analysis
generar_diagnostico_modelo = plot_model_diagnostics
generar_diagnostico_ols = plot_ols_influence_diagnostics
graficar_dispersion_estructural = plot_structural_dispersion_diagnostics
graficar_diagnostico_qq = plot_qq_diagnostic
graficar_distribuciones_grupos = plot_group_distributions
graficar_faltantes_heatmap = plot_missingness_heatmap
graficar_importancia_variables = plot_feature_importance
graficar_calibracion_probabilidades = plot_probability_calibration
graficar_transformaciones_potencia = plot_power_transformations
imputar_valores_faltantes = impute_missing_values
manejar_outliers = handle_outliers
modelo_ols_inferencial = fit_ols_inference
monitorizar_salud_pipeline = report_pipeline_health
revisar_vif = calculate_vif
test_comparacion_grupos = compare_groups
entrenar_modelo_evento_competitivo = train_competitive_event_model
evaluar_predicciones_evento_competitivo = evaluate_competitive_event_predictions
evaluar_tickets_evento_competitivo = evaluate_competitive_event_tickets
graficar_diagnostico_evento_competitivo = plot_competitive_event_diagnostics
normalizar_probabilidades_evento_competitivo = normalize_competitive_event_probabilities
construir_tablero_predicciones_evento_competitivo = build_competitive_event_prediction_board
diferenciar_fraccionalmente = fractional_difference
estudiar_parsimonia_logistica = run_logistic_parsimony_study
validar_temporalmente_con_purga = run_purged_temporal_validation
estimar_intervalos_bootstrap = run_bootstrap_prediction_intervals
auditar_equidad = run_fairness_audit
medir_brecha_consenso = build_consensus_gap_report
construir_politica_bandit = build_bandit_policy
auditar_gobernanza_deep_learning = build_deep_learning_governance_report


__all__ = [
    "analyze_correlation",
    "audit_structural_dispersion",
    "audit_dataset",
    "audit_missingness_mechanism",
    "auditar_dispersion_estructural",
    "auditar_dataset",
    "auditar_faltantes",
    "build_preprocessing_pipeline",
    "detect_simpsons_paradox",
    "detectar_paradoja_simpson",
    "evaluate_dataset_drift",
    "evaluate_probability_calibration",
    "calculate_vif",
    "check_normality",
    "check_variance_homogeneity",
    "comparar_transformaciones_potencia",
    "compare_groups",
    "compare_power_transformations",
    "comprobar_homocedasticidad",
    "comprobar_normalidad",
    "construir_tablero_predicciones_evento_competitivo",
    "construir_politica_bandit",
    "construir_pipeline_preprocesado",
    "correlacion_analitica",
    "diferenciar_fraccionalmente",
    "ejecutar_multiverse_analysis",
    "ejecutar_rfe",
    "entrenar_modelo_supervisado",
    "entrenar_modelo_evento_competitivo",
    "estimar_intervalos_bootstrap",
    "estudiar_parsimonia_logistica",
    "build_competitive_event_prediction_board",
    "build_bandit_policy",
    "build_consensus_gap_report",
    "build_deep_learning_governance_report",
    "evaluate_competitive_event_predictions",
    "evaluate_competitive_event_tickets",
    "fractional_difference",
    "evaluar_predicciones_evento_competitivo",
    "evaluar_tickets_evento_competitivo",
    "fit_ols_inference",
    "get_universal_methodology_reference",
    "graficar_calibracion_probabilidades",
    "generar_diagnostico_modelo",
    "generar_diagnostico_ols",
    "graficar_dispersion_estructural",
    "graficar_diagnostico_evento_competitivo",
    "graficar_diagnostico_qq",
    "graficar_distribuciones_grupos",
    "graficar_faltantes_heatmap",
    "graficar_importancia_variables",
    "graficar_transformaciones_potencia",
    "handle_outliers",
    "impute_missing_values",
    "imputar_valores_faltantes",
    "manejar_outliers",
    "modelo_ols_inferencial",
    "monitorizar_salud_pipeline",
    "medir_brecha_consenso",
    "plot_ols_influence_diagnostics",
    "plot_probability_calibration",
    "normalize_competitive_event_probabilities",
    "normalizar_probabilidades_evento_competitivo",
    "plot_feature_importance",
    "plot_competitive_event_diagnostics",
    "plot_group_distributions",
    "plot_missingness_heatmap",
    "plot_model_diagnostics",
    "plot_structural_dispersion_diagnostics",
    "plot_power_transformations",
    "plot_qq_diagnostic",
    "report_pipeline_health",
    "revisar_vif",
    "run_bootstrap_prediction_intervals",
    "run_fairness_audit",
    "run_logistic_parsimony_study",
    "run_multiverse_analysis",
    "run_purged_temporal_validation",
    "run_rfe_feature_selection",
    "test_comparacion_grupos",
    "train_supervised_model",
    "validar_temporalmente_con_purga",
    "auditar_equidad",
    "auditar_gobernanza_deep_learning",
]