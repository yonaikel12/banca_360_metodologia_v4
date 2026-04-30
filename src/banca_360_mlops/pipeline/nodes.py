"""Nodos funcionales del pipeline industrializado.

Cada nodo encapsula una fase logica del caso y se ejecuta en hilo aparte para que
la orquestacion asincroma mantenga el notebook libre de logica operativa pesada.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from ..services.bank360_case import Bank360CaseService


async def build_context_node(service: Bank360CaseService) -> dict[str, Any]:
    return await asyncio.to_thread(service.build_context)


async def build_dataset_node(service: Bank360CaseService) -> dict[str, Any]:
    return await asyncio.to_thread(service.build_dataset)


async def benchmark_node(service: Bank360CaseService, df_bank: pd.DataFrame) -> dict[str, Any]:
    return await asyncio.to_thread(service.run_benchmark, df_bank)


async def bi_layer_node(
    service: Bank360CaseService,
    df_bank: pd.DataFrame,
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    return await asyncio.to_thread(service.run_bi_layer, df_bank, benchmark_result)


async def methodology_node(
    service: Bank360CaseService,
    df_bank: pd.DataFrame,
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    return await asyncio.to_thread(service.run_methodology_validation, df_bank, benchmark_result)


async def shap_node(service: Bank360CaseService, bi_result: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(service.run_shap_transparency, bi_result)


async def activation_node(
    service: Bank360CaseService,
    df_bank: pd.DataFrame,
    bi_result: dict[str, Any],
) -> dict[str, Any]:
    return await asyncio.to_thread(service.run_clv_activation, df_bank, bi_result)