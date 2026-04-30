"""Control centralizado de estocasticidad y semilla del proyecto."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Aplica la misma semilla a los generadores usados por el proyecto."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)