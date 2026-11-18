"""Shared observation readiness rules for fitting and fold planning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mri.features.core import FeatureError


def observation_values(
    frame: pd.DataFrame, columns: tuple[str, str], *, allow_warmup: bool
) -> tuple[int, NDArray[np.float64]]:
    """Return the warm-up length and complete return/volatility observations.

    Only the leading missing-volatility prefix is warm-up. Infinities and
    negative volatility are invalid even there; once volatility is available,
    neither input may be missing. Dates and calendar coverage belong to callers.
    """
    if frame.empty or not frame.columns.is_unique or any(column not in frame for column in columns):
        raise FeatureError("Missing or duplicate observation columns")
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureError("Observation columns must be numeric") from exc
    invalid = "Incomplete, non-finite, or negative volatility observations"
    if np.isinf(values).any() or (values[:, 1] < 0).any():
        raise FeatureError(invalid)
    first = 0
    if allow_warmup:
        ready = np.flatnonzero(~np.isnan(values[:, 1]))
        if len(ready) == 0:
            raise FeatureError("No complete observations after warm-up")
        first = int(ready[0])
    complete = values[first:]
    if not np.isfinite(complete).all():
        raise FeatureError(invalid)
    return first, complete
