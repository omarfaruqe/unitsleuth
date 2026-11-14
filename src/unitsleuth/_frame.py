"""Dataframe adapters built on Narwhals (pandas, Polars, PyArrow, ...)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import narwhals.stable.v1 as nw
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def is_frame(obj: Any) -> bool:
    """True for native eager dataframes that Narwhals understands."""
    if isinstance(obj, Mapping):
        return False
    try:
        nw.from_native(obj, eager_only=True)
    except TypeError:
        return False
    return True


def columns_of(data: Any, names: list[str] | tuple[str, ...]) -> dict[str, ArrayLike]:
    """Extract named columns as float arrays from a mapping or a dataframe."""
    if isinstance(data, Mapping):
        return {n: data[n] for n in names}
    frame = nw.from_native(data, eager_only=True)
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise KeyError(f"columns not in data: {missing}")
    out: dict[str, ArrayLike] = {}
    for n in names:
        s = frame[n].cast(nw.Float64)
        out[n] = np.array(s.to_list(), dtype=np.float64)
        # to_list maps nulls to None; np.array(..., float64) turns None into nan
    return out


def replace_columns(data: Any, cols: Mapping[str, FloatArray]) -> Any:
    """Return a native dataframe of the same type with ``cols`` replaced."""
    frame = nw.from_native(data, eager_only=True)
    ns = nw.get_native_namespace(frame)
    new = [
        nw.new_series(name, arr.tolist(), nw.Float64, native_namespace=ns)
        for name, arr in cols.items()
    ]
    return frame.with_columns(*new).to_native()
