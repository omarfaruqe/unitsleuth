"""Catalog of unit conversions that unitsleuth can test for.

Every conversion is affine: ``x_to = scale * x_from + offset``. Multiplicative
conversions (``offset == 0``) cover most unit pairs; temperature needs the offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Conversion:
    """An affine map between two units of the same quantity."""

    quantity: str
    from_unit: str
    to_unit: str
    scale: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.scale) and np.isfinite(self.offset)):
            raise ValueError("scale and offset must be finite")
        if self.scale == 0.0:
            raise ValueError("scale must be non-zero")

    @property
    def name(self) -> str:
        """Short label such as ``'kg->lb'``."""
        return f"{self.from_unit}->{self.to_unit}"

    @property
    def is_multiplicative(self) -> bool:
        """True when the conversion has no additive offset."""
        return self.offset == 0.0

    def forward(self, x: ArrayLike) -> FloatArray:
        """Convert values expressed in ``from_unit`` to ``to_unit``."""
        arr = np.asarray(x, dtype=np.float64)
        return np.asarray(self.scale * arr + self.offset, dtype=np.float64)

    def inverse(self, y: ArrayLike) -> FloatArray:
        """Convert values expressed in ``to_unit`` back to ``from_unit``."""
        arr = np.asarray(y, dtype=np.float64)
        return np.asarray((arr - self.offset) / self.scale, dtype=np.float64)


def _c(
    quantity: str, from_unit: str, to_unit: str, scale: float, offset: float = 0.0
) -> Conversion:
    return Conversion(quantity, from_unit, to_unit, scale, offset)


_CATALOG: tuple[Conversion, ...] = (
    # length
    _c("length", "m", "ft", 3.280839895013123),
    _c("length", "cm", "in", 1.0 / 2.54),
    _c("length", "km", "mi", 0.621371192237334),
    _c("length", "m", "cm", 100.0),
    _c("length", "ft", "in", 12.0),
    # mass
    _c("mass", "kg", "lb", 2.2046226218487757),
    _c("mass", "kg", "g", 1000.0),
    _c("mass", "lb", "oz", 16.0),
    _c("mass", "g", "oz", 0.035273961949580414),
    # temperature
    _c("temperature", "degC", "degF", 1.8, 32.0),
    _c("temperature", "degC", "K", 1.0, 273.15),
    # volume
    _c("volume", "L", "mL", 1000.0),
    _c("volume", "L", "gal_us", 0.26417205235814845),
    # pressure
    _c("pressure", "kPa", "Pa", 1000.0),
    _c("pressure", "kPa", "psi", 0.14503773773375),
    _c("pressure", "mmHg", "kPa", 0.133322387415),
    # speed
    _c("speed", "km/h", "mph", 0.621371192237334),
    _c("speed", "km/h", "m/s", 1.0 / 3.6),
    # time
    _c("time", "h", "min", 60.0),
    _c("time", "min", "s", 60.0),
    # proportions
    _c("ratio", "fraction", "percent", 100.0),
    # generic power-of-ten scale errors (classic keying/recording mistakes)
    _c("generic", "x1", "x10", 10.0),
    _c("generic", "x1", "x100", 100.0),
    _c("generic", "x1", "x1000", 1000.0),
)


def conversions(quantity: str | None = None) -> tuple[Conversion, ...]:
    """Return the built-in conversions, optionally filtered by quantity."""
    if quantity is None:
        return _CATALOG
    selected = tuple(c for c in _CATALOG if c.quantity == quantity)
    if not selected:
        raise KeyError(f"unknown quantity {quantity!r}; known: {quantities()}")
    return selected


def quantities() -> tuple[str, ...]:
    """Return the quantities covered by the catalog, in catalog order."""
    return tuple(dict.fromkeys(c.quantity for c in _CATALOG))


def get(from_unit: str, to_unit: str) -> Conversion:
    """Look up a built-in conversion by its unit names."""
    for conv in _CATALOG:
        if conv.from_unit == from_unit and conv.to_unit == to_unit:
            return conv
    raise KeyError(f"no built-in conversion {from_unit}->{to_unit}")
