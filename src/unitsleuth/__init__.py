"""unitsleuth: detect mixed measurement units in numeric columns."""

from unitsleuth import catalog
from unitsleuth.catalog import Conversion
from unitsleuth.detect import UnivariateFit, detect, fit_conversion

__version__ = "0.1.0"

__all__ = ["Conversion", "UnivariateFit", "catalog", "detect", "fit_conversion"]
