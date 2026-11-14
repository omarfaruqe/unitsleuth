"""unitsleuth: detect mixed measurement units in numeric columns."""

from unitsleuth import catalog
from unitsleuth.catalog import Conversion
from unitsleuth.detect import UnivariateFit, detect, fit_conversion
from unitsleuth.multivariate import RowFit, detect_rows, repair

__version__ = "0.1.2"

__all__ = [
    "Conversion",
    "RowFit",
    "UnivariateFit",
    "catalog",
    "detect",
    "detect_rows",
    "fit_conversion",
    "repair",
]
