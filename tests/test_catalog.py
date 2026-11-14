import numpy as np
import pytest

from unitsleuth import catalog
from unitsleuth.catalog import Conversion


@pytest.mark.parametrize("conv", catalog.conversions(), ids=lambda c: c.name)
def test_round_trip(conv: Conversion) -> None:
    x = np.array([0.5, 1.0, 37.0, 250.0])
    np.testing.assert_allclose(conv.inverse(conv.forward(x)), x, rtol=1e-12)


def test_known_values() -> None:
    np.testing.assert_allclose(catalog.get("degC", "degF").forward(100.0), 212.0)
    np.testing.assert_allclose(catalog.get("cm", "in").forward(2.54), 1.0)
    np.testing.assert_allclose(catalog.get("kg", "lb").forward(1.0), 2.20462262)
    np.testing.assert_allclose(catalog.get("degC", "K").forward(0.0), 273.15)


def test_multiplicative_flag() -> None:
    assert catalog.get("kg", "lb").is_multiplicative
    assert not catalog.get("degC", "degF").is_multiplicative


def test_invalid_conversion_rejected() -> None:
    with pytest.raises(ValueError):
        Conversion("x", "a", "b", 0.0)
    with pytest.raises(ValueError):
        Conversion("x", "a", "b", float("nan"))


def test_lookup_errors() -> None:
    with pytest.raises(KeyError):
        catalog.get("kg", "furlong")
    with pytest.raises(KeyError):
        catalog.conversions("flavour")


def test_quantity_filter_and_names_unique() -> None:
    assert {c.quantity for c in catalog.conversions("mass")} == {"mass"}
    names = [c.name for c in catalog.conversions()]
    assert len(names) == len(set(names))
    assert "temperature" in catalog.quantities()
