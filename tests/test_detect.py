import numpy as np
import pytest

from unitsleuth import catalog, detect
from unitsleuth.detect import fit_conversion


def _mixed(rng: np.random.Generator, n: int, frac: float, conv, mu: float, sd: float):
    base = rng.normal(mu, sd, n)
    flag = rng.random(n) < frac
    x = np.where(flag, conv.forward(base), base)
    return x, flag


def test_heights_cm_in() -> None:
    rng = np.random.default_rng(1)
    conv = catalog.get("cm", "in")
    x, flag = _mixed(rng, 2000, 0.15, conv, 170.0, 9.0)
    fit = fit_conversion(x, conv)
    assert fit.detected and fit.direction == "forward" and fit.scale == "log"
    assert abs(fit.weight - 0.15) < 0.03
    assert np.mean(fit.converted == flag) > 0.99


def test_inverse_direction_majority_other_unit() -> None:
    rng = np.random.default_rng(2)
    conv = catalog.get("cm", "in")
    inches = rng.normal(67.0, 3.5, 1500)
    flag = rng.random(1500) < 0.2
    x = np.where(flag, conv.inverse(inches), inches)
    fit = fit_conversion(x, conv)
    assert fit.detected and fit.direction == "inverse"
    assert fit.label == "in->cm"
    assert np.mean(fit.converted == flag) > 0.99


def test_temperature_affine_uses_linear_scale() -> None:
    rng = np.random.default_rng(3)
    conv = catalog.get("degC", "degF")
    x, flag = _mixed(rng, 1500, 0.1, conv, 15.0, 6.0)
    fit = fit_conversion(x, conv)
    assert fit.detected and fit.scale == "linear"
    assert np.mean(fit.converted == flag) > 0.97


def test_clean_data_not_flagged() -> None:
    conv = catalog.get("kg", "lb")
    for seed in range(10):
        x = np.random.default_rng(seed).lognormal(np.log(70), 0.2, 1000)
        assert not fit_conversion(x, conv).detected


def test_nan_preserved_and_ignored() -> None:
    rng = np.random.default_rng(4)
    conv = catalog.get("cm", "in")
    x, _ = _mixed(rng, 800, 0.2, conv, 170.0, 9.0)
    x[::50] = np.nan
    fit = fit_conversion(x, conv)
    assert np.isnan(fit.posterior[::50]).all()
    assert np.isfinite(fit.posterior[1::50]).all()
    assert fit.detected


def test_too_few_values_and_constant() -> None:
    conv = catalog.get("cm", "in")
    with pytest.raises(ValueError):
        fit_conversion([1.0, 2.0], conv)
    fit = fit_conversion(np.full(50, 170.0), conv)
    assert not fit.detected


def test_detect_ranks_true_conversion_first() -> None:
    rng = np.random.default_rng(5)
    conv = catalog.get("kg", "lb")
    x, _ = _mixed(rng, 2000, 0.2, conv, 75.0, 12.0)
    fits = detect(x, quantity="mass")
    assert fits[0].label == "kg->lb"
    assert fits[0].detected
