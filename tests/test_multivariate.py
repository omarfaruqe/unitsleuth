import numpy as np
import pytest

from unitsleuth import catalog, detect_rows, repair
from unitsleuth.detect import fit_conversion

H = catalog.get("cm", "in")
W = catalog.get("kg", "lb")


def _people(seed: int, n: int = 1500, frac: float = 0.15):
    rng = np.random.default_rng(seed)
    h = rng.normal(170.0, 9.0, n)
    w = rng.normal(75.0, 16.0, n)
    flag = rng.random(n) < frac
    return (
        {
            "height": np.where(flag, H.forward(h), h),
            "weight": np.where(flag, W.forward(w), w),
        },
        flag,
        h,
        w,
    )


def test_joint_detects_and_beats_single_column() -> None:
    data, flag, _, _ = _people(1)
    fit = detect_rows(data, {"height": H, "weight": W})
    assert fit.detected
    assert abs(fit.weight - 0.15) < 0.03
    joint_acc = np.mean(fit.converted == flag)
    weight_only = fit_conversion(data["weight"], W, direction="forward")
    assert joint_acc > 0.99
    assert joint_acc >= np.mean(weight_only.converted == flag)


def test_clean_data_not_flagged() -> None:
    for seed in range(8):
        data, _, h, w = _people(seed, frac=0.0)
        fit = detect_rows(data, {"height": H, "weight": W})
        assert not fit.detected


def test_missing_values() -> None:
    data, flag, _, _ = _people(2)
    data["weight"][::7] = np.nan
    fit = detect_rows(data, {"height": H, "weight": W})
    assert fit.detected
    assert np.isfinite(fit.posterior).all()
    assert np.mean(fit.converted == flag) > 0.99


def test_repair_restores_base_unit() -> None:
    data, flag, h, w = _people(3)
    fit = detect_rows(data, {"height": H, "weight": W})
    fixed = repair(data, fit)
    ok = fit.converted == flag
    np.testing.assert_allclose(fixed["height"][ok], h[ok], rtol=1e-9)
    np.testing.assert_allclose(fixed["weight"][ok], w[ok], rtol=1e-9)
    assert abs(np.mean(fixed["height"]) - 170.0) < 1.5
    assert data["height"][flag].max() < 100  # input untouched


def test_majority_target_when_flagged_is_majority() -> None:
    data, flag, h, _ = _people(4, frac=0.8)
    fit = detect_rows(data, {"height": H, "weight": W})
    assert fit.detected and fit.weight > 0.5
    fixed = repair(data, fit)
    assert abs(np.mean(fixed["height"]) - np.mean(H.forward(h))) < 0.5


def test_summary_and_validation() -> None:
    data, _, _, _ = _people(5)
    fit = detect_rows(data, {"height": H})
    assert "mixed units detected" in fit.summary()
    with pytest.raises(KeyError):
        detect_rows(data, {"nope": H})
    with pytest.raises(ValueError):
        detect_rows({"a": [1.0, 2.0], "b": [1.0]}, {"a": H, "b": W})
    with pytest.raises(ValueError):
        detect_rows(data, {})
