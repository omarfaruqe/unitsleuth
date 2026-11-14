import numpy as np
import pandas as pd
import polars as pl
import pytest

from unitsleuth import catalog, detect_rows, repair

H = catalog.get("cm", "in")
W = catalog.get("kg", "lb")


def _data(seed: int = 1, n: int = 1200, frac: float = 0.15):
    rng = np.random.default_rng(seed)
    h = rng.normal(170.0, 9.0, n)
    w = rng.normal(75.0, 16.0, n)
    flag = rng.random(n) < frac
    return (
        {
            "id": np.arange(n),
            "height": np.where(flag, H.forward(h), h),
            "weight": np.where(flag, W.forward(w), w),
        },
        flag,
        h,
    )


@pytest.mark.parametrize("make", [pd.DataFrame, pl.DataFrame])
def test_frames_match_mapping(make) -> None:
    data, flag, h = _data()
    convs = {"height": H, "weight": W}
    ref = detect_rows(data, convs)
    fit = detect_rows(make(data), convs)
    np.testing.assert_allclose(fit.posterior, ref.posterior, atol=1e-9)
    fixed = repair(make(data), fit)
    assert type(fixed) is make
    assert list(fixed.columns) == ["id", "height", "weight"]
    ok = fit.converted == flag
    got = np.asarray(
        fixed["height"].to_list() if make is pl.DataFrame else fixed["height"]
    )
    np.testing.assert_allclose(got[ok], h[ok], rtol=1e-9)
    assert np.asarray(
        fixed["id"].to_list() if make is pl.DataFrame else fixed["id"]
    ).tolist() == list(range(len(flag)))


def test_nulls_and_missing_column() -> None:
    data, _, _ = _data()
    df = pl.DataFrame(data).with_columns(
        pl.when(pl.col("id") % 9 == 0)
        .then(None)
        .otherwise(pl.col("weight"))
        .alias("weight")
    )
    fit = detect_rows(df, {"height": H, "weight": W})
    assert fit.detected
    with pytest.raises(KeyError):
        detect_rows(pd.DataFrame(data), {"nope": H})


def test_pandas_nan_and_int_columns() -> None:
    data, _, _ = _data()
    df = pd.DataFrame(data)
    df.loc[::11, "weight"] = np.nan
    assert detect_rows(df, {"height": H, "weight": W}).detected
