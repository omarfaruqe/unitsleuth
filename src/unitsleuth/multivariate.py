"""Row-consistent detection across several columns, plus repair.

Each row carries one latent flag: either every column is in its base unit, or
every column was recorded in the converted unit (for example a whole record
entered in imperial units). The flag is shared by all observed columns of the
row, so a row with a clear height gives evidence about its ambiguous weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unitsleuth._frame import columns_of, is_frame, replace_columns
from unitsleuth.catalog import Conversion
from unitsleuth.detect import Scale, _normal_logpdf

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Target = Literal["majority", "base", "converted"]


@dataclass(frozen=True)
class RowFit:
    """Joint fit of one shared row flag over several columns."""

    columns: tuple[str, ...]
    conversions: dict[str, Conversion]
    scales: dict[str, Scale]
    weight: float
    mu: dict[str, float]
    sigma: dict[str, float]
    loglik: float
    loglik_null: float
    delta_bic: float
    detected: bool
    posterior: FloatArray

    @property
    def converted(self) -> BoolArray:
        """Rows whose posterior probability of being converted exceeds 0.5."""
        with np.errstate(invalid="ignore"):
            return np.asarray(self.posterior > 0.5, dtype=np.bool_)

    def summary(self) -> str:
        """Plain-text report of the fit."""
        n_conv = int(self.converted.sum())
        n_obs = int(np.isfinite(self.posterior).sum())
        lines = [
            f"unitsleuth row-consistent fit over {len(self.columns)} column(s)",
            f"  rows analysed: {n_obs}",
            f"  rows flagged as converted: {n_conv} "
            f"({self.weight:.1%} estimated share)",
            f"  evidence (delta BIC vs no mixing): {self.delta_bic:.1f} "
            f"-> {'mixed units detected' if self.detected else 'no mixing detected'}",
        ]
        for col in self.columns:
            c = self.conversions[col]
            lines.append(
                f"  {col}: {c.name} on {self.scales[col]} scale, "
                f"mu={self.mu[col]:.4g}, sigma={self.sigma[col]:.4g}"
            )
        return "\n".join(lines)


def _to_arrays(data: Mapping[str, ArrayLike]) -> dict[str, FloatArray]:
    out: dict[str, FloatArray] = {
        k: np.asarray(v, dtype=np.float64).ravel() for k, v in data.items()
    }
    lengths = {a.size for a in out.values()}
    if len(lengths) > 1:
        raise ValueError("all columns must have the same length")
    return out


def detect_rows(
    data: Any,
    conversions: Mapping[str, Conversion],
    *,
    min_delta_bic: float = 10.0,
    min_weight: float = 0.005,
    min_n: int = 10,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> RowFit:
    """Fit a shared row flag over the columns named in ``conversions``.

    A flagged row is one where every observed column is ``conversion.forward`` of
    its base-unit value. The share of flagged rows is free, so the flagged unit may
    be the majority.
    """
    cols = tuple(conversions)
    if not cols:
        raise ValueError("conversions is empty")
    missing = [c for c in cols if isinstance(data, Mapping) and c not in data]
    if missing:
        raise KeyError(f"columns not in data: {missing}")
    arrays = _to_arrays(columns_of(data, cols))
    raw = np.column_stack([arrays[c] for c in cols])
    mask = np.isfinite(raw)
    row_obs = mask.any(axis=1)
    if int(row_obs.sum()) < min_n:
        raise ValueError(f"need at least {min_n} rows with a finite value")

    t_cols: list[FloatArray] = []
    y_cols: list[FloatArray] = []
    jac = np.zeros(len(cols))
    scales: dict[str, Scale] = {}
    for j, c in enumerate(cols):
        conv = conversions[c]
        x = raw[:, j]
        obs = mask[:, j]
        positive = bool(np.all(x[obs] > 0)) if obs.any() else False
        if conv.is_multiplicative and conv.scale > 0 and positive:
            scales[c] = "log"
            t = np.where(obs, np.log(np.where(obs, x, 1.0)), 0.0)
            y = t - np.log(conv.scale)
        else:
            scales[c] = "linear"
            t = np.where(obs, x, 0.0)
            y = (t - conv.offset) / conv.scale
            jac[j] = -np.log(abs(conv.scale))
        if obs.sum() < 2 or float(np.std(t[obs])) == 0.0:
            raise ValueError(f"column {c!r} is constant or has too few values")
        t_cols.append(t)
        y_cols.append(y)
    t_mat = np.column_stack(t_cols)
    y_mat = np.column_stack(y_cols)
    m = mask.astype(np.float64)

    n_obs_col = m.sum(axis=0)
    sd = np.array([np.std(t_mat[mask[:, j], j]) for j in range(len(cols))])
    floor = 1e-3 * sd
    med = np.array([np.median(t_mat[mask[:, j], j]) for j in range(len(cols))])
    mad = np.array(
        [
            1.4826 * np.median(np.abs(t_mat[mask[:, j], j] - med[j]))
            for j in range(len(cols))
        ]
    )
    sig_init = np.maximum(mad / 2.0, floor)

    def logpost(
        w: float, mu: FloatArray, sig: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        a = np.log1p(-w) + (_normal_logpdf(t_mat, mu, sig) * m).sum(axis=1)
        b = np.log(w) + ((_normal_logpdf(y_mat, mu, sig) + jac) * m).sum(axis=1)
        return a, b

    best: tuple[float, FloatArray, FloatArray, float] | None = None
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        mu_start = np.array(
            [np.quantile(t_mat[mask[:, j], j], q) for j in range(len(cols))]
        )
        for w0 in (0.05, 0.2, 0.5, 0.8):
            mu, sig, w = mu_start.copy(), sig_init.copy(), w0
            prev, ll = -np.inf, -np.inf
            for _ in range(max_iter):
                a, b = logpost(w, mu, sig)
                mx = np.maximum(a, b)
                lse = mx + np.log(np.exp(a - mx) + np.exp(b - mx))
                ll = float(lse[row_obs].sum())
                r = np.exp(b - lse)
                w = float(np.clip(r[row_obs].mean(), 1e-6, 1 - 1e-6))
                rc = r[:, None]
                mu = np.sum(m * ((1 - rc) * t_mat + rc * y_mat), axis=0) / n_obs_col
                var = (
                    np.sum(
                        m * ((1 - rc) * (t_mat - mu) ** 2 + rc * (y_mat - mu) ** 2),
                        axis=0,
                    )
                    / n_obs_col
                )
                sig = np.maximum(np.sqrt(var), floor)
                if abs(ll - prev) < tol * (1.0 + abs(ll)):
                    break
                prev = ll
            if best is None or ll > best[3]:
                best = (w, mu, sig, ll)
    assert best is not None
    w, mu, sig, ll = best

    ll0 = 0.0
    for j in range(len(cols)):
        tj = t_mat[mask[:, j], j]
        ll0 += float(_normal_logpdf(tj, float(tj.mean()), float(tj.std())).sum())
    n = int(row_obs.sum())
    j_cols = len(cols)
    delta_bic = (-2.0 * ll0 + 2 * j_cols * np.log(n)) - (
        -2.0 * ll + (2 * j_cols + 1) * np.log(n)
    )

    a, b = logpost(w, mu, sig)
    post = 1.0 / (1.0 + np.exp(np.clip(a - b, -700, 700)))
    post[~row_obs] = np.nan
    return RowFit(
        columns=cols,
        conversions=dict(conversions),
        scales=scales,
        weight=w,
        mu={c: float(mu[j]) for j, c in enumerate(cols)},
        sigma={c: float(sig[j]) for j, c in enumerate(cols)},
        loglik=ll,
        loglik_null=ll0,
        delta_bic=float(delta_bic),
        detected=bool(delta_bic >= min_delta_bic and w >= min_weight),
        posterior=post,
    )


def repair(
    data: Any,
    fit: RowFit,
    *,
    threshold: float = 0.5,
    target: Target = "majority",
) -> Any:
    """Return copies of the fitted columns harmonised to a single unit.

    ``target="base"`` converts flagged rows back to the base unit,
    ``"converted"`` converts unflagged rows forward, and ``"majority"`` (default)
    picks whichever unit most rows are already in. Rows with an undefined
    posterior are left untouched. For a dataframe input the result is a
    dataframe of the same type with the fitted columns replaced; for a mapping
    it is a dict of arrays.
    """
    arrays = _to_arrays(columns_of(data, fit.columns))
    post = fit.posterior
    with np.errstate(invalid="ignore"):
        flagged = post > threshold
        unflagged = post <= threshold
    if target == "majority":
        target = "base" if int(unflagged.sum()) >= int(flagged.sum()) else "converted"
    out: dict[str, FloatArray] = {}
    for c in fit.columns:
        conv = fit.conversions[c]
        x = arrays[c].copy()
        if target == "base":
            x[flagged] = conv.inverse(x[flagged])
        else:
            x[unflagged] = conv.forward(x[unflagged])
        out[c] = x
    return replace_columns(data, out) if is_frame(data) else out
