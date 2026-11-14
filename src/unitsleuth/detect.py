"""Univariate detection of mixed units in a single numeric column.

Model (after Di Zio, Guarnera & Luzi, 2005): true values follow a normal
distribution on a working scale (log for multiplicative conversions of positive
data, raw otherwise). A fraction ``weight`` of rows was recorded in a different
unit, i.e. passed through a known conversion. The mixture is fitted by EM and
compared with the single-unit model through BIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unitsleuth import catalog
from unitsleuth.catalog import Conversion

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Direction = Literal["forward", "inverse"]
Scale = Literal["log", "linear"]

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class UnivariateFit:
    """Result of testing one conversion against one column."""

    conversion: Conversion
    direction: Direction
    scale: Scale
    weight: float
    mu: float
    sigma: float
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

    @property
    def label(self) -> str:
        """Human-readable description of the hypothesis, e.g. ``'cm->in'``."""
        if self.direction == "forward":
            return self.conversion.name
        return f"{self.conversion.to_unit}->{self.conversion.from_unit}"


def _normal_logpdf(
    t: FloatArray, mu: FloatArray | float, sigma: FloatArray | float
) -> FloatArray:
    z = (t - mu) / sigma
    return np.asarray(-0.5 * z * z - np.log(sigma) - 0.5 * _LOG_2PI, dtype=np.float64)


def _fit_mixture(
    t: FloatArray, shift: float, mult: float, jac: float, max_iter: int, tol: float
) -> tuple[float, float, float, float]:
    """EM for tied-parameter two-component model.

    Converted rows are pulled back with ``y = (t - shift) / mult`` and then share
    ``mu`` and ``sigma`` with the base rows. ``jac`` is the log-density Jacobian
    of the converted component. Returns ``(weight, mu, sigma, loglik)``.
    """
    n = t.size
    y = (t - shift) / mult
    sd_all = float(np.std(t))
    floor = 1e-3 * sd_all if sd_all > 0 else 1e-9
    med = float(np.median(t))
    mad = 1.4826 * float(np.median(np.abs(t - med)))
    sig0 = max(mad / 2.0, floor)
    best: tuple[float, float, float, float] | None = None
    for mu0 in np.quantile(t, [0.25, 0.5, 0.75]):
        for w0 in (0.05, 0.2, 0.4):
            mu, sigma, w = float(mu0), sig0, w0
            prev = -np.inf
            ll = -np.inf
            for _ in range(max_iter):
                a = np.log1p(-w) + _normal_logpdf(t, mu, sigma)
                b = np.log(w) + _normal_logpdf(y, mu, sigma) + jac
                m = np.maximum(a, b)
                lse = m + np.log(np.exp(a - m) + np.exp(b - m))
                ll = float(lse.sum())
                r = np.exp(b - lse)
                w = float(np.clip(r.mean(), 1e-6, 1 - 1e-6))
                mu = float(np.sum((1 - r) * t + r * y) / n)
                var = float(np.sum((1 - r) * (t - mu) ** 2 + r * (y - mu) ** 2) / n)
                sigma = max(np.sqrt(var), floor)
                if abs(ll - prev) < tol * (1.0 + abs(ll)):
                    break
                prev = ll
            if best is None or ll > best[3]:
                best = (w, mu, sigma, ll)
    assert best is not None
    return best


def _params(conv: Conversion, direction: Direction) -> tuple[float, float]:
    if direction == "forward":
        return conv.scale, conv.offset
    return 1.0 / conv.scale, -conv.offset / conv.scale


def fit_conversion(
    x: ArrayLike,
    conversion: Conversion,
    *,
    direction: Direction | None = None,
    scale: Scale | None = None,
    min_delta_bic: float = 10.0,
    min_weight: float = 0.005,
    min_n: int = 10,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> UnivariateFit:
    """Test whether a share of ``x`` was recorded in a different unit.

    ``direction=None`` fits both ``from->to`` and ``to->from`` and keeps the better
    one. ``scale=None`` uses the log scale for multiplicative conversions of
    strictly positive data and the raw scale otherwise.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    ok = np.isfinite(arr)
    xs = arr[ok]
    if xs.size < min_n:
        raise ValueError(f"need at least {min_n} finite values, got {xs.size}")
    if direction is None:
        fits = [
            fit_conversion(
                arr,
                conversion,
                direction=d,
                scale=scale,
                min_delta_bic=min_delta_bic,
                min_weight=min_weight,
                min_n=min_n,
                max_iter=max_iter,
                tol=tol,
            )
            for d in ("forward", "inverse")
        ]
        return max(fits, key=lambda f: f.loglik)

    a, c = _params(conversion, direction)
    can_log = c == 0.0 and a > 0.0 and bool(np.all(xs > 0))
    if scale is None:
        scale = "log" if can_log else "linear"
    if scale == "log" and not can_log:
        raise ValueError(
            "log scale needs a positive multiplicative conversion and x > 0"
        )

    if scale == "log":
        t = np.log(xs)
        shift, mult, jac = float(np.log(a)), 1.0, 0.0
    else:
        t = xs
        shift, mult, jac = c, a, -float(np.log(abs(a)))
    if float(np.std(t)) == 0.0:
        post = np.full(arr.shape, np.nan)
        post[ok] = 0.0
        return UnivariateFit(
            conversion,
            direction,
            scale,
            0.0,
            float(t[0]),
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            post,
        )

    w, mu, sigma, ll = _fit_mixture(t, shift, mult, jac, max_iter, tol)

    mu0 = float(np.mean(t))
    sig0 = float(np.std(t))
    ll0 = float(_normal_logpdf(t, mu0, sig0).sum())
    n = t.size
    delta_bic = (-2.0 * ll0 + 2 * np.log(n)) - (-2.0 * ll + 3 * np.log(n))

    la = np.log1p(-w) + _normal_logpdf(t, mu, sigma)
    lb = np.log(w) + _normal_logpdf((t - shift) / mult, mu, sigma) + jac
    r = 1.0 / (1.0 + np.exp(np.clip(la - lb, -700, 700)))
    post = np.full(arr.shape, np.nan)
    post[ok] = r
    return UnivariateFit(
        conversion=conversion,
        direction=direction,
        scale=scale,
        weight=w,
        mu=mu,
        sigma=sigma,
        loglik=ll,
        loglik_null=ll0,
        delta_bic=float(delta_bic),
        detected=bool(delta_bic >= min_delta_bic and w >= min_weight),
        posterior=post,
    )


def detect(
    x: ArrayLike,
    conversions: list[Conversion] | tuple[Conversion, ...] | None = None,
    *,
    quantity: str | None = None,
    **kwargs: float | int,
) -> list[UnivariateFit]:
    """Test several conversions and return fits sorted by evidence (best first).

    Pass ``conversions`` explicitly, or a catalog ``quantity`` such as ``"mass"``;
    by default every catalog conversion is tried. Extra keyword arguments go to
    :func:`fit_conversion`.
    """
    cands = (
        tuple(conversions) if conversions is not None else catalog.conversions(quantity)
    )
    out: list[UnivariateFit] = []
    for conv in cands:
        try:
            out.append(fit_conversion(x, conv, **kwargs))  # type: ignore[arg-type]
        except ValueError:
            continue
    return sorted(out, key=lambda f: f.delta_bic, reverse=True)
