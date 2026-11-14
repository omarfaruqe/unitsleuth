# unitsleuth usage guide

## Concepts

unitsleuth looks at the numbers in a column and asks whether a share of the rows
was recorded in a different unit than the rest. For each candidate conversion it
fits a mixture of two groups: rows in the base unit and rows that went through the
conversion. It compares this with a single-unit model and reports the evidence as
`delta_bic` (higher means stronger evidence of mixing; 10 or more counts as
detection by default).

Every row also gets a **posterior**: the probability that the row is in the
converted unit. `fit.converted` is the boolean mask of rows with posterior > 0.5.

## One column: `detect` and `fit_conversion`

```python
import numpy as np
import unitsleuth as us

heights = np.concatenate([np.random.normal(170, 9, 900), np.random.normal(67, 3.5, 100)])

fits = us.detect(heights, quantity="length")   # tries every length conversion
best = fits[0]                                  # sorted, strongest evidence first
print(best.label, best.weight, best.delta_bic, best.detected)
mask = best.converted                           # rows likely in the other unit
```

`detect` accepts `quantity=` (one of `us.catalog.quantities()`), an explicit list
of `Conversion` objects, or nothing (every catalog conversion). To test a single
conversion use `us.fit_conversion(x, conv)`. Both directions (`cm->in` and
`in->cm`) are tried, and `best.label` tells you which one fit. Missing values
(NaN) are ignored and their posterior is NaN.

Useful options: `min_delta_bic` (detection threshold, default 10), `min_weight`
(smallest share worth reporting, default 0.005), `scale` (`"log"` or `"linear"`;
chosen automatically).

## Several columns: `detect_rows` and `repair`

When a whole record was entered in one unit system, every column in the row is
affected. `detect_rows` shares one flag per row across columns, so a clear height
helps decide an ambiguous weight.

```python
cat = us.catalog
fit = us.detect_rows(df, {"height": cat.get("cm", "in"), "weight": cat.get("kg", "lb")})
print(fit.summary())

fixed = us.repair(df, fit)                       # same type as df
fixed = us.repair(df, fit, target="base")        # force the base unit (cm, kg)
fixed = us.repair(df, fit, threshold=0.9)        # repair only confident rows
```

`df` may be a pandas or Polars DataFrame, or a dict of arrays (a dict returns a
dict). `target` is `"majority"` (default: the unit most rows are already in),
`"base"` or `"converted"`. A flagged row means every observed column in it is in
the converted unit.

## The catalog and custom conversions

```python
us.catalog.quantities()            # ('length', 'mass', 'temperature', ...)
us.catalog.conversions("mass")     # all mass conversions
us.catalog.get("degC", "degF")     # one conversion

from unitsleuth import Conversion
mm_in = Conversion("length", "mm", "in", scale=1 / 25.4)       # x_in = x_mm / 25.4
c_f = Conversion("temperature", "degC", "degF", scale=1.8, offset=32.0)
```

A conversion is affine: `to = scale * from + offset`. Multiplicative conversions
use a log-scale model for positive data; conversions with an offset (temperature)
use the raw scale.

## Command line

Options: `--target {majority,base,converted}`, `--min-delta-bic N`. Exit code 1 means
mixed units were detected, 0 means none, 2 means a usage error.

## Reading the results

- `delta_bic >= 10` and a plausible `weight` (share of converted rows): treat the
  flag as real, then look at a few flagged rows yourself.
- `delta_bic` near zero or negative: no evidence of mixing.
- A posterior near 0.5 means the row is ambiguous; use `repair(..., threshold=0.9)`
  to change only confident rows.

## Limits

Detection works from the numbers alone, so it needs the units to be separated
relative to the natural spread of the data. A column of weights with a very wide
range can hide a kg/lb mix; a column of adult heights cannot. Row-consistent
detection helps when another column in the same rows is easier to separate. The
tool finds candidates; a human should confirm before data is overwritten.
