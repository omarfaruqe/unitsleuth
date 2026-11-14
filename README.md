# unitsleuth

Detect mixed measurement units in an unlabeled numeric column, from the numbers alone.

A height column where most rows are centimetres and a few are inches, or a
temperature column that mixes °C and °F, looks fine to a schema check. unitsleuth
fits a mixture model per known unit conversion and tells you which rows were
probably recorded in the other unit.

This is a Python implementation of the mixture-model approach of Di Zio, Guarnera
and Luzi (2005, *Survey Methodology*), extended with affine conversions
(e.g. °C to °F) and a catalog of common unit pairs. It is not a new method.

## Quick start

```python
import numpy as np
import unitsleuth as us

heights = np.concatenate([np.random.normal(170, 9, 900), np.random.normal(67, 3.5, 100)])
best = us.detect(heights, quantity="length")[0]
print(best.label, best.weight, best.detected)
rows = best.converted  # boolean mask of rows likely in the other unit
```

### Whole rows in the wrong unit

When a record was entered in one unit system, every column in that row is
affected. `detect_rows` shares one flag per row across columns, and `repair`
harmonises the data. `df` can be a pandas or Polars DataFrame (via
[Narwhals](https://narwhals-dev.github.io/narwhals/)) or a dict of arrays:

```python
cat = us.catalog
fit = us.detect_rows(df, {"height": cat.get("cm", "in"), "weight": cat.get("kg", "lb")})
print(fit.summary())
fixed = us.repair(df, fit)  # same dataframe type, columns corrected
```

### Command line

```
unitsleuth data.csv --col height:cm,in --col weight:kg,lb --out data_fixed.csv
```

Prints the report and, with `--out`, writes a harmonised CSV. The exit code is 1
when mixed units are detected, 0 when not and 2 on usage errors, so it can gate a
data pipeline.

## Status

Alpha. Univariate and row-consistent multivariate detection and repair work;
dataframe integration (pandas, Polars) and a CLI are planned. Detection needs the units to be
separated relative to the spread of the data, so it cannot distinguish them when
natural variation is wide compared with the conversion factor.

## Development

```
uv sync
uv run pytest
```
