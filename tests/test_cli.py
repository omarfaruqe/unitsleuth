import csv

import numpy as np
import pytest

from unitsleuth import catalog
from unitsleuth.cli import main

H = catalog.get("cm", "in")
W = catalog.get("kg", "lb")


def _write(path, flag_frac=0.15, seed=1, n=800):
    rng = np.random.default_rng(seed)
    h, w = rng.normal(170, 9, n), rng.normal(75, 16, n)
    f = rng.random(n) < flag_frac
    hh, ww = np.where(f, H.forward(h), h), np.where(f, W.forward(w), w)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "height", "weight"])
        for i in range(n):
            wr.writerow([i, hh[i], "" if i % 10 == 0 else ww[i]])
    return f, h


def test_detect_and_repair(tmp_path, capsys) -> None:
    src, out = tmp_path / "in.csv", tmp_path / "out.csv"
    f, h = _write(src)
    code = main(
        [str(src), "--col", "height:cm,in", "--col", "weight:kg,lb", "--out", str(out)]
    )
    assert code == 1
    assert "mixed units detected" in capsys.readouterr().out
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fixed = np.array([float(r["height"]) for r in rows])
    assert abs(np.median(fixed) - 170) < 2
    assert rows[0]["weight"] == ""  # missing stays missing
    assert [r["id"] for r in rows][:3] == ["0", "1", "2"]


def test_reverse_unit_order_accepted(tmp_path, capsys) -> None:
    src = tmp_path / "in.csv"
    _write(src)
    assert main([str(src), "--col", "height:in,cm"]) == 1


def test_clean_file_exit_zero(tmp_path) -> None:
    src = tmp_path / "in.csv"
    _write(src, flag_frac=0.0)
    assert main([str(src), "--col", "height:cm,in", "--col", "weight:kg,lb"]) == 0


def test_errors(tmp_path, capsys) -> None:
    src = tmp_path / "in.csv"
    _write(src)
    assert main([str(src), "--col", "nope:cm,in"]) == 2
    with pytest.raises(SystemExit):
        main([str(src), "--col", "height:cm,furlong"])
