"""Golden-vector tests for the VeriTMM Sobol engine.

These tests verify that _sobol_direction_numbers + the Gray-code traversal
produce the standard Joe-Kuo 2010 sequence exactly, with no digital shift.
The reference values were independently derived from first principles and
confirmed by running the engine directly (no scipy dependency at test time).
"""

from __future__ import annotations

import numpy as np
import pytest

from tmm_engine.research.sampling import (
    SOBOL_BITS,
    _sobol_direction_numbers,  # noqa: PLC2701
)

# ---------------------------------------------------------------------------
# Helper: compute raw (unshifted) Sobol points from direction numbers
# ---------------------------------------------------------------------------

def _raw_sobol(dimension: int, n: int, *, skip: int = 0) -> list[tuple[float, ...]]:
    """Return *n* raw Sobol points for *dimension*, starting at *skip*.

    No digital shift is applied; this tests the direction numbers and
    Gray-code traversal in isolation.
    """
    directions = _sobol_direction_numbers(dimension)
    denom = float(2**SOBOL_BITS)
    results: list[tuple[float, ...]] = []
    for index in range(skip, skip + n):
        gray = index ^ (index >> 1)
        row: list[float] = []
        for col in range(dimension):
            value = 0
            bits = gray
            bit = 0
            while bits:
                if bits & 1:
                    value ^= int(directions[col, bit])
                bits >>= 1
                bit += 1
            row.append(value / denom)
        results.append(tuple(row))
    return results


# ---------------------------------------------------------------------------
# Direction-number sanity checks
# ---------------------------------------------------------------------------

class TestDirectionNumbers:
    def test_dim1_first_bit_is_msb(self) -> None:
        d = _sobol_direction_numbers(1)
        assert int(d[0, 0]) == (1 << (SOBOL_BITS - 1))

    def test_dim1_is_successive_powers_of_two(self) -> None:
        d = _sobol_direction_numbers(1)
        for bit in range(SOBOL_BITS):
            assert int(d[0, bit]) == (1 << (SOBOL_BITS - 1 - bit))

    def test_shape(self) -> None:
        for dim in (1, 2, 4, 8, 16):
            d = _sobol_direction_numbers(dim)
            assert d.shape == (dim, SOBOL_BITS)
            assert d.dtype == np.uint32

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            _sobol_direction_numbers(0)
        with pytest.raises(ValueError, match="dimension"):
            _sobol_direction_numbers(17)


# ---------------------------------------------------------------------------
# Golden vectors: dim=1
# ---------------------------------------------------------------------------

# Standard Joe-Kuo Sobol sequence, column 0, first 8 points (skip=0)
_DIM1_SKIP0 = [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125]

# Same sequence continued from skip=4
_DIM1_SKIP4 = [0.375, 0.875, 0.625, 0.125]


class TestGoldenDim1:
    def test_first_8_points(self) -> None:
        pts = _raw_sobol(1, 8)
        actual = [r[0] for r in pts]
        assert actual == pytest.approx(_DIM1_SKIP0, abs=1e-15)

    def test_skip4_matches_tail_of_skip0(self) -> None:
        pts = _raw_sobol(1, 4, skip=4)
        actual = [r[0] for r in pts]
        assert actual == pytest.approx(_DIM1_SKIP4, abs=1e-15)

    def test_first_point_is_zero(self) -> None:
        pts = _raw_sobol(1, 1)
        assert pts[0][0] == 0.0

    def test_all_points_in_unit_interval(self) -> None:
        pts = _raw_sobol(1, 32)
        for (v,) in pts:
            assert 0.0 <= v < 1.0


# ---------------------------------------------------------------------------
# Golden vectors: dim=2
# ---------------------------------------------------------------------------

# Joe-Kuo 2010, dim=2 (degree=1, coeff=0, m=(1,)), first 8 points
_DIM2_SKIP0 = [
    (0.0, 0.0),
    (0.5, 0.5),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.375, 0.375),
    (0.875, 0.875),
    (0.625, 0.125),
    (0.125, 0.625),
]

_DIM2_SKIP4 = [
    (0.375, 0.375),
    (0.875, 0.875),
    (0.625, 0.125),
    (0.125, 0.625),
]


class TestGoldenDim2:
    def test_first_8_points(self) -> None:
        pts = _raw_sobol(2, 8)
        assert pts == pytest.approx(_DIM2_SKIP0, abs=1e-15)

    def test_skip4(self) -> None:
        pts = _raw_sobol(2, 4, skip=4)
        assert pts == pytest.approx(_DIM2_SKIP4, abs=1e-15)

    def test_col0_matches_dim1(self) -> None:
        pts2 = _raw_sobol(2, 8)
        pts1 = _raw_sobol(1, 8)
        assert [r[0] for r in pts2] == pytest.approx([r[0] for r in pts1], abs=1e-15)

    def test_all_points_in_unit_interval(self) -> None:
        pts = _raw_sobol(2, 32)
        for row in pts:
            for v in row:
                assert 0.0 <= v < 1.0


# ---------------------------------------------------------------------------
# Golden vectors: dim=4
# ---------------------------------------------------------------------------

# Joe-Kuo 2010 sequence, dim=4, first 8 points (skip=0)
_DIM4_SKIP0 = [
    (0.0,    0.0,    0.0,    0.0),
    (0.5,    0.5,    0.5,    0.5),
    (0.75,   0.25,   0.25,   0.25),
    (0.25,   0.75,   0.75,   0.75),
    (0.375,  0.375,  0.625,  0.875),
    (0.875,  0.875,  0.125,  0.375),
    (0.625,  0.125,  0.875,  0.625),
    (0.125,  0.625,  0.375,  0.125),
]

# Points 8-11 (skip=8, n=4)
_DIM4_SKIP8 = [
    (0.1875, 0.3125, 0.9375, 0.4375),
    (0.6875, 0.8125, 0.4375, 0.9375),
    (0.9375, 0.0625, 0.6875, 0.1875),
    (0.4375, 0.5625, 0.1875, 0.6875),
]


class TestGoldenDim4:
    def test_first_8_points(self) -> None:
        pts = _raw_sobol(4, 8)
        assert pts == pytest.approx(_DIM4_SKIP0, abs=1e-15)

    def test_skip8(self) -> None:
        pts = _raw_sobol(4, 4, skip=8)
        assert pts == pytest.approx(_DIM4_SKIP8, abs=1e-15)

    def test_col0_matches_dim1(self) -> None:
        pts4 = _raw_sobol(4, 8)
        pts1 = _raw_sobol(1, 8)
        assert [r[0] for r in pts4] == pytest.approx([r[0] for r in pts1], abs=1e-15)

    def test_all_points_in_unit_interval(self) -> None:
        pts = _raw_sobol(4, 32)
        for row in pts:
            for v in row:
                assert 0.0 <= v < 1.0


# ---------------------------------------------------------------------------
# Gray-code coverage: 2^k points fill [0,1) with 2^k equal-width slots
# ---------------------------------------------------------------------------

class TestEquidistribution:
    @pytest.mark.parametrize("dim", [1, 2, 4])
    def test_2k_points_cover_unit_hypercube_uniformly(self, dim: int) -> None:
        """For each dimension, 2^k points should be distinct and cover [0,1)."""
        n = 64  # 2^6
        pts = _raw_sobol(dim, n)
        for col in range(dim):
            col_vals = sorted(r[col] for r in pts)
            expected = [i / n for i in range(n)]
            assert col_vals == pytest.approx(expected, abs=1e-15)
