# -*- coding: utf-8 -*-
"""Shared spectral helper functions for dataset generation."""

from __future__ import annotations

import numpy as np


def hemisphere_weights_unitless(angles_rad: np.ndarray) -> np.ndarray:
    """
    Return normalized quadrature weights for hemisphere averaging.

    With these weights:
      integral_hemisphere / pi = 2 * sum_i w_i * f(theta_i)
    """
    ang = np.asarray(angles_rad, dtype=np.float64).ravel()
    if ang.size == 0:
        raise ValueError("angles_rad cannot be empty")
    edges = np.zeros(ang.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 0.5 * np.pi
    for i in range(1, ang.size):
        edges[i] = 0.5 * (ang[i - 1] + ang[i])
    s2 = np.sin(edges) ** 2
    return 0.5 * (s2[1:] - s2[:-1])

