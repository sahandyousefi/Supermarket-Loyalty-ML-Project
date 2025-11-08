"""Array type aliases shared by the strictly typed packages."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]
Split = tuple[IntArray, IntArray]
