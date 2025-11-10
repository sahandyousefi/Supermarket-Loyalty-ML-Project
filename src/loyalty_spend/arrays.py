"""Array type aliases shared by the strictly typed packages."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]
Split = tuple[IntArray, IntArray]
