from numpy import float64
from numpy.typing import NDArray
from typing import Sequence
from types import EllipsisType

Shape = tuple[int, ...]
NDFloat = float64
FloatArray = NDArray[NDFloat]
ArrayIndex = int | slice | None | EllipsisType | Sequence[int]
ArrayIndices = ArrayIndex | tuple[ArrayIndex, ...]
