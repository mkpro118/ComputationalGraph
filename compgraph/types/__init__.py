from numbers import Number
from typing import TypeVar

import numpy as np

NumericTypes = (Number, np.number, np.ndarray,)
Numeric = TypeVar('Numeric', *NumericTypes)
