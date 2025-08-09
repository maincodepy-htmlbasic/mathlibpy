
"""
mathlibpy: A Python math library for numeric, vector, and symbolic operations. It provides utilities for mathematical computations, including:
- Numeric and mathematical utilities (num)
- Vector algebra (vec)
- Symbolic math (sym)

It also provides some utilities for things other than math, such as:
- Units (unit) # will be implemented in 0.2.0

Submodules:
    - num: Numeric and mathematical utilities
    - vec: Vector algebra
    - sym: Symbolic math
"""

from .num import *
from .vec import *
from .sym import *

import mathlibpy.num as num
import mathlibpy.vec as vec
import mathlibpy.sym as sym

__version__ = "0.1.0"

# Define the public API
__all__ = []
__all__ += num.__all__ if hasattr(num, "__all__") else []
__all__ += vec.__all__ if hasattr(vec, "__all__") else []
__all__ += sym.__all__ if hasattr(sym, "__all__") else []

