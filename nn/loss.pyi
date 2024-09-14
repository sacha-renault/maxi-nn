# type: ignore
import sys
from typing import Literal, List, Any, Optional, Tuple, Dict, Callable
import numpy as np
import enum
import numpy
from .tensor import Tensor_float

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  AUTOGENERATED CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# <litgen_stub> // Autogenerated code below! Do not edit!
####################    <generated_from:Loss.hpp>    ####################


#  ------------------------------------------------------------------------
#      <template specializations for function meanSquaredError>
@overload
def mean_squared_error(pred: Tensor_float, real: Tensor_float) -> Tensor_float:
    pass
#      </template specializations for function meanSquaredError>
#  ------------------------------------------------------------------------

#  ------------------------------------------------------------------------
#      <template specializations for function meanAbsoluteError>
@overload
def mean_absolute_error(pred: Tensor_float, real: Tensor_float) -> Tensor_float:
    pass
#      </template specializations for function meanAbsoluteError>
#  ------------------------------------------------------------------------

#  ------------------------------------------------------------------------
#      <template specializations for function categoricalCrossEntropy>
@overload
def categorical_cross_entropy(
    pred: Tensor_float,
    real: Tensor_float,
    epsilon: float = 1e-10
    ) -> Tensor_float:
    pass
#      </template specializations for function categoricalCrossEntropy>
#  ------------------------------------------------------------------------

####################    </generated_from:Loss.hpp>    ####################

# </litgen_stub> // Autogenerated code end!