# type: ignore
import sys
from typing import Literal, List, Any, Optional, Tuple, Dict, Callable
import numpy as np
import enum
import numpy
from .tensor import Tensor_float

class ComputationGraph_float:  # Python specialization for ComputationGraph<float>
    def __init__(self, root: Tensor_float) -> None:
        pass

    def get_nodes(self) -> List[Tensor_float]:
        """/ @brief get nodes in one topological sort
        / @return sortes nodes of the graph
        """
        pass

    def forward(self) -> None:
        """/ @brief Apply forward() on each node of the graph"""
        pass

    def backward(self) -> None:
        """/ @brief Apply backward() on each node of the graph (reverse order)"""
        pass

    def zero_grad(self) -> None:
        """/ @brief Set gradient to 0 to every node"""
        pass
#      </template specializations for class ComputationGraph>
#  ------------------------------------------------------------------------