import warnings
import math
from operator import mul
from functools import reduce

import torch
from torch._C import _infer_size, _add_docstr
from . import _functions
from .modules import utils
from ._functions.linear import Bilinear
from ._functions.padding import ConstantPadNd
from ._functions.vision import GridSampler, AffineGridGenerator
from torch.autograd import Variable
from .modules.utils import _single, _pair, _triple
r"""Functional interface"""
from typing import Callable, List, Optional, Tuple, Union
import math
import warnings

def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        return torch.addmm(bias, input, weight.t())

    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output
