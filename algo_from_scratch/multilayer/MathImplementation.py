import os
from typing import Union

from AbstractClasses import LibMath
import numpy as np
from dotenv import load_dotenv
load_dotenv()

gpu_usage = int(os.getenv("gpu_use"))

if gpu_usage:
    import torch


class NumpyMath(LibMath):
    @staticmethod
    def apply_filter(first2darray: np.ndarray, filter: np.ndarray) -> Union[float, int]:
        assert first2darray.shape == filter.shape, "Must be of the same shape, apply_filter"
        element_wise_mult = np.multiply(first2darray, filter)
        return np.sum(element_wise_mult)




a = np.array([[1,2, 5],[3,4, 5]])
b = np.array([[5,6],[7,8]])

(f, f) = a.shape
print(f)

# print(NumpyMath.apply_filter(a, b))
