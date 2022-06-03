from src.neuronal_network.types import Loss
import numpy as np
import numpy.typing as npt

class MSE(Loss):
  def __init__(self):
    self.y_output = None
    self.y_expected = None
    
  def __call__(self, y_output : np.ndarray, y_expected : np.ndarray) -> npt.NDArray:
    self.y_output = y_output
    self.y_expected = y_expected
    return np.square(y_output - y_expected).mean(axis=1)

  def gradient(self) -> npt.NDArray:
    if self.y_output is None or self.y_expected is None:
      raise RuntimeError("MSE.gradient(): no prior call to __call__()")
    return -2/self.y_output.shape[0] * (self.y_expected - self.y_output)

  def __str__(self) -> str:
    return "MSE"