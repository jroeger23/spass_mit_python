from src.neuronal_network.types import ActivationLayer
import numpy as np
import numpy.typing as npt

class Sigmoid(ActivationLayer):
  def __init__(self):
    self.sigmoid = None

  def forward(self, x_input: np.ndarray) -> npt.NDArray:
    self.sigmoid = 1 / (1 + np.exp(-x_input))
    return self.sigmoid

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.sigmoid == None:
      raise RuntimeError("Sigmoid.backward(): no prior forward() call")
    return gradient * self.sigmoid * (1 - self.sigmoid)

  def __str__(self) -> str:
    return "sigmoid"

class ReLU(ActivationLayer):
  def __init__(self):
    self.x_input = None

  def forward(self, x_input: np.ndarray) -> npt.NDArray:
    self.x_input = x_input
    np.stack((x_input, np.zeros(x_input.shape)), axis=-1).max(axis=-1)

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.x_input == None:
      raise RuntimeError("ReLU.backward(): no prior forward() call")
    g = self.x_input > 0
    return g*gradient

  def __str__(self) -> str:
    return "ReLU"