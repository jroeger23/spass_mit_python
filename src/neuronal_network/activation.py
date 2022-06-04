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
    if self.sigmoid is None:
      raise RuntimeError("Sigmoid.backward(): no prior forward() call")
    return gradient * self.sigmoid * (1 - self.sigmoid)

  def __str__(self) -> str:
    return "sigmoid"

class ReLU(ActivationLayer):
  def __init__(self):
    self.x_input = None

  def forward(self, x_input: np.ndarray) -> npt.NDArray:
    self.x_input = x_input
    return np.stack((x_input, np.zeros(x_input.shape)), axis=-1).max(axis=-1)

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.x_input is None:
      raise RuntimeError("ReLU.backward(): no prior forward() call")
    g = self.x_input > 0
    return g*gradient

  def __str__(self) -> str:
    return "ReLU"

class SoftMax(ActivationLayer):
  def __init__(self) -> None:
    self.softmax = None

  def forward(self, x_input: np.ndarray) -> npt.NDArray:
    e = np.exp(x_input - np.c_[x_input.max(axis=1)]) # subtracting max cancels out
    s = np.c_[e.sum(axis=1)]
    self.softmax = np.divide(e, s, where=(s != 0.0))
    return self.softmax

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.softmax is None:
      raise RuntimeError("SoftMax.backward(): no prior forward() call")

    g = -self.softmax.dot(self.softmax)
    np.fill_diagonal(g, self.softmax * (1 - self.softmax))
    return g.dot(gradient)

  def __str__(self) -> str:
      return "softmax"