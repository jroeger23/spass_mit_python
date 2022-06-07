from src.neuronal_network.types import ActivationLayer
import numpy as np
import numpy.typing as npt
import torch

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
    e = np.exp(x_input - x_input.max(axis=1, keepdims=True)) # subtracting max cancels out
    s = e.sum(axis=1, keepdims=True)
    self.softmax = np.divide(e, s, where=(s != 0.0))
    return self.softmax

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.softmax is None:
      raise RuntimeError("SoftMax.backward(): no prior forward() call")


    g = -np.einsum('ij,ik->ijk', self.softmax, self.softmax)
    diags = self.softmax * (1 - self.softmax)
    for ix in range(0,g.shape[0]):
      np.fill_diagonal(g[ix], diags[ix])
    return np.einsum('ik,ijk -> ij', gradient, g)

  def __str__(self) -> str:
      return "softmax"
    
class SoftMaxTorch(ActivationLayer):
  def __init__(self):
    self.x_input = None
    self.softmax = None

  def forward(self, x_input):
    self.x_input = torch.tensor(x_input, dtype=float, requires_grad=True)
    self.softmax = torch.nn.functional.softmax(self.x_input, dim=1)
    return self.softmax.detach().numpy()
  
  def backward(self, gradient):
    if self.x_input is None or self.softmax is None:
      raise RuntimeError("SoftMaxTorch.backward(): no prior forward() call")
    gradient = torch.tensor(gradient, dtype=float)
    gradient = torch.autograd.grad(self.softmax, self.x_input, gradient)[0]
    return gradient.detach().numpy()

  def __str__(self) -> str:
      return "softmax"