from src.neuronal_network.types import Optimizer
import numpy as np

class Momentum(Optimizer):
  def __init__(self, learning_rate: float, momentum: float) -> None:
    self.alpha = learning_rate
    self.eta = momentum
    self.x_input = None
    self.last_v = np.zeros(1)

  def forward(self, x_input: np.ndarray) -> None:
    self.x_input = x_input

  def backward(self, gradient: np.ndarray) -> None:
    if self.x_input is None:
      raise RuntimeError("Optimizer.backward(): no prior call to forward()")
    dw = self.x_input.transpose().dot(gradient)
    self.last_v = self.eta * self.last_v + self.alpha * dw

  def adjust(self, tgt_weight: np.ndarray) -> None:
    tgt_weight -= self.last_v

  def __str__(self) -> str:
    return f"Momentum ($\\alpha$={self.alpha}, $\\eta$={self.eta})"