from src.neuronal_network.types import Optimizer
import numpy as np

class Momentum(Optimizer):
  def __init__(self, learning_rate: float, momentum: float) -> None:
    self.alpha = learning_rate
    self.eta = momentum
    self.last_v = np.zeros(1)

  def backward(self, weights_gradient: np.ndarray) -> None:
    self.last_v = self.eta * self.last_v + self.alpha * weights_gradient

  def adjust(self, tgt_weight: np.ndarray) -> None:
    tgt_weight -= self.last_v

  def __str__(self) -> str:
    return f"Momentum ($\\alpha$={self.alpha}, $\\eta$={self.eta})"


class Adam(Optimizer):
  def __init__(self, alpha: float, beta1: float, beta2: float, epsilon: float) -> None:
    self.alpha = alpha
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.v = np.zeros(1)
    self.m = np.zeros(1)
    self.x_input = None
    self.t = 1

  def backward(self, weights_gradient: np.ndarray) -> None:
    self.m = self.beta1 * self.m + (1-self.beta1) * weights_gradient
    self.v = self.beta2 * self.v + (1-self.beta2) * np.square(weights_gradient)

  def adjust(self, tgt_weight: np.ndarray) -> None:
    m = self.m / (1 - np.power(self.beta1, self.t))
    v = self.v / (1 - np.power(self.beta2, self.t))
    self.t += 1
    tgt_weight -= self.alpha * m / (np.sqrt(v) + self.epsilon)

  def __str__(self) -> str:
    return f"Adam ($\\alpha$={self.alpha}, $\\beta_1$={self.beta1}, $\\beta_2$={self.beta2})"