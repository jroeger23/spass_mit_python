from src.neuronal_network.types import Optimizer, NNLayer
import numpy as np
import numpy.typing as npt

class LinearLayer(NNLayer):
  def __init__(self, n_input : int, n_output : int, optimizer = Optimizer()):
    n_input += 1
    self.weights = np.random.normal(0, np.sqrt(2/n_input), (n_input, n_output))
    self.optimizer = optimizer
    self.x_input = None

  def addBias(m : np.ndarray) -> npt.NDArray:
    n_samples, _ = m.shape
    bias = np.ones((n_samples, 1))
    return np.hstack([bias, m])

  def removeBias(m : np.ndarray) -> npt.NDArray:
    return m[:,1:]

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    self.x_input = LinearLayer.addBias(x_input)
    return self.x_input.dot(self.weights)

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.x_input is None:
      raise RuntimeError("LinearLayer.backward(): No prior call to forward()")

    dw = self.x_input.transpose().dot(gradient)
    self.optimizer.backward(dw)
    return LinearLayer.removeBias(gradient.dot(self.weights.transpose()))

  def fit(self):
    self.optimizer.adjust(self.weights)