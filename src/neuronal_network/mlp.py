from src.neuronal_network.types import ActivationLayer, NNLayer
import numpy as np
import numpy.typing as npt

class LinearLayer(NNLayer):
  def __init__(self, n_input : int, n_output : int, mu = 0, sigma = 0.01):
    self.weights = np.random.normal(mu, sigma, (n_input, n_output))
    self.weights = np.vstack((np.zeros(n_output), self.weights))
    self.x_input = None

  def addBias(m : np.ndarray) -> npt.NDArray:
    n_samples, _ = m.shape
    bias = np.ones((n_samples, 1))
    return np.hstack([bias, m])

  def removeBias(m : np.ndarray) -> npt.NDArray:
    return m[:,1:]

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    self.x_input = self.addBias(x_input)
    return self.x_input.dot(self.weights)

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.x_input == None:
      raise RuntimeError("LinearLayer.backward(): no prior call to forward()")

    self.dw = self.x_input.transpose().dot(gradient)
    return self.removeBias(gradient.dot(self.weights.transpose()))

  def fit(self, learning_rate):
    if self.dw == None:
      raise RuntimeError("LinearLayer.fit(): no prior call to backward()")
    self.weights -= learning_rate * self.dw


class MLP():
  def __init__(self, n_input : int, n_hidden : list, n_output : int,
               hidden_act = ActivationLayer, output_act = ActivationLayer):
    self.layers = []

    for i,o in zip([n_input] + n_hidden, n_hidden[1:]):
      self.layers.append(LinearLayer(i,o))
      self.layers.append(hidden_act())
    
    self.layers.append(LinearLayer(n_hidden[-1], n_output))
    self.layers.append(output_act())

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    for l in self.layers:
      x_input = l.forward(x_input)
    return x_input

  def backward(self, loss : np.ndarray) -> npt.NDArray:
    for l in reversed(self.layers):
      loss = l.backward(loss)
    return loss

  def fit(self, learning_rate):
    for l in self.layers:
      l.fit(learning_rate)