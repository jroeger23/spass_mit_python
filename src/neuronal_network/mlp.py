from src.neuronal_network.types import ActivationLayer, NNLayer, Optimizer
import numpy as np
import numpy.typing as npt
import pickle

class LinearLayer(NNLayer):
  def __init__(self, n_input : int, n_output : int, optimizer = Optimizer()):
    n_input += 1
    self.weights = np.random.normal(0, np.sqrt(2/n_input), (n_input, n_output))
    self.optimizer = optimizer

  def addBias(m : np.ndarray) -> npt.NDArray:
    n_samples, _ = m.shape
    bias = np.ones((n_samples, 1))
    return np.hstack([bias, m])

  def removeBias(m : np.ndarray) -> npt.NDArray:
    return m[:,1:]

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    self.x_input = LinearLayer.addBias(x_input)
    self.optimizer.forward(self.x_input)
    return self.x_input.dot(self.weights)

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    self.optimizer.backward(gradient)
    return LinearLayer.removeBias(gradient.dot(self.weights.transpose()))

  def fit(self):
    self.optimizer.adjust(self.weights)

class MLP():
  def __init__(self, n_input : int, n_hidden : list, n_output : int,
               hidden_act = ActivationLayer, output_act = ActivationLayer, optimizer = Optimizer):
    self.layers = []

    if len(n_hidden) == 0:
      self.layers.append(LinearLayer(n_input, n_output))
    else:
      for i,o in zip([n_input] + n_hidden, n_hidden):
        self.layers.append(LinearLayer(i,o, optimizer()))
        self.layers.append(hidden_act())
      self.layers.append(LinearLayer(n_hidden[-1], n_output, optimizer()))
    
    self.layers.append(output_act())

    dims = f"{n_input}"
    for h in n_hidden:
      dims += f"x{h}"
    dims += f"x{n_output}"
    self.description = f"MLP {dims}, {hidden_act()} (hidden), {output_act()} (out), {optimizer()}"

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    for l in self.layers:
      x_input = l.forward(x_input)
    return x_input

  def backward(self, gradient : np.ndarray) -> npt.NDArray:
    for l in reversed(self.layers):
      gradient = l.backward(gradient)
    return gradient

  def fit(self):
    for l in self.layers:
      l.fit()

  def classify(self, x_input : np.ndarray) -> npt.NDArray:
    return np.argmax(self.forward(x_input), axis=1)

  def weightsList(self):
    return [l.weights.copy() for l in self.layers[::2]]

  def save(self, fname : str) -> None:
    with open(fname, mode='wb') as f:
      pickle.dump(self, f)

  def load(fname : str):
    with open(fname, mode='rb') as f:
      return pickle.load(f)

  def __str__(self) -> str:
    return self.description