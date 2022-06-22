from src.neuronal_network.types import ActivationLayer, Optimizer
from src.neuronal_network.layers import LinearLayer, ConvolutionLayer, FlattenLayer
import numpy as np
import numpy.typing as npt
import pickle


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

class CNN():
  def __init__(self, input_dims, n_classes : int, kernel_dims, fully_connected,
               hidden_act = ActivationLayer, output_act = ActivationLayer, optimizer = Optimizer):
    self.layers = []

    ih, iw = input_dims
    for k_dim in kernel_dims:
      self.layers.append(ConvolutionLayer((ih,iw,1), k_dim, optimizer()))
      self.layers.append(hidden_act())
      ih -= k_dim[0]-1
      iw -= k_dim[1]-1

    self.layers.append(FlattenLayer())

    last_n = ih*iw
    for fc_n in fully_connected:
      self.layers.append(LinearLayer(last_n, fc_n, optimizer()))
      self.layers.append(hidden_act())
      last_n = fc_n

    self.layers.append(LinearLayer(last_n, n_classes, optimizer()))
    self.layers.append(output_act())

    self.description = "CNN"

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

  def save(self, fname : str) -> None:
    with open(fname, mode='wb') as f:
      pickle.dump(self, f)

  def load(fname : str):
    with open(fname, mode='rb') as f:
      return pickle.load(f)

  def __str__(self) -> str:
    return self.description