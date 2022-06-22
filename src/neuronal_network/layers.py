from typing import Tuple
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


class ConvolutionLayer(NNLayer):
  def __init__(self, input_dims : Tuple[int, int, int], kernel_dims : Tuple[int, int],
               optimizer = Optimizer()) -> None:
    ih, iw, ic = input_dims
    kh, kw = kernel_dims

    self.kernel = np.random.normal(0, np.sqrt(2/(iw*ih)), (1, kh, kw, 1))
    self.optimizer = optimizer
    self.x_input = None


  def forward(self, x_input: np.ndarray) -> npt.NDArray:
    il, ih, iw, ic = x_input.shape
    _, kh, kw, _ = self.kernel.shape

    self.x_input = x_input

    y_output = np.zeros((il, ih-kh+1, iw-kw+1, ic))

    for y, x in np.ndindex(y_output.shape[1:3]):
      y_output[:,y,x,:] = np.sum(x_input[:,y:y+kh,x:x+kw,:] * self.kernel, axis=(1,2))

    return y_output

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    il, ih, iw, ic = self.x_input.shape
    _, kh, kw, _ = self.kernel.shape
    pad_y, pad_x = kh//2, kw//2

    dk = np.zeros(self.kernel.shape)

    for y in range(kh):
      for x in range(kw):
        x_input_cut = self.x_input[:,y:ih-kh+y+1,x:iw-kw+x+1,:]
        dk[0,y,x,0] = np.sum(x_input_cut * gradient)

    self.optimizer.backward(dk)

  def fit(self):
    self.optimizer.adjust(self.kernel)