from typing import Tuple
from src.neuronal_network.types import Optimizer, NNLayer
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as fn

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


class ConvolutionLayerTorch(NNLayer):
  def __init__(self, n_input : int, n_output : int, kernel : Tuple[int, int], stride=1, padding=0, optimizer=Optimizer()):
    weights = np.random.normal(0, np.sqrt(2/n_input), (n_input, n_output, kernel[0], kernel[1]))

    self.weights = torch.tensor(weights, dtype=float, requires_grad=True)
    self.x_input = None
    self.y_output = None
    self.stride = stride
    self.padding = padding
    self.optimizer = optimizer


  def forward(self, x_input: np.ndarray) -> npt.NDArray:
    self.x_input = torch.tensor(x_input, dtype=float, requires_grad=True)


    self.y_output = fn.conv2d(self.x_input, self.weights, stride=self.stride, padding=self.padding)

    return self.y_output.detach().numpy()

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    if self.x_input is None or self.y_output is None:
      raise RuntimeError("ConvolutionLayerTorch.backward(): no prior call to forward()")
      
    gradient = torch.tensor(gradient, dtype=float)
    gradient = torch.autograd.grad(self.y_output, self.x_input, gradient, is_grads_batched=True)[0]

    w_gradient = torch.autograd.grad(self.y_output, self.weights, gradient, is_grads_batched=True)[0]
    w_gradient = w_gradient.detach().numpy()
    self.optimizer.backward(w_gradient)

  def fit(self):
    weights = self.weights.detach().numpy()
    self.optimizer.adjust(weights)
    self.weights = torch.tensor(weights, dtype=float, requires_grad=True)


    

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

    dk = np.zeros(self.kernel.shape)
    
    for y,x in np.ndindex(dk.shape[1:3]):
        x_input_cut = self.x_input[:,y:ih-kh+y+1,x:iw-kw+x+1,:]
        dk[0,y,x,0] = np.sum(x_input_cut * gradient)

    self.optimizer.backward(dk)

    grad_pad_d = ((0,0), (kh-1, kh-1), (kw-1, kw-1), (0,0))
    grad_pad_v = ((0,0), (0,0), (0,0), (0,0))
    grad_pad = np.pad(gradient, grad_pad_d, mode='constant', constant_values=grad_pad_v)
    dx = np.zeros(self.x_input.shape)
    flipped_kernel = np.flip(self.kernel, axis=(1,2))

    for y,x in np.ndindex((ih,iw)):
      dx[:,y,x,:] = np.sum(grad_pad[:,y:y+kh,x:x+kw,:] * flipped_kernel)

    return dx

  def fit(self):
    self.optimizer.adjust(self.kernel)

  
class FlattenLayer(NNLayer):
  def __init__(self) -> None:
    self.shape = None

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    self.shape = x_input.shape
    n_samples, ih, iw, ic = self.shape
    return x_input.reshape((n_samples, ih*iw*ic))

  def backward(self, gradient : np.ndarray) -> npt.NDArray:
    if self.shape is None:
      raise RuntimeError("FlattenLayer.backward(): no prior call to forward()")
    return gradient.reshape(self.shape)