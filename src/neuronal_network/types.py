import numpy as np
import numpy.typing as npt

class NNLayer():
  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    raise NotImplementedError("NNLayer.forward() not implemented")

  def backward(self, gradient : np.ndarray) -> npt.NDArray:
    raise NotImplementedError("NNLayer.backward() not implemented")

  def fit(self, learning_rate):
    pass


class ActivationLayer(NNLayer):
  def forward(self, x_input: np.ndarray) -> npt.NDArray:
    return x_input

  def backward(self, gradient: np.ndarray) -> npt.NDArray:
    return gradient

  def __str__(self) -> str:
    return "Id"

class Loss:
  def __call__(self, x_input: np.ndarray) -> npt.NDArray:
    raise NotImplementedError("Loss({self}).__call__() not implemented")

  def gradient(self) -> npt.NDArray:
    raise NotImplementedError("Loss({self}).gradient() not implemented")

  def __str__(self) -> str:
    return ""
