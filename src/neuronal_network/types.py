import numpy as np
import numpy.typing as npt

class NNLayer():
  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    raise NotImplementedError(f"NNLayer.forward() not implemented")

  def backward(self, gradient : np.ndarray) -> npt.NDArray:
    raise NotImplementedError(f"NNLayer.backward() not implemented")

  def fit(self, learning_rate, momentum=0.01):
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
    raise NotImplementedError(f"Loss({self}).__call__() not implemented")

  def gradient(self) -> npt.NDArray:
    raise NotImplementedError(f"Loss({self}).gradient() not implemented")

  def __str__(self) -> str:
    return ""

class Optimizer:
  def forward(self, x_input: np.ndarray) -> None:
    pass

  def backward(self, gradient: np.ndarray) -> None:
    pass

  def adjust(self, tgt_weight: np.ndarray) -> None:
    raise NotImplementedError(f"Optimizer({self}).adjust() not implemented")

  def __str__(self) -> str:
    return ""