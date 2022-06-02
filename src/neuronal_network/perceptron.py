import numpy as np
import numpy.typing as npt
from src.common.vectorize import one_hot

class LinearLayer:
  def __init__(self, n_input : int, n_output : int):
    self.weights = np.random.normal(0, 0.01, (n_input, n_output))
    self.weights = np.vstack((np.zeros(n_output), self.weights))
    self.x_input = np.ndarray(0)

  def addBias(self, m : np.ndarray) -> npt.NDArray:
    n_samples, _ = m.shape
    bias = np.ones((n_samples, 1))
    return np.hstack([bias, m])

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    self.x_input = self.addBias(x_input)
    return self.x_input.dot(self.weights)

  def adjustWeights(self, loss_gradient : np.ndarray, learning_rate : float):
    x = np.expand_dims(self.x_input.sum(axis=0), axis=1)
    g = np.expand_dims(loss_gradient.sum(axis=0), axis=0)
    self.weights -= learning_rate*g*x


class NNError:
  def __call__(self):
    pass

  def gradient(self):
    pass

  def __str__(self) -> str:
    return "unspecified"

class MSE(NNError):
  def __init__(self):
    self.y_output = np.ndarray(0)
    self.y_expected = np.ndarray(0)
    
  def __call__(self, y_output : np.ndarray, y_expected : np.ndarray) -> npt.NDArray:
    self.y_output = y_output
    self.y_expected = y_expected
    return np.square(y_output - y_expected).mean(axis=1)

  def gradient(self) -> npt.NDArray:
    return -2/self.y_output.shape[0] * (self.y_expected - self.y_output)

  def __str__(self) -> str:
    return "MSE"

class NNActivation():
  def __call__(self, y_output : np.ndarray) -> npt.NDArray:
    pass

  def gradient(self, chain_gradient : np.ndarray):
    pass

  def __str__(self) -> str:
    return "unspecified"

class SigmoidActivation(NNActivation):
  def __init__(self):
    self.sigmoid = None

  def __call__(self, y_output : np.ndarray) -> npt.NDArray:
    self.sigmoid = 1.0/(1.0 + np.exp(-y_output))
    return self.sigmoid

  def gradient(self, chain_gradient : np.ndarray):
    return chain_gradient * self.sigmoid * (1-self.sigmoid)

  def __str__(self) -> str:
    return "Sigmoid"

class IdentityActivation(NNActivation):
  def __call__(self, y_output: np.ndarray) -> npt.NDArray:
    return y_output

  def gradient(self, chain_gradient: np.ndarray):
    return chain_gradient

  def __str__(self) -> str:
    return "id"

class ReLU(NNActivation):
  def __init__(self):
    self.y_output = None

  def __call__(self, y_output: np.ndarray) -> npt.NDArray:
    self.y_output = y_output
    y = np.stack((y_output, np.zeros(y_output.shape)), axis=-1)
    return np.max(y, axis=-1)

  def gradient(self, chain_gradient: np.ndarray):
    g = self.y_output > 0
    return g*chain_gradient

  def __str__(self) -> str:
    return "ReLU"


class Perceptron:
  def __init__(self, n_input : int, n_output : int, learning_rate: float,
               loss : NNError = MSE(),
               activation : NNActivation = SigmoidActivation()):
    self.linear_layer = LinearLayer(n_input, n_output)
    self.learning_rate = learning_rate
    self.loss = loss
    self.activation = activation
    self.y_pred = np.ndarray(0)

  def forward(self, x_input : np.ndarray) -> npt.NDArray:
    return self.activation(self.linear_layer.forward(x_input))

  def train(self, x_input : np.ndarray, y_expected : np.ndarray) -> npt.NDArray:
    y_output = self.forward(x_input)
    err = self.loss(y_output, y_expected)
    gradient = self.loss.gradient()
    gradient = self.activation.gradient(gradient)
    self.linear_layer.adjustWeights(gradient, self.learning_rate)
    return err


  def classify(self, x_input : np.ndarray):
    return one_hot(self.classifyIx(x_input), self.linear_layer.weights.shape[1])

  def classifyIx(self, x_input : np.ndarray):
    return np.argmax(self.forward(x_input), axis=1)