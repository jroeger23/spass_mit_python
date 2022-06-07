import src.neuronal_network.activation as act
import numpy as np

def test_SoftMax():
  softmax = act.SoftMax()
  verifier = act.SoftMaxTorch()

  data = np.random.random((5,8))
  data2 = np.random.random((5,8))

  fwd_is = softmax.forward(data)
  fwd_should = verifier.forward(data)
  bwd_is = softmax.backward(data2)
  bwd_should = verifier.backward(data2)

  np.testing.assert_array_almost_equal(fwd_is, fwd_should)
  np.testing.assert_array_almost_equal(bwd_is, bwd_should)
  