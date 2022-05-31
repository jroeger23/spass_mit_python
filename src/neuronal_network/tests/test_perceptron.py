from src.neuronal_network.perceptron import LinearLayer

import numpy as np

def test_addBias():
  n, m = (5,10)
  ds = np.asarray(range(0,n*m)).reshape(n,m)

  test_layer = LinearLayer(1,1)

  ret = test_layer.addBias(ds)

  ret_n, ret_m = ret.shape

  assert ret_n == n, "N Dimension mismatch"
  assert ret_m == m+1, "M Dimension mismatch"
  np.testing.assert_array_equal(ret[:,[0]], np.ones((n,1)))
  np.testing.assert_array_equal(ret[:,1:], ds)
