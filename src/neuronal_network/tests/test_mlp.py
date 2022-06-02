from src.neuronal_network.mlp import LinearLayer

import numpy as np

def test_addBias():
  n, m = (5,10)
  ds = np.asarray(range(0,n*m)).reshape(n,m)

  ret = LinearLayer.addBias(ds)

  ret_n, ret_m = ret.shape

  assert ret_n == n, "N Dimension mismatch"
  assert ret_m == m+1, "M Dimension mismatch"
  np.testing.assert_array_equal(ret[:,[0]], np.ones((n,1)))
  np.testing.assert_array_equal(ret[:,1:], ds)

def test_removeBias():
  n, m = (5,10)
  ds = np.asarray(range(0,n*m)).reshape(n,m)

  ret = LinearLayer.removeBias(LinearLayer.addBias(ds))

  np.testing.assert_array_equal(ds, ret)