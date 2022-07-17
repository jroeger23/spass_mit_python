from src.neuronal_network.layers import ConvolutionLayer, LinearLayer, ConvolutionLayerTorch

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


def test_ConvolutionLayerForward():
  # shape check

  il, ih, iw, ic = 20,50,40,1

  data = np.random.random((il,ih,iw,ic))

  cl = ConvolutionLayer((ih,iw,ic), (3,3))

  ret = cl.forward(data)

  rl, rh, rw, rc = ret.shape

  # invariant
  assert il == rl
  assert ic == rc

  # pad
  assert rh == ih-2
  assert rw == iw-2

  # with torch

  verifier = ConvolutionLayerTorch(1, 1, (3,3))
  expected = verifier.forward(data)
