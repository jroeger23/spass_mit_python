from gradients import norm
import numpy as np


def test_norm():
  ds = np.array(range(-100, 100, 3))

  res1 = norm(ds)
  res2 = norm(ds, -2, 3)

  assert np.min(res1) == 0
  assert np.max(res1) == 1
  assert res1.shape == ds.shape
  assert np.min(res2) == -2 
  assert np.max(res2) == 3
  assert res2.shape == ds.shape