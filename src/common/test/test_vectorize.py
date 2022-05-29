import numpy as np
from src.common.vectorize import one_hot


def test_one_hot():
  upper = 40
  n_ds = 1000

  ds = np.floor(np.random.random(n_ds)*upper).astype(int)

  ret = one_hot(ds, upper)
  ds_recon = np.argmax(ret, axis=1).reshape(-1)

  assert ret.shape == (n_ds, upper)
  np.testing.assert_array_equal(ds_recon, ds)