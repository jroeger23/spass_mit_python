from src.common.dot import norm, histogram
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


def test_histogram():
  ds = (40 * np.random.rand(10,15,5)).round().astype(int)
  ds_flat = [ x for (_,x) in np.ndenumerate(ds) ]

  hist = histogram(ds)
  histId = histogram(ds, lambda x:x)

  for (k,v) in hist.items():
    assert ds_flat.count(k) == v, "histogram[{}] = {}, but {} required".format(k,v, ds_flat.count(k))

  for (k,v) in histId.items():
    assert ds_flat.count(k)*k == v, "histogramId[{}] = {}, but {} required".format(k,v, ds_flat.count(k)*k)