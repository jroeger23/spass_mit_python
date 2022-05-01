import numpy as np
import descriptor as dsc

def test_histogram():
  ds = (40 * np.random.rand(10,15,5)).round().astype(int)
  ds_flat = [ x for (_,x) in np.ndenumerate(ds) ]

  hist = dsc.histogram(ds)
  histId = dsc.histogram(ds, lambda x:x)

  for (k,v) in hist.items():
    assert ds_flat.count(k) == v, "histogram[{}] = {}, but {} required".format(k,v, ds_flat.count(k))

  for (k,v) in histId.items():
    assert ds_flat.count(k)*k == v, "histogramId[{}] = {}, but {} required".format(k,v, ds_flat.count(k)*k)
