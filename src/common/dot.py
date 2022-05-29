import numpy as np
import numpy.typing as npt
from collections import defaultdict

def norm(m: npt.ArrayLike, lb: float = 0, ub: float = 1) -> npt.NDArray:
  ret = np.ndarray(m.shape)
  np.copyto(ret, m)
  ret -= np.min(ret)
  ret *= (ub-lb) / np.max(ret)
  ret += lb
  return ret

def histogram(m : npt.NDArray, weight : callable = lambda x:1) -> defaultdict:
  hist = defaultdict(int)
  for (_,x) in np.ndenumerate(m):
    hist[x] += weight(x)
  return hist