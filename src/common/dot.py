import numpy as np
import numpy.typing as npt

def norm(m: npt.ArrayLike, lb: float = 0, ub: float = 1) -> npt.NDArray:
  ret = np.ndarray(m.shape)
  np.copyto(ret, m)
  ret -= np.min(ret)
  ret *= (ub-lb) / np.max(ret)
  ret += lb
  return ret