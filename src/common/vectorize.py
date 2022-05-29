import numpy as np
import numpy.typing as npt


def one_hot(m : np.ndarray, n_classes : int) -> npt.NDArray:
  base = np.diagflat(np.ones(n_classes))
  return base[m]
