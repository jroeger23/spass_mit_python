import numpy as np
import numpy.typing as npt
import cv2
from collections import defaultdict

def histogram(m : npt.NDArray, weight : callable = lambda x:1) -> defaultdict:
  hist = defaultdict(int)
  for (_,x) in np.ndenumerate(m):
    hist[x] += weight(x)
  return hist


