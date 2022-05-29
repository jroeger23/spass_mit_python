import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib
import cv2

def norm(m: npt.ArrayLike, lb: float = 0, ub: float = 1) -> npt.NDArray:
  ret = np.ndarray(m.shape)
  np.copyto(ret, m)
  ret -= np.min(ret)
  ret *= (ub-lb) / np.max(ret)
  ret += lb
  return ret


def sobelCartesian(img: npt.ArrayLike) -> npt.NDArray:
  if len(img.shape) != 2:
    raise ValueError("img.shape must be 2 dimensional, got {} dimensions".format(len(img.shape)))

  sobel_horz = np.array([1,2,1,
                        0,0,0,
                        -1,-2,-1]).reshape(3,3)
  sobel_vert = np.array([1,0,-1,
                        2,0,2,
                        1,0,-1]).reshape(3,3)

  sobel_horz = cv2.filter2D(img, -1, sobel_horz)
  sobel_vert = cv2.filter2D(img, -1, sobel_vert)

  return (sobel_horz, sobel_vert)


def sobelPolar(img: npt.ArrayLike) -> np.ndarray:
  (sobel_horz, sobel_vert) = sobelCartesian(img)

  img_gradient_theta = np.arctan2(sobel_horz, sobel_vert)
  img_gradient_r = np.sqrt(sobel_horz**2 + sobel_vert**2)

  return (img_gradient_theta, img_gradient_r)
