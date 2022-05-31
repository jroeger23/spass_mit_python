import matplotlib as plt
import numpy as np


def sample2DClassifier(img, classifier : callable, cmap, y0, y1, x0, x1, resolution = 100):
  my,mx = np.meshgrid(np.linspace(y0, y1, resolution), np.linspace(x0, x1, resolution))
  sample_x = np.c_[mx.flatten()]
  sample_y = np.c_[my.flatten()]
  sample_data = np.hstack([sample_y, sample_x])

  classified = classifier(sample_data)
  classified_mesh = classified.reshape(resolution,resolution)
  color_mesh = np.flip(cmap(classified_mesh / 4), axis=0)

  img.set(data=color_mesh, extent=(y0, y1, x0, x1))