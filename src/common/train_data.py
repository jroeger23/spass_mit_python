import numpy as np

def normal2DCluster(clusters):
  train_data = []
  train_labels = []

  for (l, mean, cov, n) in clusters:
    train_data.append(np.random.multivariate_normal(mean, cov, size=n))
    train_labels.append(np.repeat(l, n))

  return (np.vstack(train_data), np.hstack(train_labels))