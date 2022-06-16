import numpy as np

def normal2DCluster(clusters):
  train_data = []
  train_labels = []

  for (l, mean, cov, n) in clusters:
    train_data.append(np.random.multivariate_normal(mean, cov, size=n))
    train_labels.append(np.repeat(l, n))

  return (np.vstack(train_data), np.hstack(train_labels))


# r=a*e^(b*theta)
def logSpiral2D(a, b, num_class_samples=1000, revs=2, off=1):
  thetas = np.c_[np.logspace(1, np.log((revs*2+1)*np.pi), num_class_samples, base=np.e)]
  thetas = np.abs(thetas - thetas.max())

  rs = a * np.exp(b * thetas)
  rs = np.vstack((rs[::2]-a*off, rs, rs[::2]+a*off))

  thetas = np.vstack((thetas[::2], thetas, thetas[::2]))

  ys = np.sin(thetas) * rs
  xs = np.cos(thetas) * rs

  labels = [np.repeat(0, num_class_samples//2), np.repeat(1, num_class_samples), np.repeat(0, num_class_samples//2)]
  labels = np.hstack(labels)

  return (np.hstack((ys,xs)), labels)
