import torch
import torch.nn.functional as F
import numpy as np

def evalLabelProbability(net : torch.nn.Module, X : torch.Tensor):
  out = net(X)

  preds = np.squeeze(out.argmax(1).cpu().detach().numpy())

  return preds, np.array([F.softmax(sample, dim=0)[l].item() for l, sample in zip(preds, out)])