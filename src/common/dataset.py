import torch

def selectRandomBatch(data, labels, n):

    assert len(data) == len(labels), "data and labels must have the same length"

    perm = torch.randperm(len(data))[:n]
    return data[perm], labels[perm]