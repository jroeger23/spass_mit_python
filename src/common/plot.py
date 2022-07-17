from tkinter import W
from src.common.dot import norm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from src.common.eval import evalLabelProbability


def sample2DClassifier(img, classifier : callable, colorize : callable,
                       cmap, y0, y1, x0, x1, resolution = 100):
  my,mx = np.meshgrid(np.linspace(y0, y1, resolution), np.linspace(x0, x1, resolution))
  sample_x = np.c_[mx.flatten()]
  sample_y = np.c_[my.flatten()]
  sample_data = np.hstack([sample_y, sample_x])

  classified = classifier(sample_data)
  classified_mesh = classified.reshape(resolution,resolution)
  color_mesh = np.flip(colorize(classified_mesh), axis=1).transpose(1,0,2)

  img.set(data=color_mesh, extent=(x0, x1, y0, y1))


def plotMultiWeightLayer(ax, weights_list, dx : float, dy : float, colorize):
  n_i0,n_o0 = weights_list[0].shape
  n_in,n_on = weights_list[-1].shape

  c_input = plt.colors.to_rgba('royalblue')
  c_bias = plt.colors.to_rgba('seagreen')
  c_hidden = plt.colors.to_rgba('lightgray')

  norm_weights = lambda w: np.sign(w) * norm(np.log1p(np.abs(w)), lb=0.05, ub=2.2)

  # gather adj matrix, node positions, node labels and node colors
  adj = np.hstack((np.zeros((n_i0,n_i0+1)), norm_weights(weights_list[0])))
  n_pos = [(0.2*dx, -(n_i0/2)*dy)] + [(0, (y-(n_i0/2))*dy) for y in range(1, n_i0)]
  n_labels = ['B'] + [f'$i_{x}$' for x in range(0, n_i0-1)]
  n_colors = [c_bias] + [c_input]*(n_i0-1)

  for ix, weights in list(enumerate(weights_list))[1:]:
    n_i, n_o = weights.shape
    y1,x1 = adj.shape

    # extend adjacency matrix
    corr = 0 if ix == (len(weights_list)-1) else 1 # do not add bias space for last layer
    adj = np.pad(adj, ((0,n_i), (0,n_o+corr)), 'constant', constant_values=(0,0))
    y2,x2 = adj.shape
    adj[y1:y2,x1+corr:x2] = norm_weights(weights)

    n_pos += [(dx*(ix+0.2),-(n_i/2)*dy)] + [(dx*ix,(y-(n_i/2))*dy) for y in range(1,n_i)]
    n_labels += ['B'] + ['']*(n_i-1)
    n_colors += [c_bias] + [c_hidden]*(n_i-1)

  # finalize
  a_h, a_w = adj.shape
  adj = np.pad(adj, ((0,a_w-a_h), (0,0)), 'constant', constant_values=(0,0))
  n_pos += [(dx*len(weights_list),(y-(n_on/2))*dy) for y in range(0, n_on)]
  n_pos = dict(enumerate(n_pos))
  n_labels += [f'$o_{x}$' for x in range(0,weights_list[-1].shape[1])]
  n_labels = dict(enumerate(n_labels))
  n_colors += [colorize(x) for x in range(0,weights_list[-1].shape[1])]

  g = nx.from_numpy_array(adj, create_using=nx.Graph)

  # colorize edges
  e_colors = ['b' if d['weight'] < 0 else 'r' for u,v,d in g.edges.data()]
  widths = list(nx.get_edge_attributes(g, 'weight').values())

  # draw
  nx.draw_networkx_nodes(g, ax=ax, pos=n_pos, node_color=n_colors)
  nx.draw_networkx_labels(g, ax=ax, pos=n_pos, labels=n_labels)
  nx.draw_networkx_edges(g, ax=ax, pos=n_pos, edge_color=e_colors, width=widths)

def plotWeightLayer(ax, weights : np.ndarray, dx : float, dy : float, cmap, show_weight : bool = False):
  n_i, n_o = weights.shape

  adj = np.hstack([np.zeros((n_i, n_i)), weights])
  adj = np.vstack([adj, np.zeros((n_o, n_i+n_o))])

  in_pos = [(0, (dy/3,0))] + [(x, (0,x*dx)) for x in range(1,n_i)]
  out_pos = [(x, (dy,(x-n_i)*dx)) for x in range(n_i, n_i+n_o)]
  pos = dict(in_pos + out_pos)

  g = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)

  n_colors = [plt.colors.to_rgba('#888888')]
  n_colors += [plt.colors.to_rgba('#DDDDDD')] * (n_i-1)
  n_colors += map(lambda x : cmap(x/4), range(0, n_o))

  e_colors = ['b' if d['weight'] < 0 else 'r' for u,v,d in g.edges.data()]

  n_labels = ['B']
  n_labels += [f'$i_{x}$' for x in range(0,n_i-1)]
  n_labels += [f'$o_{x}$' for x in range(0,n_o)]
  n_labels = dict(zip(range(0,n_i+n_o), n_labels))

  e_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in g.edges(data=True)])

  widths = nx.get_edge_attributes(g, 'weight')
  widths = np.array(list(widths.values()))
  widths = norm(np.log1p(np.abs(widths)), lb=0, ub=3)
  widths = list(widths)

  nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_color=n_colors)
  nx.draw_networkx_labels(g, ax=ax, pos=pos, labels=n_labels)
  nx.draw_networkx_edges(g, ax=ax, pos=pos, edge_color=e_colors, width=widths)
  if show_weight:
    nx.draw_networkx_edge_labels(g, ax=ax, pos=pos, edge_labels=e_labels)



def plotImagesProb(net, images, labels, labels_map):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = evalLabelProbability(net, images)
    images = images.cpu().detach()
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(2.5*len(images),3), dpi=100)
    for idx, (img, label, pred, prob) in enumerate(zip(images, labels, preds, probs)):
        ax = fig.add_subplot(1, len(images), idx+1, xticks=[], yticks=[])
        ax.imshow(img.permute(1,2,0))
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            labels_map[pred],
            prob * 100.0,
            labels_map[label.item()]),
                    color=("green" if pred==label.item() else "red"))
    return fig