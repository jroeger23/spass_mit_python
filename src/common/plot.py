import imp
import matplotlib as plt
import numpy as np
import networkx as nx


def sample2DClassifier(img, classifier : callable, cmap, y0, y1, x0, x1, resolution = 100):
  my,mx = np.meshgrid(np.linspace(y0, y1, resolution), np.linspace(x0, x1, resolution))
  sample_x = np.c_[mx.flatten()]
  sample_y = np.c_[my.flatten()]
  sample_data = np.hstack([sample_y, sample_x])

  classified = classifier(sample_data)
  classified_mesh = classified.reshape(resolution,resolution)
  color_mesh = np.flip(cmap(classified_mesh / 4), axis=1).transpose(1,0,2)

  img.set(data=color_mesh, extent=(x0, x1, y0, y1))


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
  widths = np.log1p(np.abs(widths))/2
  widths = list(widths)

  nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_color=n_colors)
  nx.draw_networkx_labels(g, ax=ax, pos=pos, labels=n_labels)
  nx.draw_networkx_edges(g, ax=ax, pos=pos, edge_color=e_colors, width=widths)
  if show_weight:
    nx.draw_networkx_edge_labels(g, ax=ax, pos=pos, edge_labels=e_labels)