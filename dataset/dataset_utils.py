
import os
import numpy as np
import pickle
from numpy.random import RandomState

def adj_to_edge_index(adj):
    """
    Convert an adjacency matrix to an edge index
    :param adj: Original adjacency matrix
    :return: Edge index representation of the graphs
    """
    converted = []
    for d in adj:
        edge_index = np.argwhere(d > 0.).T
        mask = edge_index[0] != edge_index[1]
        converted.append(edge_index[:, mask])

    return converted


def load_graph_dataset(dataset_name, shuffle=True):
    """Load and optionally shuffle a graph dataset.
    
    Parameters:
    - dataset_name (str): Name of the dataset to load.
    - shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
    Tuple containing edge indices, features, labels, and masks for training, validation, and testing.
    
    Raises:
    FileNotFoundError: If the dataset pickle file does not exist and cannot be created.
    NotImplementedError: If the dataset is unknown.
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dir_path, dataset_name, 'raw', f'{dataset_name}.pkl') 
    #dataset_path = os.path.join(dir_path, dataset_name, 'processed', 'data.pt') 
    with open(dataset_path, 'rb') as fin:
        adjs, features, labels = pickle.load(fin)
    print(adjs)

    n_graphs = adjs.shape[0]
    indices = np.arange(n_graphs)
    if shuffle:
        prng = RandomState(42) 
        indices = prng.permutation(indices)


    adjs = adjs[indices]
    features = features[indices].astype('float32')
    labels = labels[indices]

    n_train = int(n_graphs * 0.8)
    n_val = int(n_graphs * 0.9)
    train_mask = np.zeros(n_graphs, dtype=bool)
    val_mask = np.zeros(n_graphs, dtype=bool)
    test_mask = np.zeros(n_graphs, dtype=bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_val] = True
    test_mask[n_val:] = True

    edge_index = adj_to_edge_index(adjs)

    print(adjs.shape)
    print(features.shape)
    #return edge_index, features, labels, train_mask, val_mask, test_mask
dataset_name = 'BA-2motif'
load_graph_dataset(dataset_name)