
import numpy as np
import torch
import pickle as pkl
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data
import scipy.sparse as sp
from dataloader import get_dataloaders

"""
Loads the BA-2motifs dataset from a pickle file.

Args:
    dataset (str): Name of the dataset to be loaded. 

Returns:
    (adjs, features, labels) (tuple): A tuple containing three elements:
        - adjs (list of numpy.ndarray): A list of adjacency matrices for each graph in the dataset.
        - features (list of numpy.ndarray): A list of feature matrices corresponding to each graph.
        - labels (list of numpy.ndarray): A list of labels associated with each graph.
"""
def load_ba_2motifs(dataset):
    with open('../dataset/BA-2motif/raw/BA-2motif.pkl', 'rb') as fin:
        adjs, features, labels = pkl.load(fin)
    return adjs, features, labels

"""
Preprocess the BA-2motifs dataset.

Args: 
    dataset (str): Name of the dataset to be loaded. 
    padded (bool): Set to True --> padding, False --> no padding.
    save_falg (bool): Saves data if flag is set.

Returns:
    data (Pytorch Geometric Data object): Preprocessed data, also saved to .pt file.
"""
def preprocess_ba_2motifs(dataset, padded=False, save_flag=True):
    adjs, features, labels = load_ba_2motifs(dataset)
    
    # Define max number of nodes
    max_num_nodes = 30
    
    adj_all = []            # List to store adjacency matrices
    features_all = []       # List to store feature matrices
    labels_all = []         # List to store labels
    
    for i in range(len(adjs)):
        adj = adjs[i]
        feature = features[i]
        label = labels[i]
        
        # Skip graphs with more than max num nodes
        if adj.shape[0] > max_num_nodes:
            continue
        
        if padded:
            # Pad the adjacency matrix
            if adj.shape[0] < max_num_nodes:
                padded_adj = np.zeros((max_num_nodes, max_num_nodes))
                padded_adj[:adj.shape[0], :adj.shape[1]] = adj
                adj = padded_adj
        
            # Pad the feature matrix
            if feature.shape[0] < max_num_nodes:
                padded_features = np.zeros((max_num_nodes, feature.shape[1]))
                padded_features[:feature.shape[0], :] = feature
                feature = padded_features
        
        # Convert adjacency matrix to edge_index
        adj_sparse = sp.coo_matrix(adj)  
        edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
        
        # Create PyTorch Geometric Data object
        data = Data(x=torch.tensor(feature, dtype=torch.float), 
                    edge_index=edge_index,
                    y=torch.tensor(label, dtype=torch.float))
        
        adj_all.append(adj)
        features_all.append(feature)
        labels_all.append(label)
        
    # Save the processed data
    if save_flag:
        path_save = '../dataset/BA-2motif/processed/ba2motifs.pt'
        torch.save({'data': [Data(x=torch.tensor(features[i], dtype=torch.float), 
                                  edge_index=from_scipy_sparse_matrix(sp.coo_matrix(adjs[i]))[0],
                                  y=torch.tensor(labels[i], dtype=torch.float)) 
                              for i in range(len(adjs))]}, path_save)
        print('Saved data:', path_save)
    
    return [Data(x=torch.tensor(features[i], dtype=torch.float), 
                 edge_index=from_scipy_sparse_matrix(sp.coo_matrix(adjs[i]))[0],
                 y=torch.tensor(labels[i], dtype=torch.float)) 
            for i in range(len(adjs))]

if __name__ == '__main__':
    dataset = 'ba2motifs'
    data = preprocess_ba_2motifs(dataset, padded=False)
    train_loader, val_loader, test_loader = get_dataloaders(data)
    

