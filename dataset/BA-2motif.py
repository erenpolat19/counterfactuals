import torch
from torch_geometric.datasets import BA2MotifDataset

# Trigger dataset reprocessing if necessary (optional, depends on changes)
dataset = BA2MotifDataset(root='BA2Motif/')

for data in dataset:
    print(data)
