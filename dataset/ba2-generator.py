import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif

dataset1 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=20, num_edges=1),
    motif_generator=HouseMotif(),
    num_motifs=1,
    num_graphs=500,
)

dataset2 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=20, num_edges=1),
    motif_generator=CycleMotif(5),
    num_motifs=1,
    num_graphs=500,
)

house_graphs = []
for data in dataset1:
    data
torch.save(dataset, 'ba2motif-generated.pt')

