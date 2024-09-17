import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class GCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.h_dim = 0
        self.conv1 = GCNConv(num_features, self.h_dim)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(self.h_dim, self.h_dim)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(self.h_dim, self.h_dim)
        self.relu3 = ReLU()
        self.lin = Linear(self.h_dim * 2, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)

        out = self.lin(out1)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
            #edge_weights = torch.ones(edge_index.size(1)).cuda()

        stack = []
        x = x.float()
        out1 = self.conv1(x, edge_index, edge_weights)
        
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin

class GCN_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_3layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_max_pool(x, batch)

        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x