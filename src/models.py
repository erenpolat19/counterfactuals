import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv,global_max_pool, global_mean_pool
import torch.nn.functional as F

class GraphModel(nn.Module):
    def __init__(self, x_dim, h_dim):
        self.conv1 = GCNConv(x_dim, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.conv3 = GCNConv(512, h_dim)

    def forward(self, x, edge_index, edge_weight):
    
        out1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = F.relu(out1)

        out2 = self.conv2(out1, edge_index, edge_weight=edge_weight)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = F.relu(out2)

        out3 = self.conv3(out2, edge_index, edge_weight=edge_weight)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = F.relu(out3)

        return out3  

class GraphEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, embedder=None):
        super(GraphEncoder, self).__init__()
        if embedder == None:
            embedder = GraphModel(x_dim, h_dim)
        self.embedder = embedder
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out
    
    def forward(self, x, edge_index, edge_weight, y_target): 
        graph_emb = self.graph_pooling(self.embedder(x, edge_index, edge_weight))
        
        z_mu = self.encoder_mean(torch.cat((graph_emb, y_target), dim=1))
        z_logvar = self.encoder_var(torch.cat((graph_emb, y_target), dim=1))

        return z_mu,z_logvar
    
class Decoder(nn.Module):
    def __init__(self, z_dim, output_size):
    
        self.decoder = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.ReLU(),
            nn.Linear(z_dim * 2, output_size)
        )
    def forward(self, z):
        return torch.sigmoid(self.decoder(z))

class Explainer(nn.Module):
    def __init__(self, encoder, z_dim, a_out_size, x_out_size):
        self.encoder = encoder
        self.decoder_a = Decoder(z_dim, a_out_size)
        self.decoder_x = Decoder(z_dim, x_out_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def decode(self ,z_sample):
        return self.decoder_a(z_sample), self.decoder_x(z_sample)
    
    def forward(self, x, edge_index, edge_weight, y_target, beta=1):
        mu, logvar = self.encoder(x, edge_index, edge_weight, y_target)
        z_sample = self.reparameterize(mu, beta * logvar)

        recons_a, recons_x = self.decode(z_sample)