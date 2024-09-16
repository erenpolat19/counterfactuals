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

'''
    g, y_g
    y_cf = !y_g
    encoder = Encoder(x_dim , h_dim, z_dim)

    factExplainer = Explainer(encoder, z_dim, a_out_size, x_out_size)
    cfExplainer = Explainer(encoder, z_dim, a_out_size, x_out_size)

    factual_exp = factExplainer(x, edge_index, edge_weight, y_g)
    cf_exp = cfExplainer(x, edge_index, edge_weight, y_cf)

'''
        


class PROXYExplainer(BaseExplainer):
    def __init__(self, model_to_explain, graphs, features, device='cpu',epochs=30, lr=0.003, temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features,device)
        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.expl_embedding = self.model_to_explain.embedding_size * 2

    def _create_explainer_input(self, pair, embeds):
        row_embeds = embeds[pair[0]]
        col_embeds = embeds[pair[1]]
        input_expl = torch.cat([row_embeds, col_embeds], dim=1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size(),device=self.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def loss_function(self, recon_x, x, mu, logvar, batch_weight_tensor):
        recon_x = batch_weight_tensor * recon_x
        recon_loss= F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def _loss(self, pred, target, mask, reg_coefs):
        
        scale=0.99
        mask = mask*(2*scale-1.0)+(1.0-scale)
        
        cce_loss = F.cross_entropy(pred, target)
        size_loss = torch.sum(mask) * reg_coefs[0]
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = reg_coefs[1] * torch.mean(mask_ent_reg)
        
        return cce_loss + size_loss + mask_ent_loss
    
    def prepare(self, indices=None):
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.train(indices or range(len(self.graphs)))

    def train(self, indices=None):
        
        self.explainer_model.train()
        
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))
        transformer = GraphTransformer(self.graphs, self.features)
        
        edgelists = []
        offset = 0
        batch_features = []
        offsets = []
        node_indicator = [] 
        edge_num = []
        
        self.graphs_new = [self.graphs[i] for i in indices]
        if len(self.graphs_new) > 0:
            self.reg_coefs[0] /= len(self.graphs_new)

        for id, i in enumerate(indices):
            edge_list,nmb_node,feature = transformer.renew_graph(self.graphs[i],self.features[i])
            edge_num.append(edge_list.shape[1])
            node_indicator.append(torch.tensor([id]*nmb_node))
            edgelists.append(edge_list+offset)
            offsets.append(offset)
            batch_features.append(feature)
            offset+=int(nmb_node)

        batch_features_tensor_ori = torch.concat(batch_features,0).to(self.device)
        batch_edge_list_ori = torch.concat(edgelists,-1).to(self.device)
        all_one_edge_weights = torch.ones(batch_edge_list_ori.size(1)).to(self.device)
        
        with torch.no_grad():
            embeds = self.model_to_explain.embedding(batch_features_tensor_ori,batch_edge_list_ori,all_one_edge_weights)

        newedgelists, unionNodes, unionFeature = transformer.process_graphs(indices)
        nmb_node = unionNodes.shape[0]
        feature = torch.tensor(unionFeature,dtype=torch.float32)

        self.vae = GNN_MLP_VariationalAutoEncoder(feature.shape[1], feature.shape[0]*feature.shape[0]).to(self.device)
        self.vae.train()
        optimizer_vae = Adam(self.vae.parameters(), lr=1e-4) 

        labels = []
        edgelists = []
        offset = 0
        batch_features = []
        offsets = []
        node_indicator = []
        weight_tensors = []
        vis_edge_list = []
        
        for i, edge_list in enumerate(newedgelists):
            vis_edge_list.append(edge_list)
            node_indicator.append(torch.tensor([i]*nmb_node))
            edgelists.append(edge_list + offset)
            offsets.append(offset)
            batch_features.append(feature)

            offset += int(nmb_node)

            sparseMatrix = csr_matrix((torch.ones(edge_list.shape[1]), edge_list), 
                            shape = (nmb_node,nmb_node))
            label = sparseMatrix.todense()
            label = torch.tensor(label,dtype=torch.float32)
            label = label.view(-1)
            weight_mask = (label == 1)
            labels.append(label)

            nodes = []
            for i in edge_list:
                for j in i:
                    nodes.append(j)
            nodes = list(set(nodes))

            weight_tensor = torch.ones(weight_mask.size(0))

            weight_mask = torch.zeros((nmb_node, nmb_node))
            for i in nodes:
                for j in nodes:
                    weight_mask[i,j] = 1.0
            weight_tensor *= weight_mask.view(-1)
            weight_tensors.append(weight_tensor)

        
        batch_weight_tensor = torch.stack(weight_tensors).to(self.device)
        batch_label = torch.stack(labels).to(self.device)
        batch_features_tensor = torch.concat(batch_features,0).to(self.device)
        batch_edge_list = torch.concat(edgelists,-1).to(self.device) 
        all_one_edge_weights = torch.ones(batch_edge_list.size(1)).to(self.device)
        node_indicator_tensor = torch.concat(node_indicator,-1).to(self.device)
        
        original_pred  = self.model_to_explain(batch_features_tensor, 
                                batch_edge_list,
                                batch=node_indicator_tensor, 
                                edge_weights=all_one_edge_weights)  


        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            t = temp_schedule(e)
            input_expl = self._create_explainer_input(batch_edge_list_ori, embeds).unsqueeze(0)
            sampling_weights = self.explainer_model(input_expl)
            mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
            edge_weight_delta = 1.0 - mask
            
            vae_epoch = 10
            for a in range(vae_epoch):
                optimizer_vae.zero_grad()
                mask_used = mask.detach() if a < vae_epoch - 1 else mask
                edge_weight_delta_used = edge_weight_delta.detach() if a < vae_epoch - 1 else edge_weight_delta

                recon_batch_mask, _, _ = self.vae((batch_features_tensor, batch_edge_list, mask_used), beta=0, batch=node_indicator_tensor)
                recon_batch, mu, logvar = self.vae((batch_features_tensor, batch_edge_list, edge_weight_delta_used), beta=1, batch=node_indicator_tensor)
                aug_mask = torch.max(recon_batch_mask, recon_batch)
                
                loss_vae = self.loss_function(aug_mask, batch_label, mu, logvar, batch_weight_tensor)
                loss_vae.backward(retain_graph=(a == vae_epoch - 1))
                
                optimizer_vae.step()


            aug_edge_list = []
            aug_edge_weights = []
            offset_new = 0
            
            for i in range(aug_mask.shape[0]):
                adj_matrix = aug_mask[i].reshape(nmb_node, nmb_node)
                edge_list = torch.nonzero(adj_matrix)
                edge_weights = adj_matrix[edge_list[:, 0], edge_list[:, 1]]

                edge_list = edge_list + offset_new
                aug_edge_list.append(edge_list.T)
                aug_edge_weights.append(edge_weights)
                offset_new += nmb_node
                
            aug_edge_list = torch.concat(aug_edge_list,-1).to(self.device) 
            aug_edge_weights = torch.concat(aug_edge_weights,-1).to(self.device)

            masked_pred = self.model_to_explain(batch_features_tensor,
                                                aug_edge_list, 
                                                batch=node_indicator_tensor,
                                                edge_weights=aug_edge_weights) 
            
            loss = self._loss(masked_pred,torch.argmax(original_pred,-1), mask, self.reg_coefs) 
            loss.backward()
            optimizer.step()
        
 
    def explain(self, index):
        index = int(index)
        
        feats = self.features[index].clone().detach().to(self.device)
        graph = self.graphs[index].clone().detach()
        all_one_edge_weights = torch.ones(graph.size(1)).to(self.device)
        embeds = self.model_to_explain.embedding(feats, graph, all_one_edge_weights).detach()

        input_expl = self._create_explainer_input(graph, embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights