import argparse
import sys
import numpy as np
import models
import torch
# from pretrain_clf import * 
import GCN

sys.path.append('../')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

parser.add_argument('--z_dim', type=int, default=16, metavar='N', help='dimension of z')
parser.add_argument('--h_dim', type=int, default=16, metavar='N', help='dimension of h')
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--dataset', default='BA-2motif', help='dataset to use',
                    choices=['community', 'ogbg_molhiv', 'imdb_m'])
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')

parser.add_argument('--experiment_type', default='train', choices=['train', 'test', 'baseline'],
                    help='train: train CLEAR model; test: load CLEAR from file; baseline: run a baseline')
parser.add_argument
args = parser.parse_args()

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def train(decoder, explainer, optimizer_f, optimizer_cf train_loader, device):
    # edge embeddings go here -- not sure where this code goes, messed up when copy pasting --whoops 
    # edge embeddings = z, z_shape = (num_nodes, 20)

    """ (LATEER IF WE DO THIS)
    embedder = ...
    explainer = Explainer(embedder, z_dim, a_out_size, x_out_size)
    recons_a, recons_x, z_mu, z_logvar = explainer(x, edge_index, edge_weight, y_target)
    """

    for epoch in range(args.epochs):
        decoder.train()
        explainer.train()

        total_loss_f = 0
        total_loss_cf = 0

        for batch in train_loader:
            x, edge_index, edge_weight, y_target = batch.x.to(device), batch.edge_index.to(device), batch.edge_weight.to(device), batch.y.to(device)

            optimizer_f.zero_grad()
            optimizer_cf.zero_grad()

            # Forward pass through Factual explainer
            reconst = decoder(z)   

            # Loss for factual explainer
            loss_f = 

            total_loss_f.backward()
            optimizer_f.step()

            # Forward pass through the CF explainer
            recons_a, recons_x, z_mu, z_logvar = explainer(x, edge_index, edge_weight, y_target)

            # Loss for CF explainer
            loss_cf = 

            total_loss_cf.backward()
            optimizer_cf.step()

        print(f"Epoch {epoch + 1}/{args.epochs}, Factual Loss: {total_loss_f.item()}, CF Loss: {total_loss_cf.item()}")

        # validate on validation set ?

    print("Training complete!")


def run(args):
    dataset_name = args.dataset_name
    device = "cpu"
    """
    load data for train, val, test
    """
    data = preprocess_ba_2motifs(dataset_name, padded=False)
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=64, val_split=0.1, test_split=0.1)

    """
    Factual model: G -> Embedder -> h_node -> h_edge -> MLP(DECODER FOR FACTUAL) -> FACTUAL
    CF model: G -> Embedder -> h_node -> z_u / z_logvar -> sample -> decoder_x / decoder_a -> CF
    """
    # embedder
    embedder = GCN(num_node_features,2).to(device)              # load best model
    checkpoint = torch.load('best_model.pth')                   # load the model state dict
    embedder.load_state_dict(checkpoint['model_state_dict'])
    embedder.eval()                                             # set the model to evaluation mode

    # MLP (DECODER FOR FACTUAL)
    z_dim = 20              # ?
    output_size = 20        # num_features or num_edges ??
    decoder = Decoder(z_dim=z_dim, output_size=output_size)

    # initialize encoder for VAE
    x_dim = 20              # input dim = num_feartres
    h_dim = 20              # hidden dim = ??
    z_dim = 20              # latent space dim = ??
    encoder = GraphEncoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, embedder=model).to(device)

    # initialize explainer for VAE (encoder -> mu, logvar -> reparameterize -> z_sample -> decoder -> reconst)
    explainer = Explainer(encoder, z_dim, a_out_size, x_out_size).to(device)
    # explainer = Explainer(encoder, z_dim=args.z_dim, a_out_size=num_edges, x_out_size=num_node_features).to(device)

    optimizer = optim.Adam(explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(decoder, explainer, criterion, optimizer, train_loader, device)
    




    
    


     






    




run(args)

'''1. dataloader for mutag
        2. pretrain the graph classifier
        3. initialize factual and counterfactual models
            (use the same encoder network w 2 decoders)
        4. for each batch:
            forward pass, encode graph G to z_mu, z_logvar
            sample z with reparam. trick
            decode explanations a_f, a_cf
            encode a_cf to z_cf_mu, z_cf_logvar
            sample z_cf
            decode a_cf_f
            we have a_f, a_cf, a_cf_f

            loss = d(a_f, a_cf_f), I(y=1, a_cf_f), I(y=0, a_f), I(y=1, a_cf)??,
                KL(z_u_logvar - prior_z_logvar), KL(a, a_cf), 

            check proxy graph paper potentially for k_1 iterations on one loss etc., and k_2 on the other

            check clear for setting some loss terms to 0 in the beginning e_0 epochs

            backpropagate

                -add something about the validation set? test our metrics on data_val
                -possibly add perturbation stuff later?
                -how to make sure of the causal graph relevant w counterfactual
                -maybe construct a dataset or find a simple observable one (smth like shapes)
                -look at how to make things global, anchors/prototypes? rkhs (ugh)?, 
                                            unsup clustering a start? extract global for locals?'''
    


'''
    g, y_g
    y_cf = !y_g
    encoder = Encoder(x_dim , h_dim, z_dim)

    factExplainer = Explainer(encoder, z_dim, a_out_size, x_out_size)
    cfExplainer = Explainer(encoder, z_dim, a_out_size, x_out_size)

    factual_exp = factExplainer(x, edge_index, edge_weight, y_g)
    cf_exp = cfExplainer(x, edge_index, edge_weight, y_cf)
'''