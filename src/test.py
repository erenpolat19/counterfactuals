from gcn import *
from data_preprocessing import *

device = "cpu"
params = {}
# params
params['x_dim'] = 10
params['num_classes'] = 2
# embedder
clf_model = GCN(params['x_dim'], params['num_classes']).to(device)              # load best model

# Load the saved state dictionary
checkpoint = torch.load('clf.pth')

# Load the weights into the model
clf_model.load_state_dict(checkpoint)
clf_model.eval()                          

data = preprocess_ba_2motifs(dataset_name, padded=False)
train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=64, val_split=0.1, test_split=0.1)