from gcn import *
from data_preprocessing import *
def test(loader, model, device):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            
            data.to(device)
            with torch.no_grad():
                out = model(data.x, data.edge_index, edge_weights = None, batch = data.batch)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                y = data.y
                correct += int((pred == y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


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

data = preprocess_ba_2motifs('BA-2motif', padded=False)
train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=64, val_split=0.1, test_split=0.1)

print('test_acc: ', test(val_loader, clf_model, device))

