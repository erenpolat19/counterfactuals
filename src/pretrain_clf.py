from gcn import *
from data_preprocessing import *



def train(model, criterion, optimizer, train_loader, val_loader, best_model, best_val_acc, device):
    model.train()

    best_epoch = 0
    epoch = 0
    for data in train_loader:  # Iterate in batches over the training dataset.  
        #print(data)          
        data.to(device)
        #print(data.edge_index.shape)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        #train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        #print('train_acc', train_acc, 'val_acc', val_acc)
        if val_acc > best_val_acc:
            best_model = model
            best_val_acc = val_acc
            best_epoch = epoch
        epoch = epoch + 1
    return best_model, best_epoch

def test(loader, model, device):
     with torch.no_grad():
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            y = data.y.squeeze().argmax()
            correct += int((pred == y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == '__main__':
    device = 'cpu'
    dataset_name = 'BA-2motif'
    data = preprocess_ba_2motifs(dataset_name, padded=False)
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=64, val_split=0.2, test_split=0.2)

    num_node_features = data[0].x.shape[1]
    #print(data[1])
    model = GCN(num_node_features,2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = None
    for epoch in range(1, 171):
        best_model, best_epoch = train(model, criterion, optimizer, train_loader, val_loader, best_model, best_val_acc, device)
        # train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        #test_acc = test(test_loader, model, data, device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    print('Final test' , test(test_loader, best_model, device), f'best epoch {best_epoch}')
