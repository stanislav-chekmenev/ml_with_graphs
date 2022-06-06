# Train MLP: 
c_in = dataset.num_features
c_out = dataset.num_classes
mlp_model = MLP(c_in=c_in, c_out=c_out, c_hidden=16)
loss_function = torch.nn.CrossEntropyLoss()  # Define loss function.
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
 
def train():
    mlp_model.train()  # Set the model to the train mode
    optimizer.zero_grad()  # Clear gradients.
    out = mlp_model(data.x)  # Perform a single forward pass.
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test(mask):
    mlp_model.eval() # Set model to evaluation mode
    out = mlp_model(data.x)
    pred = out.argmax(dim=1)  # Use the class with the highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc
 
for epoch in range(1, 201):
    train()
    train_acc = test(data.train_mask)
    val_acc= test(data.val_mask)
    test_acc = test(data.test_mask)
    print(
     f'Epoch: {epoch:03d}, Train acc: {train_acc:.4f}, '
     f'Val acc: {val_acc:.4f}, Test acc: {test_acc:4f}'
    )
    
# Write GCN together   
def __init__(self, c_in, c_out, c_hidden):
    super().__init__()
    torch.manual_seed(12345)
    self.conv1 = GCNConv(c_in, c_hidden)
    self.conv2 = GCNConv(c_hidden, c_out)
    self.dropout = Dropout(p=0.5, inplace=False)

def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = ReLU()(x)
    x = self.dropout(x)
    return self.conv2(x, edge_index)
 
c_in = dataset.num_features
c_out = dataset.num_classes
model = GCN(c_in=c_in, c_out=c_out, c_hidden=16)
print(model)
