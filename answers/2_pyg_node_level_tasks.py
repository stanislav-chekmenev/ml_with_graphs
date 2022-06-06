# Exercise 2.1
c_in = dataset.num_features
c_out = dataset.num_classes

mlp = MLP(c_in=c_in, c_out=c_out, c_hidden=256)
gcn = GCN(c_in=c_in, c_out=c_out, c_hidden=256)

mlp.eval()
gcn.eval()

out_mlp = mlp(data.x)
mlp_title = 'Untrained MLP embeddings'

out_gcn = gcn(data.x, data.edge_index)
gcn_title = 'Untrained GCN embeddings'

visualize_both(out_mlp, out_gcn, mlp_title, gcn_title, color=data.y)

# Exercise 2.2
# Train the GCN
c_in = dataset.num_features
c_out = dataset.num_classes
gcn_model = GCN(c_in=c_in, c_out=c_out, c_hidden=16)
loss_function = torch.nn.CrossEntropyLoss()  # Define loss function.
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

def train():
      gcn_model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = gcn_model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = loss_function(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test(mask):
      gcn_model.eval() # Set model to evalutation mode
      out = gcn_model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
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

# Visualize the trained MLP vs GCN
c_in = dataset.num_features
c_out = dataset.num_classes

mlp_model.eval()
gcn_model.eval()

out_mlp = mlp_model(data.x)
mlp_title = 'Trained MLP embeddings'

out_gcn = gcn_model(data.x, data.edge_index)
gcn_title = 'Trained GCN embeddings'

visualize_both(out_mlp, out_gcn, mlp_title, gcn_title, color=data.y)  
    

