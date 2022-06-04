# Exercise 1.1
node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]]])

gcn_layer = GCNLayer(c_in=2, c_out=2)
gcn_layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
gcn_layer.projection.bias.data = torch.Tensor([0., 0.])

with torch.no_grad():
    out_feats = gcn_layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)

# Exercise 1.2
a = torch.arange(8).view(2, 2, 2)
b = torch.arange(12).view(2, 2, 3)
print('A: ', a)
print('B: ', b)
torch.einsum('ijk,ikl->ijl', a, b)


# Exercise 1.3

# Please, do not reuse the code from the GATLayer 'forward()' method, but you're free to look at it, do experiments,
# use pen and paper if needed and check the outputs of the different parts of the method. 
# It helps to print the result and its shape after every step. Feel free to do it.

# Step 1: Check node_feats tensor and its shape and write down the node_feats_flat tensor. 
# Make sure you understand the dimensions of the tensor.
node_feats_flat = torch.arange(8).view(4, 2, 1)
print('node_feats_flat: \n', node_feats_flat)
print('shape of node_feats_flat: \n', node_feats_flat.shape)
print('----------------------------')

# Step 2: Write down the tensor of the incoming edges to node 1. Don't forget the edge to itself.
# HINT: Edges are the indices of the adj_matrix, keep the correct shape.
edges1 = torch.Tensor([0, 0, 0, 0, 0, 1]).view(2, 3)
print('edges1: \n', edges1)
print('shape of edges1: \n', edges1.shape)
print('----------------------------')

# Step 3: With the help of edges1 tensor compose a tensor message1, selecting the messages 
# from node_feats_flat that the 1st node receives and concatenating them. Do it manually by hand.
# Keep both heads here for now. Make sure you understand the dimensions!
message1 = torch.Tensor([[[0, 0], [1, 1]], [[0, 2], [1, 3]]])
print('message1: \n', message1)
print('shape of message1: \n', message1.shape)
print('----------------------------')

# Step 4: Exclude the concatenated messages received by the 2nd head and create a tensor message1_head1 with
# the messages only received by the 1st head.
message1_head1 = torch.Tensor([[0, 0], [0, 2]])
print('message1_head1: \n', message1_head1)
print('shape of message1_head1: \n', message1_head1.shape)
print('----------------------------')
# Step 5: Understand the layer.a MLP weight matrix and take the row corresponding to the 1st head.
a1 = torch.Tensor([-0.2, 0.3]).view(1, 2)
print('a1: \n', a1)
print('shape of a1: \n', a1.shape)
print('----------------------------')

# Step 6: Multiply the tensors message1_head1 and a1 to get the attention logits for the 1st node.
# HINT: Do matrix-vector multiplication and keep track of the dimensions. You will
# need to use transpose operation.
attn_logits1 = message1_head1 @ a1.transpose(1, 0)
print('attn_logits1: \n', attn_logits1)
print('shape of attn_logits1: \n', attn_logits1.shape)
print('----------------------------')

# Step 7: Apply softmax to the attn_logits1 and compare to the result to the 
# 1st row of the first head of the attn_logits tensor above.
# HINT: Feel free to use either the softmax function from torch or just the formula from the internet.
attn_probs1 = torch.exp(attn_logits1) / torch.sum(torch.exp(attn_logits1))
print('attn_probs1: \n', attn_probs1)
print('shape of attn_probs1: \n', attn_probs1.shape)
