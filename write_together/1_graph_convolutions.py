# Forward pass of the GCN layer
def forward(self, node_feats, adj_matrix):
    """
    Inputs:
        node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
        adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                     Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                     Shape: [batch_size, num_nodes, num_nodes]
    """
<<<<<<< HEAD
    # Num neighbours = number of incoming edges
    num_neighbours = adj_matrix.sum(dim=-1, keepdims=True) # Sum across all columns but don't reduce the last dim.
=======
    # TODO: write the forward pass together.
    # Num neighbours = number of incoming edges
    num_neighbours = adj_matrix.sum(dim=-1, keepdims=True) # Sum across all columnsm but don't reduce the last dim.
>>>>>>> e4d97d5a249a1817d2a1941693e9ed286e77ea02
    node_feats = self.projection(node_feats) # Apply the weight matrix to the features to create a message.
    node_feats = torch.bmm(adj_matrix, node_feats) # A batch matrix multiplication. Check the update rule.
    node_feats = node_feats / num_neighbours # Averaging the messages
    return node_feats
 

# Forward pass of the GAT layer.
def forward(self, node_feats, adj_matrix, print_attn_probs=False):
    """
    Inputs:
        node_feats - Input features of the node. Shape: [batch_size, num_nodes, c_in]
        adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
        print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
    """
    batch_size, num_nodes = node_feats.size(0), node_feats.size(1) 

    # Apply the linear layer to get messages.
    node_feats = self.projection(node_feats)
    # Reshape/view node_feats so we could pass the same amount of node features
    # through each attention head
    node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

    # Calculate the attention logits for every edge in the adjacency matrix
    # It's very expensive to do with all combination of nodes
    # => Create a tensor of concatenated messages [W*h_i||W*h_j], taking the nodes i and j if there's an edge between them
    # By doing so we compute the attention weights only for the 1-hop neighbourhood of a node and masking the rest
    # Shape edges = [num_edges, 3], 3 is for a batch index, sending and receiving nodes indices
    edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges exist
    node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1) 
    # edges[:, 0] - batches idx, edges[:, 1] - sending nodes idx; edges[:, 2] - receiving nodes idx
    edge_indices_row = edges[:,0] * num_nodes + edges[:,1] # 1-D tensor of indices of sending nodes
    edge_indices_col = edges[:,0] * num_nodes + edges[:,2] # 1-D tensor of indices of receiving nodes
    a_input = torch.cat([
        torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0), # message of node i
        torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)  # message of node j
    ], dim=-1) # index_select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

    # Calculate attention MLP output (independent for each head)
    # Here, b is the total number of edges in the graph, h is the number of heads, c is the number of channels
    attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a) 
    attn_logits = self.leakyrelu(attn_logits)

    # Map list of attention values back into a matrix
    # new_zeros makes sure that the output tensor is of the same dtype and on the same device as attn_logits
    # Create an attention matrix of the shape of adj_matrix, but for each head -> add (self.num_heads, ) tuple
    attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15) # fill with -inf not to affect the softmax()
    attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1) # Map logits if an edge exists

    # Weighted average of attention
    attn_probs = torch.nn.functional.softmax(attn_matrix, dim=2) # Softmax along the receiving node dimension, formula
    if print_attn_probs:
        print("Attention probs\n", attn_probs.permute(0, 3, 1, 2)) # Permute to separate attn_probs for each node by head
    # Here, b is the batch dimension
    # Multiply each incoming message from a neighbour with the attention probability and then sum them. 
    # Check the formula, we iterate over the neighbouring (receiving) nodes.
    # attn_prob.shape = [batch, num_nodes, num_nodes, num_heads]
    # node_feats = [batch, num_nodes, num_heads, num_channels]
    node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

    # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
    if self.concat_heads:
        node_feats = node_feats.reshape(batch_size, num_nodes, -1)
    else:
        node_feats = node_feats.mean(dim=2)

    return node_feats
