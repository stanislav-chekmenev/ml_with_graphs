# Write the update networks
class EdgeModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        num_features = dataset.num_features * 2 + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, dataset.num_edge_features),
            torch.nn.ReLU()
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.node_mlp = torch.nn.Sequential(
              torch.nn.Linear(num_features, hidden_channels),
              torch.nn.ReLU(),
              torch.nn.Dropout(0.5),
              torch.nn.Linear(hidden_channels, dataset.num_features),
              torch.nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes in all graphs of the batch.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        dest_node_idx = edge_index[1]
        # Average all attributes of incoming edges for each destination node
        edge_out_bar = scatter_mean(src=edge_attr, index=dest_node_idx, dim=0, dim_size=x.size(0))
        out = torch.cat([x, edge_out_bar, u[batch]], 1)
        return self.node_mlp(out), edge_out_bar

class GlobalModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.global_mlp = torch.nn.Sequential(
              torch.nn.Linear(num_features, hidden_channels),
              torch.nn.ReLU(),
              torch.nn.Dropout(0.5),
              torch.nn.Linear(hidden_channels, 1),
              torch.nn.ReLU()
        )

    def forward(self, node_attr_prime, edge_out_bar, u, batch):
        # node_attr_bar: [N, F_x], where N is the number of nodes in the batch.
        # edge_attr: [N, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # Average all node attributes for each graph, using batch tensor.
        node_attr_bar = scatter_mean(node_attr_prime, batch, dim=0)
        # Average all edge_out_bar attributes for each graph
        edge_attr_bar = scatter_mean(edge_out_bar, batch, dim=0)
        out = torch.cat([u, node_attr_bar, edge_attr_bar], dim=1)
        return self.global_mlp(out)
    

# Write the forward pass for the GN block
def forward(self, node_attr, edge_attr, u, edge_index, batch):
    # 1. Perform message passing to obtain node embeddings
    for _ in range(self.num_passes):
        src_node_idx = edge_index[0]
        dest_node_idx = edge_index[1]

        edge_attr = self.edge_model(
          node_attr[src_node_idx], node_attr[dest_node_idx], edge_attr, 
          u, batch[src_node_idx]
            
        )
        node_attr, edge_out_bar = self.node_model(
            node_attr, edge_index, edge_attr, u, batch
        )
        node_attr, edge_out_bar = self.node_model(
            node_attr, edge_index, edge_attr, u, batch
        )
        u = self.global_model(node_attr, edge_out_bar, u, batch)

    # 2. Readout layer
    graph_attr = torch.cat([node_attr, edge_out_bar, u[batch]], dim=1)
    graph_attr = global_mean_pool(graph_attr, batch)

    return self.lin(graph_attr)