class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second GCN layer
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Fully connected layer
        
    def forward(self, data):
        # Extract data components
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = x.unsqueeze(1)

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if DROPOUT:
          x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling to aggregate node-level features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph-level classification/regression
        out = self.fc(x)
        return out