class SimpleMPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMPNN, self).__init__(aggr='add')  # 'add' aggregation (sum messages)
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.final_linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Apply initial linear transformation to node features


        x = self.linear(x)


        # Message passing
        x = self.propagate(edge_index, x=x)

        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)

        # Final layer for output
        x = self.final_linear(x)
        return x

    def message(self, x_j):
        # The simplest message function: use neighbor node features directly
        return x_j

    def update(self, aggr_out):
        # Apply a non-linearity to the aggregated output
        return F.relu(aggr_out)