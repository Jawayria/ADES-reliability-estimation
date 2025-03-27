import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Set2Set

class GAT_LN_SET(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(GAT_LN_SET, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(normalized_shape=hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 2)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=hidden_dim // 2)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pool = Set2Set(hidden_dim // 2, processing_steps=2)  # Set2Set pooling
        self.fc = torch.nn.Linear(2 * (hidden_dim // 2), output_dim)  # Adjusted for Set2Set output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.pool(x, batch)  # Set2Set pooling

        return self.fc(x)
