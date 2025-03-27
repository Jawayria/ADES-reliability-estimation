import torch
import torch.nn.functional as F

#from torch_geometric.data import Data
#from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, MessagePassing, global_mean_pool
from torch_geometric.nn import Set2Set

from torch_geometric.nn import GlobalAttention


class GAT_HEAD(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, num_heads=4):
        super(GAT_HEAD, self).__init__()
        self.num_heads = num_heads  # Store the number of heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)  # Specify heads for the first layer
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * num_heads)  # BatchNorm over all head outputs
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim//2, heads=num_heads) # Specify heads for the second layer
        self.bn2 = torch.nn.BatchNorm1d((hidden_dim//2) * num_heads)  # BatchNorm over all head outputs
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear((hidden_dim//2) * num_heads, output_dim)  # Linear layer input adjusted

    def forward(self, data):
        # Extract data components
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GATConv layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)  # Using ELU as it's common with multi-head attention

        # Second GATConv layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layer
        out = self.fc(x)
        return out