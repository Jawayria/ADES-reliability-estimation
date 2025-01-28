from typing import Any, Union

import pandas as pd
import networkx as nx
from tqdm import tqdm
from topology_parameters import prefix_to_type, type_failure_rates
from filepath import configs_list_path


def calculate_failure_rates() -> dict[int, float]:
    # Initialize the dictionary to store the index to type mapping
    index_to_type = {}

    # Read the first line from 'configs.txt'
    with open(configs_list_path, 'r') as file:
        first_line = file.readline().strip()

    # Extract the part after 'config_0: [' and before the closing ']'
    start = first_line.find('[')
    end = first_line.find(']')
    node_list_str = first_line[start + 1:end]

    # Split the node names by comma and strip whitespace
    node_names = [node.strip() for node in node_list_str.split(',')]

    # Iterate over the node names and map index to type
    for index, node_name in enumerate(node_names):
        if node_name.startswith('Sw'):
            node_type = 'switch'
        else:
            # Use the first character as prefix
            prefix = node_name[0]
            node_type = prefix_to_type.get(prefix, 'unknown')
        # Map the index to the node type
        index_to_type[index] = node_type

    index_to_failure_rate = {}
    for index, node_type in index_to_type.items():
        failure_rate = type_failure_rates.get(node_type, 0)
        index_to_failure_rate[index] = failure_rate

    return index_to_failure_rate


def extract_features_from_data(merged_df_exploded: pd.DataFrame) -> list[list[list[Union[float, int]]]]:
    # Create node features for the GNN

    all_node_features: list[list[list[Union[float, int]]]] = []
    for _, row in tqdm(merged_df_exploded.iterrows()):

        adj_matrix = row['matrix']  # Matrix of the graph
        timestamp = row['timestamp']  # Timestamp when reliability was measured

        # Get indegree and centrality using NetworksX graph
        G = nx.from_numpy_array(adj_matrix)

        # Calculate centrality measures
        degree_centrality: dict[Any, float] = nx.degree_centrality(G)
        closeness_centrality: dict[Any, float] = nx.closeness_centrality(G)
        degree = adj_matrix.sum(axis=1)
        # Calculate failure rates based on given topology parameters
        failure_rates = calculate_failure_rates()
        node_features_of_one_graph = []
        # Assign features to each node
        for node in G.nodes():
            node_features_of_one_node = [degree_centrality[node], closeness_centrality[node], degree[node], timestamp,
                                         failure_rates[node]]
            node_features_of_one_graph.append(node_features_of_one_node)
        all_node_features.append(node_features_of_one_graph)
    return all_node_features
