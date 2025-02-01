from typing import Any, Union

import pandas as pd
import networkx as nx
import numpy as np
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
    all_node_features: list[list[list[Union[float, int]]]] = []
    
    for _, row in tqdm(merged_df_exploded.iterrows()):
        adj_matrix = row['matrix']  
        timestamp = row['timestamp']

        G = nx.from_numpy_array(adj_matrix)

        # Centrality measures (handle NaN cases)
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = {node: val if not np.isnan(val) else 0.0 for node, val in nx.closeness_centrality(G).items()}
        betweenness_centrality = nx.betweenness_centrality(G)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = {node: 0.0 for node in G.nodes}
        
        pagerank = nx.pagerank(G, alpha=0.85)
        k_core = nx.core_number(G)
        degree = adj_matrix.sum(axis=1)

        failure_rates = calculate_failure_rates()
        
        mean_failure_rate_neighbors = {
            node: np.mean([failure_rates.get(n, 0.0) for n in G.neighbors(node)]) 
            if len(list(G.neighbors(node))) > 0 else 0.0
            for node in G.nodes
        }
        
        node_features_of_one_graph = []
        for node in G.nodes():
            node_features_of_one_node = [
                degree_centrality[node], 
                closeness_centrality[node], 
                degree[node], 
                timestamp,
                failure_rates.get(node, 0.0),
                betweenness_centrality[node], 
                eigenvector_centrality[node],
                pagerank[node], 
                k_core[node], 
                mean_failure_rate_neighbors[node]
            ]
            node_features_of_one_graph.append(node_features_of_one_node)
        
        all_node_features.append(node_features_of_one_graph)

    return all_node_features
