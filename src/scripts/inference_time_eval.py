import os
import re

import numpy as np
import pandas as pd
import torch
from torch.ao.ns.fx.weight_utils import extract_weight_from_node
from torch_geometric.data import Data

from datamanip.dataset_parts_construction import construct_edge_indices
from datamanip.feature_extraction import extract_features_from_data
from filepath import dataset_path


def extract_config_id(path: str) -> int:
    """
    Extract the configuration id from the file path.
    For example, for "ctmc_s10.lab", it returns 10.
    """
    match = re.search(r's(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract config_id from path: {path}")

def read_matrix_file(path: str) -> list:
    """
    Read a single matrix file and return the matrix as a list of lists of ints.
    Assumes the file contains a string representation of a matrix with square brackets.
    """
    with open(path, 'r') as f:
        content = f.read()

    # Remove the first and last character ('[' and ']')
    content = content[1:-1]
    string_list = []
    current_row = ''
    start_line = False

    for c in content:
        if c == '[':
            start_line = True
        elif c == ']':
            start_line = False
            string_list.append(current_row)
            current_row = ''
        elif start_line:
            if c != '\n':
                current_row += c

    # Convert the string representation into a list of lists of ints
    matrix = []
    for string in string_list:
        # Replace '.' with a space, split by whitespace, and convert to int
        vals = string.replace('.', ' ').split()
        vals = [int(v) for v in vals]
        matrix.append(vals)
    return matrix

def read_reliability_file(path: str) -> pd.DataFrame:
    """
    Read a single reliability CSV file and return the DataFrame.
    Assumes the CSV uses semicolon as separator, comma as decimal, and has two columns:
    timestamp and reliability.
    """
    df = pd.read_csv(
        path,
        sep=';',
        decimal=',',
        names=['timestamp', 'reliability']
    )
    # Only keep data up to (and including) 1000 hours.
    df = df[df['timestamp'] == 1000]
    return df

def read_matrices_for_configs(matrix_directory: str, configs: list) -> pd.DataFrame:
    """
    For each configuration id in `configs`, build the file path for its matrix file,
    read the matrix, and return a DataFrame with columns 'matrix' and 'config_id'.
    """
    all_matrices = []
    for cfg in configs:
        path = os.path.join(matrix_directory, f"config_{cfg}.txt")
        matrix = read_matrix_file(path)
        matrix = np.array(matrix)
        all_matrices.append({"matrix": matrix, "config_id": cfg})
    return pd.DataFrame(all_matrices)

def read_reliability_for_configs(rel_directory: str, configs: list) -> pd.DataFrame:
    """
    For each configuration id in `configs`, build the file path for its reliability file,
    read the reliability data (only up to 1000 hours), and return a DataFrame with columns
    'rel_data' and 'config_id'.
    """
    all_rels = []
    for cfg in configs:
        path = os.path.join(rel_directory, f"config_{cfg}.csv")
        rel_data = read_reliability_file(path)
        all_rels.append({"rel_data": rel_data, "config_id": cfg})
    return pd.DataFrame(all_rels)

# --- Main usage ---
def calc_inference_times():
    # List of configuration IDs to read
    desired_configs = [10, 451, 695]

    # Directories where your files are stored.
    # Adjust these to your actual directories.

    # Read only the desired matrix and reliability files.
    matrices_path = os.path.join(dataset_path, "raw", "01_system", "matrix")
    reliabilities_path = os.path.join(dataset_path, "raw", "03_reliability", "hours")
    matrices_df = read_matrices_for_configs(matrices_path, desired_configs)
    reliability_df = read_reliability_for_configs(reliabilities_path, desired_configs)
    # Display the results.

    print("Matrices for configs 10, 451, and 695:")
    print(matrices_df)

    print("\nReliability values (up to 1000 hours) for configs 10, 451, and 695:")
    for idx, row in reliability_df.iterrows():
        print(f"\nConfig ID: {row['config_id']}")
        print(row['rel_data'])
    matrices_df['timestamp'] = 1000
    y_list = {10: 0.961912, 451: 0.999913, 695:0.999982}
    for cng in desired_configs:
        df = matrices_df[matrices_df['config_id'] == cng]
        node_features = torch.Tensor(extract_features_from_data(df))
        edge_indices = construct_edge_indices(df)
        data = Data(x=node_features[0], edge_index=edge_indices[0], y=y_list[cng])
        print(f"Data for config {cng}:")
        print(data)



if __name__ == "__main__":
    calc_inference_times()
