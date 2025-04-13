import os
import re
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from datamanip.datasetmanip.feature_extraction import extract_features_for_one_data_point
from filepath import dataset_path
from train_eval.predict import predict_one_ensemble
from train_eval.load_models import load_best_model_based_on_match


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


def read_matrices_for_configs(matrix_directory: str, config: int) -> np.ndarray:
    """
    For each configuration id in `configs`, build the file path for its matrix file,
    read the matrix, and return a DataFrame with columns 'matrix' and 'config_id'.
    """
    all_matrices = []

    path = os.path.join(matrix_directory, f"config_{config}.txt")
    matrix = read_matrix_file(path)
    matrix = np.array(matrix)
    return matrix


def read_reliability_for_configs(rel_directory: str, config: int, timestamp: int) -> float:
    """
    For each configuration id in `configs`, build the file path for its reliability file,
    read the reliability data (only up to 1000 hours), and return a DataFrame with columns
    'rel_data' and 'config_id'.
    """

    path = os.path.join(rel_directory, f"config_{config}.csv")
    df = pd.read_csv(
        path,
        sep=';',
        decimal=',',
        names=['timestamp', 'reliability']
    )
    # Check if the df has the timestamp if not, return 0 otherwise return the float with reliability
    temp = df['timestamp'].values
    if timestamp in temp:
        return df.loc[df['timestamp'] == timestamp, 'reliability'].iloc[0]
    else:
        return 0


def print_output(rel_class, reliability, inference_time, total_time):
    print(f"Predicted class: {rel_class}")
    # Determine te reliability range
    reliability_lower_bound = "0." + "9" * rel_class
    reliability_upper_bound = "0." + "9" * (rel_class + 1)
    print(f"Predicted reliability range: [{reliability_lower_bound}, {reliability_upper_bound})")
    if reliability == 0:
        print("Actual reliability: No data found for the given configuration id and timestamp.")
    else:
        print(f"Actual reliability: {reliability}")
    print(f"Inference time: {inference_time}")
    print(f"Total time including imports: {total_time}")



def main():
    matches = ["1-1", "2-1", "2-2", "3-1", "3-2", "3-3", "3-4"]
    ensemble = {}
    for match in matches:
        ensemble[match] = load_best_model_based_on_match(match)

    while True:
        desired_config = input("Enter the configuration id (between 0 and 964): ")
        if not desired_config.isdigit():
            print("Please enter a valid integer.")
            continue
        if not 0 <= int(desired_config) <= 964:
            print("Please enter a number between 0 and 964.")
            continue
        if int(desired_config) == 1:
            print("Configuration id 1 is not available.")
            continue
        desired_config = int(desired_config)
        timestamp = input("Enter the timestamp (between 0 and 8500): ")
        if not timestamp.isdigit():
            print("Please enter a valid integer.")
            continue
        if  int(timestamp) < 0:
            print("Please enter a number greater than 0.")
            continue
        timestamp = int(timestamp)
        import_start_time = time.perf_counter()
        matrices_path = os.path.join(dataset_path, "raw", "01_system", "matrix")
        reliabilities_path = os.path.join(dataset_path, "raw", "03_reliability", "hours")
        matrix = read_matrices_for_configs(matrices_path, desired_config)
        reliability = read_reliability_for_configs(reliabilities_path, desired_config, timestamp)
        start_time = time.perf_counter()
        edge_indices = np.nonzero(matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        node_features = extract_features_for_one_data_point(matrix, timestamp, desired_config)
        node_features = torch.Tensor(node_features)
        data = Data(x=node_features, edge_index=edge_index, y=reliability)
        rel_class = predict_one_ensemble(ensemble, data)
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        total_time = end_time - import_start_time
        print_output(rel_class, reliability, inference_time, total_time)


if __name__ == "__main__":
    print("Calculating inference times...")
    main()
