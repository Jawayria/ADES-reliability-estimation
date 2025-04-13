import os
import re
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.ao.ns.fx.weight_utils import extract_weight_from_node
from torch_geometric.data import Data

from datamanip.dataset_parts_construction import construct_edge_indices
from datamanip.feature_extraction import extract_features_from_data
from datamanip.read_csvs import read_matrix_file
from filepath import dataset_path
from train_eval.predict import predict_one_ensemble
from utils import load_best_model_based_on_match


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
    matches = ["1-1", "2-1", "2-2", "3-1", "3-2", "3-3", "3-4"]
    ensemble = {}
    for match in matches:
        ensemble[match] = load_best_model_based_on_match(match)
        ensemble[match].eval()
    configs = range(1001,1008)
    for config in configs:
        start_time = perf_counter()
        dict = {'matrix': np.array(read_matrix_file(f'../../data/5-switches-7-slaves/matrix/config_{config}.txt')),
                'config_id': config, 'timestamp': 1000}
        matrices_df = pd.DataFrame([dict])
        df = matrices_df
        node_features = torch.Tensor(extract_features_from_data(df))
        edge_indices = construct_edge_indices(df)
        data = Data(x=node_features[0], edge_index=edge_indices[0], y=100)
        predicted_class = predict_one_ensemble(ensemble, data)
        end_time = perf_counter()
        inference_time = end_time - start_time
        print(f"config_{config}: inference time: {inference_time}, predicted class: {predicted_class}")


if __name__ == "__main__":
    calc_inference_times()
