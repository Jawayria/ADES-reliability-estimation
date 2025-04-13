# Import necessary libraries
import glob
import os

import numpy as np
import pandas as pd
from time import perf_counter

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from datamanip.dataset_parts_construction import construct_edge_indices
from datamanip.feature_extraction import extract_features_from_data
from datamanip.read_csvs import read_matrix_file
from filepath import matrices_path
from train_eval.predict import predict_one_ensemble
from utils import load_best_model_based_on_match


def calc_inference_times():
    # Measure inference times
    inference_times = []
    total_time = 0

    matches = ["1-1", "2-1", "2-2", "3-1", "3-2", "3-3", "3-4"]
    ensemble = {}
    for match in matches:
        ensemble[match] = load_best_model_based_on_match(match)

    with torch.no_grad():
        # Check each file in matrices_path and create a Data object for each
        #Get list of all files in the directory
        files = sorted(glob.glob(matrices_path))
        for file in tqdm(files):
            config_id = os.path.basename(file).split('_')[1].split('.')[0]
            if (config_id == "1"):
                continue
            matrix = read_matrix_file(os.path.join(matrices_path, file))
            start_time = perf_counter()
            dict = {'matrix': np.array(matrix),
                    'config_id': config_id, 'timestamp': 1000}
            df = pd.DataFrame([dict])
            node_features = torch.Tensor(extract_features_from_data(df))
            edge_indices = construct_edge_indices(df)
            data = Data(x=node_features[0], edge_index=edge_indices[0], y=100)
            _ = predict_one_ensemble(ensemble, data)
            end_time = perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            total_time += inference_time

    # Sort by inference time
    sorted_times = sorted(inference_times)

    # Extract results
    best_time = sorted_times[0]
    worst_time = sorted_times[-1]
    average_time = np.mean(sorted_times)

    # Print results
    print("Best Inference Time:", best_time)
    print("Worst Inference Time:", worst_time)
    print("Average Inference Time:", average_time)
    print("Total Inference Time:", total_time)


if __name__ == "__main__":
    print("Calculating inference times...")
    calc_inference_times()