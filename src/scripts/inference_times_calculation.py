# Import necessary libraries
import glob
from time import perf_counter
from tqdm import tqdm

import numpy as np
import torch

from datamanip.data_point import create_one_data_point_from_file
from filepath import matrices_path, matrices_5_7_path
from train_eval.predict import predict_one_meta_learner
from train_eval.load_models import load_best_solution
from utils import extract_config_id


def calc_inference_times():
    TIMESTAMP = 1000
    RELIABILITY_PLACEHOLDER = 0
    PRINT_PREDICTIONS = True
    DATASET = '5-7'  # or '5-7'
    # Measure inference times
    inference_times = []
    total_time = 0
    # Load the best ensemble of models and the meta-learner
    ensemble, meta_learner = load_best_solution()

    with torch.no_grad():
        # Check each file in matrices_path and create a Data object for each
        # Get list of all files in the directory
        if DATASET == '3-5':
            path_to_matrices = matrices_path
        elif DATASET == '5-7':
            path_to_matrices = matrices_5_7_path
        files = sorted(glob.glob(path_to_matrices))
        for file in tqdm(files):
            start_time = perf_counter()
            data = create_one_data_point_from_file(file, TIMESTAMP, RELIABILITY_PLACEHOLDER, DATASET)
            if data is None:
                continue
            predicted_class = predict_one_meta_learner(ensemble, meta_learner, data)
            end_time = perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            total_time += inference_time

            if PRINT_PREDICTIONS:
                print(f"Config: {extract_config_id(file)}, Predicted Class: {predicted_class}")

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
