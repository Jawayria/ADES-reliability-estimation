import glob
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from datamanip.datasetmanip.dataset_parts_construction import construct_edge_indices
from datamanip.datasetmanip.feature_extraction import extract_features_from_data
from datamanip.read_csvs import read_matrix_file, read_one_reliability_file
from filepath import matrices_path, reliabilities_path, matrices_5_7_path, reliabilities_5_7_path
from utils import extract_config_id

SENTINEL_VALUE = -1


# TODO: Make reliability working for 5-7 (currently, since 5-7 does not have reliability .csvs, if you provide a reliability value, it will be ignored)
def create_one_data_point(config_id: int, timestamp: int, reliability=SENTINEL_VALUE, dataset='3-5') -> Data | None:
    if config_id == 1:
        return None
    if dataset == '3-5':
        path_to_matrices = matrices_path
        path_to_reliabilities = reliabilities_path
    elif dataset == '5-7':
        path_to_matrices = matrices_5_7_path
        path_to_reliabilities = reliabilities_5_7_path
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use '3-5' or '5-7'.")

    all_files = glob.glob(path_to_matrices)
    matrix_file = sorted(all_files)[config_id]
    matrix = read_matrix_file(str(os.path.join(path_to_matrices, matrix_file)))
    data_dict = {'matrix': np.array(matrix),
                 'config_id': config_id, 'timestamp': timestamp}
    df = pd.DataFrame([data_dict])
    node_features = torch.Tensor(extract_features_from_data(df))
    edge_indices = construct_edge_indices(df)
    if reliability == SENTINEL_VALUE:
        if dataset == '5-7':
            raise NotImplementedError(
                "Reliability values for 5-7 dataset are not implemented yet, provide custom value instead.")
        all_reliability_files = glob.glob(path_to_reliabilities)
        rel_file = sorted(all_reliability_files)[config_id]
        reliability_df, _ = read_one_reliability_file(rel_file)
        rel_val = reliability_df.loc[reliability_df['timestamp'] == timestamp, 'reliability'].values[0]
    elif 0 <= reliability <= 1:
        rel_val = reliability
    else:
        raise ValueError(f"Reliability must be between 0 and 1, got {reliability}")
    data = Data(x=node_features[0], edge_index=edge_indices[0], y=rel_val)
    return data


def create_one_data_point_from_file(matrix_file: str, timestamp: int, reliability=SENTINEL_VALUE,
                                    dataset='3-5') -> Data | None:
    config_id = extract_config_id(matrix_file)
    return create_one_data_point(config_id, timestamp, reliability, dataset)
