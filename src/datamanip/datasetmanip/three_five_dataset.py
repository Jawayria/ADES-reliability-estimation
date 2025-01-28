import os

import torch
from torch_geometric.data import InMemoryDataset, Data

from filepath import matrices_path, reliabilities_path, config_all_path, configs_list_path, dataset_path
from datamanip.read_csvs import read_matrices, read_rel_values, merge_matrices_and_rel
from datamanip.dataset_parts_construction import construct_edge_indices, construct_reliability_classes, \
    construct_node_features


class ThreeFiveDataset(InMemoryDataset):
    def __init__(self, root, load_binary: bool, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if load_binary:
            self.load(self.processed_paths[0])
        else:
            self.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [os.path.relpath(config_all_path, start=dataset_path),
                os.path.relpath(configs_list_path, start=dataset_path)]

    @property
    def processed_file_names(self):
        return ['3-5data_bin.pt', '3-5data_multi.pt']

    def download(self):
        # Download to `self.raw_dir`.
        raise FileNotFoundError('Please download the dataset manually and place it at {}'.format(self.raw_dir))


    def compose_data_list(self, edge_indices: list[torch.Tensor], rels: torch.Tensor, nodes_features: torch.Tensor,
                          num_of_entries: int) -> list[Data]:
        # Create list of Data objects, each containing the node features, edge indices, and target values
        data_list = []

        for i in range(num_of_entries):
            node_features = nodes_features[i]
            edge_index = edge_indices[i]
            y = rels[i]
            data = Data(x=node_features, edge_index=edge_index, y=y)
            data_list.append(data)
        return data_list

    def process(self):
        # Read data into huge `Data` list.

        all_matrices_df = read_matrices(matrices_path)
        all_rels_df = read_rel_values(reliabilities_path, config_all_path)
        merged_df = merge_matrices_and_rel(all_matrices_df, all_rels_df)

        # Explode the 'timestamp' and 'reliability' columns
        merged_df_exploded = merged_df.explode(['timestamp', 'reliability']).reset_index(drop=True)
        all_edge_indices = construct_edge_indices(merged_df_exploded)
        (all_rels_tensor_binary, all_rels_tensor_multiclass) = construct_reliability_classes(merged_df_exploded)
        node_features_tensor = construct_node_features(merged_df_exploded)
        num_of_entries = len(merged_df_exploded)
        binary_data_list = self.compose_data_list(all_edge_indices, all_rels_tensor_binary, node_features_tensor,
                                                  num_of_entries)
        multi_data_list = self.compose_data_list(all_edge_indices, all_rels_tensor_multiclass, node_features_tensor,
                                                 num_of_entries)
        self.save(binary_data_list, self.processed_paths[0])
        self.save(multi_data_list, self.processed_paths[1])
