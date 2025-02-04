import os

import torch
from torch_geometric.data import InMemoryDataset, Data

from filepath import matrices_path, reliabilities_path, config_all_path, configs_list_path, dataset_path
from datamanip.read_csvs import read_matrices, read_rel_values, merge_matrices_and_rel
from datamanip.dataset_parts_construction import construct_edge_indices, construct_reliability_classes, \
    construct_node_features, convert_range_to_floats, trim_df_by_range

RANGES = [(0, 4, 8), (0, 2, 4), (4, 6, 8), (0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)]
ALT_RANGES = [(0, 1, 8), (1, 4, 8), (1, 2, 4), (2, 3, 4), (4, 6, 8), (4, 5, 6), (6, 7, 8)]


class ThreeFiveDataset(InMemoryDataset):
    def __init__(self, root, match: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        #Find file in processed_file_names that contains the 'match' string
        for i, file_name in enumerate(self.processed_file_names):
            if match in file_name:
                self.load(file_name)

    @property
    def raw_file_names(self):
        #return [os.path.relpath(config_all_path, start="raw"),
        #        os.path.relpath(configs_list_path, start="raw")]
        return ["01_system/configs.txt"]

    @property
    def processed_file_names(self):
        return ['1-1_model_data.pt', '2-1_model_data.pt', '2-2_model_data.pt',
                '3-1_model_data.pt', '3-2_model_data.pt', '3-3_model_data.pt', '3-4_model_data.pt']
                #'alt1-1_model_data.pt', 'alt2-1_model_data.pt', 'alt2-2_model_data.pt',
                #'alt3-1_model_data.pt', 'alt3-2_model_data.pt', 'alt3-3_model_data.pt', 'alt3-4_model_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        raise FileNotFoundError('Please download the dataset manually and place it at {}'.format(self.raw_dir))

    def compose_data_list(self, edge_indices: list[torch.Tensor], rels: torch.Tensor, nodes_features: torch.Tensor) -> \
    list[Data]:
        # Create list of Data objects, each containing the node features, edge indices, and target values
        data_list = []

        for i in range(len(edge_indices)):
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
        #Remove all rows where timestamp is above 10000
        filtered_df = merged_df_exploded[merged_df_exploded['timestamp'] <= 10000]
        print("Finished creating dataframe")
        for i, rel_range in enumerate(RANGES):
            converted_range = convert_range_to_floats(rel_range)
            trimmed_df = trim_df_by_range(filtered_df, converted_range[0], converted_range[2])
            all_edge_indices = construct_edge_indices(trimmed_df)
            node_features_tensor = construct_node_features(trimmed_df)
            all_rels_tensor = construct_reliability_classes(trimmed_df, converted_range[1], binary=True)
            data_list = self.compose_data_list(all_edge_indices, all_rels_tensor, node_features_tensor)
            self.save(data_list, self.processed_paths[i])
            print(f"Finished processing {i + 1} out of {len(RANGES)} ranges")
        for i, rel_range in enumerate(ALT_RANGES):
            converted_range = convert_range_to_floats(rel_range)
            trimmed_df = trim_df_by_range(filtered_df, converted_range[0], converted_range[2])
            all_edge_indices = construct_edge_indices(trimmed_df)
            node_features_tensor = construct_node_features(trimmed_df)
            all_rels_tensor = construct_reliability_classes(trimmed_df, converted_range[1], binary=True)
            data_list = self.compose_data_list(all_edge_indices, all_rels_tensor, node_features_tensor)
            self.save(data_list, self.processed_paths[i + len(RANGES)])
            print(f"Finished processing {i + 1} out of {len(ALT_RANGES)} alternative ranges")
