import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import time

from datamanip.datasetmanip.dataset_util import split_dataset
from filepath import matrices_path, reliabilities_path, config_all_path
from datamanip.read_csvs import read_matrices, read_rel_values, merge_matrices_and_rel
from datamanip.datasetmanip.dataset_parts_construction import (
    construct_edge_indices,
    construct_node_features,
    construct_reliability_classes
)


class ThreeFiveDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split: str,  # "train" | "val" | "test"
        max_hours: int | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        self.split = split
        self.max_hours = max_hours
        super().__init__(root, transform, pre_transform, pre_filter)

        split_to_file = {
            "train": self.processed_paths[0],
            "test": self.processed_paths[1],
            "val": self.processed_paths[2],
        }

        self.load(split_to_file[self.split])

    @property
    def raw_file_names(self):
        return ["01_system/configs.txt"]

    @property
    def processed_file_names(self):
        return [
            "train_data.pt",
            "test_data.pt",
            "val_data.pt",
        ]

    def download(self):
        raise FileNotFoundError(
            f"Please download the dataset manually and place it at {self.raw_dir}"
        )

    def _filter_by_time(self, df):
        if self.max_hours is None:
            return df
        return df[df["timestamp"] <= self.max_hours]



    def process(self):
        start_total = time.time()

        print("\n[1/6] Loading raw data...")
        t0 = time.time()
        all_matrices_df = read_matrices(matrices_path)
        all_rels_df = read_rel_values(reliabilities_path, config_all_path)
        merged_df = merge_matrices_and_rel(all_matrices_df, all_rels_df)
        print(f"Loaded {len(merged_df)} topologies in {time.time() - t0:.2f}s")

        print("\n[2/6] Splitting by topology...")
        t0 = time.time()
        unique_configs = merged_df["config_id"].unique()
        train_ids, test_ids, val_ids = split_dataset(unique_configs)

        train_df = merged_df[merged_df["config_id"].isin(train_ids)]
        test_df = merged_df[merged_df["config_id"].isin(test_ids)]
        val_df = merged_df[merged_df["config_id"].isin(val_ids)]

        print(f"Train configs: {len(train_ids)}, Test: {len(test_ids)}, Val: {len(val_ids)}")
        print(f"Split done in {time.time() - t0:.2f}s")

        print("\n[3/6] Exploding time dimension...")
        t0 = time.time()

        def explode(df, name):
            before = len(df)
            df = df.explode(["timestamp", "reliability"]).reset_index(drop=True)
            df = self._filter_by_time(df)
            after = len(df)
            print(f"{name}: {before} → {after} samples")
            return df

        train_df = explode(train_df, "Train")
        test_df = explode(test_df, "Test")
        val_df = explode(val_df, "Val")

        print(f"Explosion done in {time.time() - t0:.2f}s")

        print("\n[4/6] Constructing ordinal labels...")
        t0 = time.time()
        train_labels, bins = construct_reliability_classes(train_df)
        val_labels = construct_reliability_classes(val_df, bins)
        test_labels = construct_reliability_classes(test_df, bins)

        print(f"Number of classes: {len(bins)}")
        print(f"Label construction done in {time.time() - t0:.2f}s")

        print("\n[5/6] Building graph datasets...")

        for name, df, labels, path in zip(
                ["Train", "Test", "Val"],
                [train_df, test_df, val_df],
                [train_labels, test_labels, val_labels],
                self.processed_paths
        ):
            print(f"\nProcessing {name} set ({len(df)} samples)...")
            t0 = time.time()

            # Edge indices
            print("  → Constructing edge indices...")
            edge_indices = construct_edge_indices(df)

            # Node features (already uses tqdm internally)
            print("  → Constructing node features...")
            node_features = construct_node_features(df)

            print("  → Packaging Data objects...")
            data_list = []
            for i in tqdm(range(len(edge_indices)), desc=f"{name} Data objects"):
                data = Data(
                    x=node_features[i],
                    edge_index=edge_indices[i],
                    y=labels[i]
                )
                data_list.append(data)

            self.save(data_list, path)

            print(f"{name} saved ({len(data_list)} samples) in {time.time() - t0:.2f}s")

        print(f"\n[6/6] Done. Total time: {time.time() - start_total:.2f}s\n")