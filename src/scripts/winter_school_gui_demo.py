import os
import re
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QPushButton, QComboBox, QLineEdit, QTextEdit)
from PyQt6.QtGui import QPixmap

from datamanip.datasetmanip.feature_extraction import extract_features_for_one_data_point
from filepath import dataset_path
from train_eval.predict import predict_one_ensemble
from train_eval.load_models import load_best_model_based_on_match


def extract_config_id(path: str) -> int:
    match = re.search(r's(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract config_id from path: {path}")

def read_matrix_file(path: str) -> list:
    with open(path, 'r') as f:
        content = f.read()
    content = content[1:-1]
    string_list, current_row = [], ''
    start_line = False
    for c in content:
        if c == '[':
            start_line = True
        elif c == ']':
            start_line = False
            string_list.append(current_row)
            current_row = ''
        elif start_line and c != '\n':
            current_row += c
    return [list(map(int, row.replace('.', ' ').split())) for row in string_list]

def read_matrices_for_configs(matrix_directory: str, config: int) -> np.ndarray:
    path = os.path.join(matrix_directory, f"config_{config}.txt")
    return np.array(read_matrix_file(path))

def read_reliability_for_configs(rel_directory: str, config: int, timestamp: int) -> float:
    path = os.path.join(rel_directory, f"config_{config}.csv")
    df = pd.read_csv(path, sep=';', decimal=',', names=['timestamp', 'reliability'])
    return df.loc[df['timestamp'] == timestamp, 'reliability'].iloc[0] if timestamp in df['timestamp'].values else 0


def parse_config_file(config_file_path):
    config_mapping = {}
    with open(config_file_path, 'r') as file:
        for line in file:
            match = re.match(r'(config_\d+): \[(.*?)\]', line)
            if match:
                config_id = int(match.group(1).split('_')[1])
                components = match.group(2).split(', ')
                num_switches = sum(1 for c in components if c.startswith('Sw'))
                num_slaves = sum(1 for c in components if re.match(r'^S\d+$', c))  # Only count "S" followed by a number
                num_links = sum(1 for c in components if c.startswith('L') or c.startswith('I'))
                config_mapping[config_id] = (num_switches, num_slaves, num_links)
    return config_mapping

config_file_path = os.path.join(dataset_path,"raw", "01_system", "configs.txt")
config_mapping = parse_config_file(config_file_path)

def get_unique_counts(mapping, index):
    return sorted(set(value[index] for value in mapping.values()))

unique_switches = get_unique_counts(config_mapping, 0)
unique_slaves = get_unique_counts(config_mapping, 1)
unique_links = get_unique_counts(config_mapping, 2)

def get_matching_configs(switches, slaves, links):
    return [k for k, v in config_mapping.items() if v == (switches, slaves, links)]


class ReliabilityPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.ensemble = {match: load_best_model_based_on_match(match) for match in ["1-1", "2-1", "2-2", "3-1", "3-2", "3-3", "3-4"]}

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.switch_label = QLabel("Select Number of Switches:")
        self.switch_dropdown = QComboBox()
        self.switch_dropdown.addItems(map(str, unique_switches))
        self.switch_dropdown.currentIndexChanged.connect(self.update_config_options)
        
        self.slave_label = QLabel("Select Number of Nodes:")
        self.slave_dropdown = QComboBox()
        self.slave_dropdown.addItems(map(str, unique_slaves))
        self.slave_dropdown.currentIndexChanged.connect(self.update_config_options)
        
        self.link_label = QLabel("Select Number of Links:")
        self.link_dropdown = QComboBox()
        self.link_dropdown.addItems(map(str, unique_links))
        self.link_dropdown.currentIndexChanged.connect(self.update_config_options)
        
        self.config_label = QLabel("Matching Configurations:")
        self.config_dropdown = QComboBox()
        self.config_dropdown.currentIndexChanged.connect(self.display_config_image)  # Added this
        
        self.timestamp_label = QLabel("Enter Timestamp:")
        self.timestamp_input = QLineEdit()
        
        self.image_label = QLabel("Configuration Image:")
        self.image_display = QLabel()
        
        self.run_button = QPushButton("Run Prediction")
        self.run_button.clicked.connect(self.run_prediction)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        
        layout.addWidget(self.switch_label)
        layout.addWidget(self.switch_dropdown)
        layout.addWidget(self.slave_label)
        layout.addWidget(self.slave_dropdown)
        layout.addWidget(self.link_label)
        layout.addWidget(self.link_dropdown)
        layout.addWidget(self.config_label)
        layout.addWidget(self.config_dropdown)
        layout.addWidget(self.timestamp_label)
        layout.addWidget(self.timestamp_input)
        layout.addWidget(self.image_label)
        layout.addWidget(self.image_display)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)
        self.setWindowTitle("Reliability Predictor GUI")
        self.update_config_options()

    def update_config_options(self):
        switches = int(self.switch_dropdown.currentText())
        slaves = int(self.slave_dropdown.currentText())
        links = int(self.link_dropdown.currentText())
        matching_configs = get_matching_configs(switches, slaves, links)
        self.config_dropdown.clear()
        self.config_dropdown.addItems(map(str, matching_configs))
        self.display_config_image()  # Ensures the image updates immediately

    def display_config_image(self):
        fixed_width = 300  # Set your desired width
        fixed_height = 300  # Set your desired height
    
        if not self.config_dropdown.count():
            self.image_display.clear()
            return

        config_id = int(self.config_dropdown.currentText())
        image_path = os.path.join(dataset_path, "raw", "01_system", "configs", f"config_{config_id}.png")

        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(fixed_width, fixed_height)  # Scale to fixed size
            self.image_display.setPixmap(pixmap)
            self.image_display.setFixedSize(fixed_width, fixed_height)  # Fix label size
        else:
            self.image_display.clear()
            self.image_display.setText("Image not found.")
            self.image_display.setFixedSize(fixed_width, fixed_height)  # Keep the label size fixed
            
    def run_prediction(self):
        if not self.config_dropdown.count():
            self.output_text.setText("No matching configuration found.")
            return
        
        config_id = int(self.config_dropdown.currentText())
        timestamp = self.timestamp_input.text().strip()
        
        if not timestamp.isdigit() or int(timestamp) < 0:
            self.output_text.setText("Invalid timestamp. Please enter a valid number between 0 and 8500.")
            return
        timestamp = int(timestamp)
        
        matrices_path = os.path.join(dataset_path, "raw", "01_system", "matrix")
        reliabilities_path = os.path.join(dataset_path, "raw", "03_reliability", "hours")
        
        matrix = read_matrices_for_configs(matrices_path, config_id)
        reliability = read_reliability_for_configs(reliabilities_path, config_id, timestamp)
        
        start_time = time.perf_counter()
        edge_indices = np.nonzero(matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        node_features = torch.Tensor(extract_features_for_one_data_point(matrix, timestamp, config_id))
        data = Data(x=node_features, edge_index=edge_index, y=reliability)
        rel_class = predict_one_ensemble(self.ensemble, data)
        inference_time = time.perf_counter() - start_time
        
        reliability_lower_bound = "0." + "9" * rel_class
        reliability_upper_bound = "0." + "9" * (rel_class + 1)
        output_text = (f"Predicted Class: {rel_class}\n"
                       f"Predicted Reliability Range: [{reliability_lower_bound}, {reliability_upper_bound})\n"
                       f"Actual Reliability: {reliability if reliability else 'No data found'}\n"
                       f"Inference Time: {inference_time:.4f} seconds")
        self.output_text.setText(output_text)


if __name__ == "__main__":
    app = QApplication([])
    window = ReliabilityPredictor()
    window.show()
    app.exec()
