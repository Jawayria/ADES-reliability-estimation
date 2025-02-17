import os

# Get the directory of the current file.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Now join paths relative to that directory:
model_checkpoints_path = os.path.join(THIS_DIR, "..", "models")
hyperparameters_path = os.path.join(model_checkpoints_path, "booster", "best_model_params")
best_models_path = os.path.join(model_checkpoints_path, "booster", "best_models")

dataset_path = os.path.join(THIS_DIR, "..", "data", "3-switches-5-slaves")

matrices_path = os.path.join(dataset_path, "raw", "01_system", "matrix", "config_*.txt")
reliabilities_path = os.path.join(dataset_path, "raw", "03_reliability", "hours", "config_*.csv")
configs_list_path = os.path.join(dataset_path, "raw", "01_system", "configs.txt")
config_all_path = os.path.join(dataset_path, "raw", "03_reliability", "hours", "config_all.csv")