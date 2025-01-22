
import re

def extract_config_id(filename):
    match = re.search(r"config_(\d+)", filename)
    return int(match.group(1)) if match else None

