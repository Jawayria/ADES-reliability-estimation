from topology_parameters import prefix_to_type, type_failure_rates
from filepath import configs_list_path

def calculate_failure_rates() -> dict[int, float]:

    # Initialize the dictionary to store the index to type mapping
    index_to_type = {}

    # Read the first line from 'configs.txt'
    with open(configs_list_path, 'r') as file:
        first_line = file.readline().strip()

    # Extract the part after 'config_0: [' and before the closing ']'
    start = first_line.find('[')
    end = first_line.find(']')
    node_list_str = first_line[start + 1:end]

    # Split the node names by comma and strip whitespace
    node_names = [node.strip() for node in node_list_str.split(',')]

    # Iterate over the node names and map index to type
    for index, node_name in enumerate(node_names):
        if node_name.startswith('Sw'):
            node_type = 'switch'
        else:
            # Use the first character as prefix
            prefix = node_name[0]
            node_type = prefix_to_type.get(prefix, 'unknown')
        # Map the index to the node type
        index_to_type[index] = node_type

    index_to_failure_rate = {}
    for index, node_type in index_to_type.items():
        failure_rate = type_failure_rates.get(node_type, 0)
        index_to_failure_rate[index] = failure_rate

    return index_to_failure_rate

