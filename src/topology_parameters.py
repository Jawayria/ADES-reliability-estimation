# Dictionary mapping prefixes used in config files to node types
prefix_to_type = {
    'S': 'slave',  # Prefix 'S' maps to 'slave'
    'L': 'link',   # Prefix 'L' maps to 'link'
    'Sw':'switch', # Prefix 'Sw' maps to 'switch'
    'I': 'i'       # Prefix 'I' maps to 'i'
}

# Dictionary mapping node types to their respective failure rates
type_failure_rates = {
    'slave': 0.00001,  # Failure rate for 'slave' nodes
    'link': 0.0000001, # Failure rate for 'link' nodes
    'switch': 0.000001,# Failure rate for 'switch' nodes
    'i': 0.0000001     # Failure rate for 'i' nodes
}