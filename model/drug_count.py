import os

DATA_DIR = '../data/ours'
DRUG_TYPE_ID = 1

def inspect_names(data_path, type_id_to_inspect):
    node_file = os.path.join(data_path, 'node.dat')
    if not os.path.exists(node_file):
        print(f"Error: Cannot find node file at {node_file}")
        return

    unique_names = set()
    with open(node_file, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split('\t')
                if int(parts[2]) == type_id_to_inspect:
                    unique_names.add(parts[1])
            except (ValueError, IndexError):
                continue
    
    print("--- First 20 Unique Drug Names Found ---")
    for i, name in enumerate(unique_names):
        if i >= 20:
            break
        print(name)
    print("------------------------------------")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(script_dir, DATA_DIR)
    inspect_names(full_data_path, DRUG_TYPE_ID)