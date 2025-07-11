import os
from collections import Counter, defaultdict

def analyze_data_files(data_path):
    """
    A standalone script to analyze and describe the contents of the HetDrugCellNet data files.
    It checks for inconsistencies in node types within the link files.
    """
    print("="*80)
    print(f"Starting Data Analysis for: {data_path}")
    print("="*80)

    # --- 1. Load Node Information ---
    node_file = os.path.join(data_path, 'node.dat')
    if not os.path.exists(node_file):
        print(f"FATAL: Cannot find node file at {node_file}")
        return

    node_type_map = {} # Maps global_id -> type_id
    node_counts = Counter()
    node_type_names = {} # Maps type_id -> type_name (e.g., 0 -> 'cell')

    print(f"\n--- Reading Node Definitions from: {node_file} ---")
    with open(node_file, 'r') as f:
        for line in f:
            try:
                g_id, name, n_type = line.strip().split('\t')
                g_id, n_type = int(g_id), int(n_type)
                node_type_map[g_id] = n_type
                node_counts[n_type] += 1
                if n_type not in node_type_names:
                    # Infer type name from the first part of the node name string
                    # e.g., 'cell_line_123' -> 'cell'
                    type_name_inferred = name.split('_')[0].lower()
                    node_type_names[n_type] = type_name_inferred
            except ValueError:
                print(f"  > Warning: Skipping malformed line in node.dat: {line.strip()}")
                continue
    
    print("Node Analysis Complete:")
    for type_id in sorted(node_counts.keys()):
        type_name = node_type_names.get(type_id, "Unknown")
        print(f"  - Node Type {type_id} ('{type_name}'): {node_counts[type_id]} nodes")
    print("-" * 80)


    # --- 2. Create an Analysis Function ---
    def analyze_link_file(file_path):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping analysis.")
            return

        print(f"\n--- Analyzing Link File: {os.path.basename(file_path)} ---")
        
        # { relation_id: Counter({(source_type, target_type): count}) }
        relation_type_breakdown = defaultdict(Counter)
        total_links = 0
        malformed_links = 0

        with open(file_path, 'r') as f:
            for line in f:
                try:
                    h_id, t_id, r_id, _ = line.strip().split('\t')
                    h_id, t_id, r_id = int(h_id), int(t_id), int(r_id)
                    total_links += 1

                    # Look up the types of the head and tail nodes
                    h_type = node_type_map.get(h_id, 'UNKNOWN')
                    t_type = node_type_map.get(t_id, 'UNKNOWN')

                    if h_type == 'UNKNOWN' or t_type == 'UNKNOWN':
                        malformed_links += 1
                        continue
                    
                    # Record the observed (source_type, target_type) pair for this relation
                    relation_type_breakdown[r_id][(h_type, t_type)] += 1

                except (ValueError, IndexError):
                    malformed_links += 1
                    continue
        
        print(f"  > Total links found: {total_links}")
        if malformed_links > 0:
            print(f"  > WARNING: Skipped {malformed_links} malformed or invalid links.")

        # Print the summary report for the file
        for r_id in sorted(relation_type_breakdown.keys()):
            print(f"\n  Relation ID: {r_id}")
            total_for_relation = sum(relation_type_breakdown[r_id].values())
            print(f"    - Total Links for this Relation: {total_for_relation}")
            print(f"    - Observed (Source Type, Target Type) pairs:")
            for (h_type, t_type), count in relation_type_breakdown[r_id].items():
                h_name = node_type_names.get(h_type, f"Type {h_type}")
                t_name = node_type_names.get(t_type, f"Type {t_type}")
                print(f"      - ({h_name}, {t_name}): {count} links")
        print("-" * 80)

    # --- 3. Run Analysis on Both Files ---
    analyze_link_file(os.path.join(data_path, 'link.dat'))
    analyze_link_file(os.path.join(data_path, 'link.dat.test'))


if __name__ == '__main__':
    # The script assumes it's being run from the project root directory
    # e.g., /data/luis/HetDrugCellNet/
    # It will look for the data in './data/ours'
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(project_root, '..', 'data', 'ours')
    
    # If you place this script in the 'model' folder, this path will be correct.
    # If you place it elsewhere, you may need to adjust the path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory_from_model_folder = os.path.join(script_dir, '..', 'data', 'ours')

    if os.path.exists(data_directory_from_model_folder):
        analyze_data_files(data_directory_from_model_folder)
    else:
        print(f"Error: Could not find data directory at {data_directory_from_model_folder}")
        print("Please run this script from the 'model' directory, or adjust the path inside the script.")

