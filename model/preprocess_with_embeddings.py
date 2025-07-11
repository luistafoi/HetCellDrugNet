import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import json

def preprocess_data_with_embeddings(original_data_path, embeddings_path, output_path):
    """
    Filters the graph to include only nodes with valid pre-trained embeddings
    and generates a new, clean dataset directory.
    """
    print("="*80)
    print("Starting Data Preprocessing with Pre-trained Embeddings")
    print("="*80)

    # --- 1. Define File Paths ---
    node_file_in = os.path.join(original_data_path, 'node.dat')
    link_file_in = os.path.join(original_data_path, 'link.dat')
    link_test_file_in = os.path.join(original_data_path, 'link.dat.test')
    
    # **MODIFIED**: Point to your new MLP-based cell embeddings
    cell_embedding_file = '/data/luis/HetDrugCellNet/model/embeddings/mlp_cell_embeddings.npy'
    
    # These paths for drug and gene embeddings remain the same
    drug_embedding_file = '/data/luis/MoleculeSTM/drugs_with_embeddings.csv'
    gene_embedding_file = os.path.join(embeddings_path, 'gene_embeddings_esm_by_symbol.pkl')
    
    os.makedirs(output_path, exist_ok=True)
    node_file_out = os.path.join(output_path, 'node.dat')
    link_file_out = os.path.join(output_path, 'link.dat')
    link_test_file_out = os.path.join(output_path, 'link.dat.test')
    info_file_out = os.path.join(output_path, 'info.dat') # You may need to create this manually or adjust script
    
    print(f"Output Path: {output_path}\n")

    # --- 2. Load Embeddings and Identify Valid Nodes ---
    print("--- Step 1: Loading embeddings and identifying valid nodes ---")
    
    # Load cell embeddings and identify which ones are valid (non-zero)
    print(f"  > Loading cell embeddings from {cell_embedding_file}...")
    all_cell_embeddings = np.load(cell_embedding_file)
    # Find the indices of rows that are NOT all zeros
    valid_cell_indices = np.where(np.any(all_cell_embeddings != 0, axis=1))[0]
    print(f"    Found {len(valid_cell_indices)} valid (non-zero) cell embeddings.")

    # Get the names of the valid cells from the original node.dat file
    # This requires reading node.dat to map local indices back to names
    original_cell_nodes = []
    with open(node_file_in, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3 and int(parts[2]) == 0:
                original_cell_nodes.append(parts[1])
    
    # Create the set of valid cell names to filter with
    cell_names_with_embeddings = {original_cell_nodes[i] for i in valid_cell_indices}

    # Load drug embeddings (logic is unchanged)
    print(f"  > Loading drug embeddings...")
    drug_df = pd.read_csv(drug_embedding_file, index_col=0)
    drug_df.index = drug_df.index.str.upper()
    drug_df = drug_df[~drug_df.index.duplicated(keep='first')]
    drug_names_with_embeddings = set(drug_df.index)
    print(f"    Found {len(drug_names_with_embeddings)} unique drugs with embeddings.")

    # Load gene embeddings (logic is unchanged)
    print(f"  > Loading gene embeddings...")
    with open(gene_embedding_file, 'rb') as f:
        gene_embedding_dict = {str(k).upper(): v for k, v in pickle.load(f).items()}
    gene_names_with_embeddings = set(gene_embedding_dict.keys())
    print(f"    Found {len(gene_names_with_embeddings)} genes with embeddings.\n")


    # --- 3. Filter Nodes ---
    print("--- Step 2: Filtering nodes based on embedding availability ---")
    allowed_nodes = {}
    with open(node_file_in, 'r') as f:
        for line in f:
            try:
                g_id, name, n_type = line.strip().split('\t')
                g_id, n_type = int(g_id), int(n_type)
                
                # The original, robust filtering logic now works perfectly
                if n_type == 0 and name in cell_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': n_type}
                elif n_type == 1 and name.upper() in drug_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': n_type}
                elif n_type == 2 and name.upper() in gene_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': n_type}
            except (ValueError, IndexError):
                continue
    
    print(f"  > Original node count (from file): {len(open(node_file_in).readlines())}")
    print(f"  > Filtered node count (with embeddings): {len(allowed_nodes)}\n")


    # --- 4. Create New Node File and Link Files (Logic is unchanged) ---
    print("--- Step 3: Generating new, filtered node.dat and link files ---")
    old_to_new_id_map = {}
    new_filtered_nodes = []
    # (The rest of this section for creating node.dat and link.dat is identical to your original script)
    new_global_id_counter = 0
    with open(node_file_out, 'w') as f_out:
        for node_type_id in sorted([0, 1, 2]):
            nodes_of_this_type = { g_id: info for g_id, info in allowed_nodes.items() if info['type'] == node_type_id }
            for old_g_id in sorted(nodes_of_this_type.keys()):
                info = nodes_of_this_type[old_g_id]
                old_to_new_id_map[old_g_id] = new_global_id_counter
                new_filtered_nodes.append(info)
                f_out.write(f"{new_global_id_counter}\t{info['name']}\t{info['type']}\n")
                new_global_id_counter += 1
    
    def filter_link_file(input_file, output_file, id_map):
        links_written = 0
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                try:
                    h_id, t_id, r_id, weight = line.strip().split('\t')
                    if int(h_id) in id_map and int(t_id) in id_map:
                        f_out.write(f"{id_map[int(h_id)]}\t{id_map[int(t_id)]}\t{r_id}\t{weight}\n")
                        links_written += 1
                except (ValueError, IndexError): continue
        print(f"  > Filtered {os.path.basename(input_file)}: {links_written} links written.")

    filter_link_file(link_file_in, link_file_out, old_to_new_id_map)
    filter_link_file(link_test_file_in, link_test_file_out, old_to_new_id_map)


    # --- 5. Create and Save Final Embedding Matrices ---
    print("\n--- Step 4: Creating final filtered embedding matrices ---")
    
    # **MODIFIED**: Filter the cell embedding matrix
    final_cell_embeds = all_cell_embeddings[valid_cell_indices]
    
    # The rest of the logic is the same as the modified version from our last conversation
    filtered_node_counts = Counter(info['type'] for info in new_filtered_nodes)
    final_drug_embeds = np.zeros((filtered_node_counts.get(1, 0), drug_df.shape[1]))
    if 2 in filtered_node_counts and gene_embedding_dict:
        final_gene_embeds = np.zeros((filtered_node_counts.get(2, 0), next(iter(gene_embedding_dict.values())).shape[0]))
    else:
        final_gene_embeds = np.zeros((0,0))

    local_id_counters = Counter()
    for node_info in new_filtered_nodes:
        node_type = node_info['type']
        if node_type == 0:
            continue
        node_name = node_info['name']
        local_id = local_id_counters[node_type]
        if node_type == 1:
            final_drug_embeds[local_id] = drug_df.loc[node_name.upper()].values
        elif node_type == 2:
            final_gene_embeds[local_id] = gene_embedding_dict[node_name.upper()]
        local_id_counters[node_type] += 1

    np.save(os.path.join(output_path, 'cell_embeddings.npy'), final_cell_embeds)
    np.save(os.path.join(output_path, 'drug_embeddings.npy'), final_drug_embeds)
    np.save(os.path.join(output_path, 'gene_embeddings.npy'), final_gene_embeds)
    
    print("  > Successfully created and saved final embedding matrices.")
    print("\n" + "="*80)
    print(f"Preprocessing Complete! New dataset at: {output_path}")
    print("="*80)

        # --- 6. Create info.dat file ---
    print("\n--- Step 5: Creating info.dat for schema definition ---")
    info_file_out = os.path.join(output_path, 'info.dat')
    # This JSON structure tells the main script what the node types mean
    info_content = {
        "node.dat": {
            "0": ["cell"],
            "1": ["drug"],
            "2": ["gene"]
        },
        "link.dat": {
            # This part is less critical for the current script but good practice
            "0": ["cell", "drug", "c-d"],
            "1": ["drug", "cell", "d-c"]
        }
    }
    with open(info_file_out, 'w') as f:
        json.dump(info_content, f, indent=4)
    print(f"  > Successfully created {os.path.basename(info_file_out)}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(script_dir)
    
    # Path to the original, unfiltered data
    original_data_path = os.path.join(PROJECT_ROOT, 'data', 'ours')
    
    # Path to the folder containing gene and drug embeddings
    embeddings_path = os.path.join(script_dir, 'embeddings')
    
    # Name of the output directory you want to create
    output_path = os.path.join(PROJECT_ROOT, 'data', 'ours_filtered_moleculestm_mlp')

    preprocess_data_with_embeddings(original_data_path, embeddings_path, output_path)