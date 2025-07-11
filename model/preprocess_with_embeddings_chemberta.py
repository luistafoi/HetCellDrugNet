import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
import sys
import json

def preprocess_data_with_embeddings(original_data_path, embeddings_path, output_path):
    """
    This script performs a one-time preprocessing of the raw data.
    It filters the graph to only include nodes that have pre-trained embeddings
    and generates a new, clean dataset directory and embedding matrices.
    """
    print("="*80)
    print("Starting Data Preprocessing with Pre-trained Embeddings")
    print("="*80)

    # --- 1. Define File Paths ---
    node_file_in = os.path.join(original_data_path, 'node.dat')
    link_file_in = os.path.join(original_data_path, 'link.dat')
    link_test_file_in = os.path.join(original_data_path, 'link.dat.test')
    
    drug_embedding_file = '/data/luis/HetDrugCellNet/model/embeddings/drug_embeddings_chemberta.csv'
    cell_embedding_file = os.path.join(embeddings_path, 'cell_embeddings_vae.csv')
    # --- THIS IS THE CHANGE: Point to the new pickle file with symbol keys ---
    gene_embedding_file = os.path.join(embeddings_path, 'gene_embeddings_esm_by_symbol.pkl')
    cell_line_map_file = os.path.join(embeddings_path, 'cell_line_mapping.csv')

    os.makedirs(output_path, exist_ok=True)
    node_file_out = os.path.join(output_path, 'node.dat')
    link_file_out = os.path.join(output_path, 'link.dat')
    link_test_file_out = os.path.join(output_path, 'link.dat.test')
    info_file_out = os.path.join(output_path, 'info.dat')
    
    print(f"Input Data Path: {original_data_path}")
    print(f"Embeddings Path: {embeddings_path}")
    print(f"Output Path: {output_path}\n")

    # --- 2. Load Pre-trained Embeddings ---
    print("--- Step 1: Loading pre-trained embeddings ---")
    
    print(f"  > Loading cell embeddings from {cell_embedding_file}...")
    cell_df = pd.read_csv(cell_embedding_file, index_col=0)
    map_df = pd.read_csv(cell_line_map_file)
    depmap_to_ccle = pd.Series(map_df.ccle_name.values, index=map_df.depmap_id).to_dict()
    cell_df.rename(index=depmap_to_ccle, inplace=True)
    cell_names_with_embeddings = set(cell_df.index)
    print(f"    Found {len(cell_names_with_embeddings)} cells with embeddings.")

    print(f"  > Loading DRUG (MoleculeSTM) embeddings from {drug_embedding_file}...")
    drug_df = pd.read_csv(drug_embedding_file, index_col=0)
    drug_df.index = drug_df.index.str.upper()
    drug_df = drug_df[~drug_df.index.duplicated(keep='first')]
    drug_names_with_embeddings = set(drug_df.index)
    print(f"    Found {len(drug_names_with_embeddings)} unique drugs with embeddings.")

    print(f"  > Loading gene embeddings from {gene_embedding_file}...")
    try:
        with open(gene_embedding_file, 'rb') as f:
            gene_embedding_dict = pickle.load(f)
        # Convert keys to uppercase for consistent matching
        gene_embedding_dict = {str(k).upper(): v for k, v in gene_embedding_dict.items()}
        gene_names_with_embeddings = set(gene_embedding_dict.keys())
        print(f"    Found {len(gene_names_with_embeddings)} genes with embeddings.\n")
    except FileNotFoundError:
        print("    Gene embedding file not found. Proceeding without gene nodes.")
        gene_embedding_dict = {}
        gene_names_with_embeddings = set()

    # --- 3. Filter Nodes ---
    print("--- Step 2: Filtering nodes based on embedding availability ---")
    
    allowed_nodes = {}
    with open(node_file_in, 'r') as f:
        for line in f:
            try:
                g_id, name, n_type = line.strip().split('\t')
                g_id, n_type = int(g_id), int(n_type)
                
                if n_type == 0 and name in cell_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': n_type}
                elif n_type == 1 and name.upper() in drug_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': n_type}
                elif n_type == 2 and name.upper() in gene_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': n_type}
            except (ValueError, IndexError):
                continue
    
    print(f"  > Original node count: {len(open(node_file_in).readlines())}")
    print(f"  > Filtered node count: {len(allowed_nodes)}\n")

    # --- 4. Create New Node File and ID Mappings ---
    print("--- Step 3: Generating new, filtered node.dat and ID mappings ---")
    
    old_to_new_id_map = {}
    new_filtered_nodes = []
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
    
    print(f"  > New node.dat created at: {node_file_out}\n")

    # --- 5. Filter Link Files ---
    def filter_link_file(input_file, output_file, id_map):
        print(f"--- Step 4: Filtering {os.path.basename(input_file)} ---")
        links_written = 0
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                try:
                    h_id, t_id, r_id, weight = line.strip().split('\t')
                    h_id, t_id = int(h_id), int(t_id)
                    if h_id in id_map and t_id in id_map:
                        new_h_id, new_t_id = id_map[h_id], id_map[t_id]
                        f_out.write(f"{new_h_id}\t{new_t_id}\t{r_id}\t{weight}\n")
                        links_written += 1
                except (ValueError, IndexError): continue
        print(f"  > Filtered links written: {links_written}\n")

    filter_link_file(link_file_in, link_file_out, old_to_new_id_map)
    filter_link_file(link_test_file_in, link_test_file_out, old_to_new_id_map)

    # --- 6. Generate New info.dat ---
    info_content = {"node.dat": {"0": ["cell"], "1": ["drug"], "2": ["gene"]}, "link.dat": {"0": ["cell", "drug", "c-d"], "1": ["drug", "cell", "d-c"]}}
    with open(info_file_out, 'w') as f:
        json.dump(info_content, f, indent=4)
    print(f"--- Step 5: New info.dat created at: {info_file_out} ---\n")

    # --- 7. Create and Save Final Embedding Matrices ---
    print("--- Step 6: Creating final embedding matrices from filtered nodes ---")
    
    filtered_node_counts = Counter(info['type'] for info in new_filtered_nodes)
    
    final_cell_embeds = np.zeros((filtered_node_counts.get(0, 0), cell_df.shape[1]))
    final_drug_embeds = np.zeros((filtered_node_counts.get(1, 0), drug_df.shape[1]))
    if 2 in filtered_node_counts and gene_embedding_dict:
        final_gene_embeds = np.zeros((filtered_node_counts.get(2, 0), next(iter(gene_embedding_dict.values())).shape[0]))
    else:
        final_gene_embeds = np.zeros((0,0))

    local_id_counters = Counter()
    for node_info in new_filtered_nodes:
        node_type = node_info['type']
        node_name = node_info['name']
        local_id = local_id_counters[node_type]
        
        if node_type == 0:
            final_cell_embeds[local_id] = cell_df.loc[node_name].values
        elif node_type == 1:
            final_drug_embeds[local_id] = drug_df.loc[node_name.upper()].values
        elif node_type == 2:
            final_gene_embeds[local_id] = gene_embedding_dict[node_name.upper()]
        
        local_id_counters[node_type] += 1

    np.save(os.path.join(output_path, 'cell_embeddings.npy'), final_cell_embeds)
    np.save(os.path.join(output_path, 'drug_embeddings.npy'), final_drug_embeds)
    np.save(os.path.join(output_path, 'gene_embeddings.npy'), final_gene_embeds)
    
    print("  > Successfully created and saved final embedding matrices.")
    
    print("\n" + "="*80)
    print("Preprocessing Complete!")
    print(f"New, filtered dataset is ready at: {output_path}")
    print(f"You can now run main.py using '--data {os.path.basename(output_path)}'")
    print("="*80)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(script_dir)
    
    original_data_path = os.path.join(PROJECT_ROOT, 'data', 'ours')
    embeddings_path = os.path.join(script_dir, 'embeddings')
    
    output_path = os.path.join(PROJECT_ROOT, 'data', 'ours_filtered_chemberta')

    preprocess_data_with_embeddings(original_data_path, embeddings_path, output_path)
