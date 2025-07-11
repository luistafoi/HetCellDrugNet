import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import json

def preprocess_data_with_embeddings(original_data_path, embeddings_path, output_path):
    """
    Filters the graph to include only nodes with valid pre-trained embeddings
    (using VAE for cells) and generates a new, clean dataset directory.
    """
    print("="*80)
    print("Starting Data Preprocessing for VAE Cell Embeddings")
    print("="*80)

    # --- 1. Define File Paths ---
    node_file_in = os.path.join(original_data_path, 'node.dat')
    link_file_in = os.path.join(original_data_path, 'link.dat')
    link_test_file_in = os.path.join(original_data_path, 'link.dat.test')
    
    # Paths for the VAE + MoleculeSTM + ESM3 experiment
    cell_embedding_file = os.path.join(embeddings_path, 'cell_embeddings_vae.csv')
    cell_line_map_file = os.path.join(embeddings_path, 'cell_line_mapping.csv')
    drug_embedding_file = '/data/luis/MoleculeSTM/drugs_with_embeddings.csv'
    gene_embedding_file = os.path.join(embeddings_path, 'gene_embeddings_esm_by_symbol.pkl')

    os.makedirs(output_path, exist_ok=True)
    print(f"Output Path: {output_path}\n")

    # --- 2. Load Node "Whitelists" from All Embedding Sources ---
    print("--- Step 1: Identifying all nodes with available embeddings ---")
    
    # Load cell embedding names from the VAE CSV file's index
    cell_df = pd.read_csv(cell_embedding_file, index_col=0)
    map_df = pd.read_csv(cell_line_map_file)
    depmap_to_ccle = pd.Series(map_df.ccle_name.values, index=map_df.depmap_id).to_dict()
    cell_df.rename(index=depmap_to_ccle, inplace=True)
    cell_names_with_embeddings = set(cell_df.index)
    print(f"  > Identified {len(cell_names_with_embeddings)} cells with VAE embeddings.")

    # Load drug embedding names
    drug_df = pd.read_csv(drug_embedding_file, index_col=0)
    drug_df.index = drug_df.index.str.upper()
    drug_df = drug_df.groupby(drug_df.index).mean() # Average any duplicates
    drug_names_with_embeddings = set(drug_df.index)
    print(f"  > Identified {len(drug_names_with_embeddings)} drugs with embeddings.")

    # Load gene embedding names
    with open(gene_embedding_file, 'rb') as f:
        gene_embedding_dict = {str(k).upper(): v for k, v in pickle.load(f).items()}
    gene_names_with_embeddings = set(gene_embedding_dict.keys())
    print(f"  > Identified {len(gene_names_with_embeddings)} genes with embeddings.\n")

    # --- 3. Filter Nodes Based on Whitelists ---
    print("--- Step 2: Filtering nodes from original node.dat ---")
    allowed_nodes = {}
    with open(node_file_in, 'r') as f:
        for line in f:
            try:
                g_id, name, n_type_str = line.strip().split('\t')
                g_id, n_type = int(g_id), int(n_type_str)
                
                if n_type == 0 and name in cell_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': 0}
                elif n_type == 1 and name.upper() in drug_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': 1}
                elif n_type == 2 and name.upper() in gene_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': 2}
            except (ValueError, IndexError):
                continue
    print(f"  > Kept {len(allowed_nodes)} total nodes that have embeddings.")

    # --- 4. Create New, Filtered Graph Files ---
    print("\n--- Step 3: Generating new, filtered graph files ---")
    node_file_out = os.path.join(output_path, 'node.dat')
    old_to_new_id_map = {}
    new_filtered_nodes = []
    with open(node_file_out, 'w') as f_out:
        new_id_counter = 0
        for node_type_id in sorted([0, 1, 2]):
            nodes_of_this_type = {g_id: info for g_id, info in allowed_nodes.items() if info['type'] == node_type_id}
            for old_g_id in sorted(nodes_of_this_type.keys()):
                info = allowed_nodes[old_g_id]
                old_to_new_id_map[old_g_id] = new_id_counter
                new_filtered_nodes.append(info)
                f_out.write(f"{new_id_counter}\t{info['name']}\t{info['type']}\n")
                new_id_counter += 1
    print(f"  > Wrote {len(new_filtered_nodes)} nodes to new node.dat.")
    
    def filter_link_file(input_file, output_file, id_map):
        """Reads an old link file and writes a new one keeping only valid nodes."""
        links_written = 0
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                try:
                    h_id, t_id, r_id, weight = line.strip().split('\t')
                    if int(h_id) in id_map and int(t_id) in id_map:
                        f_out.write(f"{id_map[int(h_id)]}\t{id_map[int(t_id)]}\t{r_id}\t{weight}\n")
                        links_written += 1
                except (ValueError, IndexError):
                    continue
        print(f"  > Wrote {links_written} links to new {os.path.basename(output_file)}.")
    
    filter_link_file(link_file_in, os.path.join(output_path, 'link.dat'), old_to_new_id_map)
    filter_link_file(link_test_file_in, os.path.join(output_path, 'link.dat.test'), old_to_new_id_map)

    # --- 5. Create and Save Final, Filtered Embedding Matrices ---
    print("\n--- Step 4: Creating and saving final filtered embedding matrices ---")
    filtered_node_counts = Counter(info['type'] for info in new_filtered_nodes)
    final_cell_embeds = np.zeros((filtered_node_counts.get(0, 0), cell_df.shape[1]))
    final_drug_embeds = np.zeros((filtered_node_counts.get(1, 0), drug_df.shape[1]))
    final_gene_embeds = np.zeros((filtered_node_counts.get(2, 0), next(iter(gene_embedding_dict.values())).shape[0]))

    local_id_counters = Counter()
    for node_info in new_filtered_nodes:
        node_type = node_info['type']
        local_id = local_id_counters[node_type]
        
        if node_type == 0:
            final_cell_embeds[local_id] = cell_df.loc[node_info['name']].values
        elif node_type == 1:
            final_drug_embeds[local_id] = drug_df.loc[node_info['name'].upper()].values
        elif node_type == 2:
            final_gene_embeds[local_id] = gene_embedding_dict[node_info['name'].upper()]
            
        local_id_counters[node_type] += 1

    np.save(os.path.join(output_path, 'cell_embeddings.npy'), final_cell_embeds)
    np.save(os.path.join(output_path, 'drug_embeddings.npy'), final_drug_embeds)
    np.save(os.path.join(output_path, 'gene_embeddings.npy'), final_gene_embeds)
    print("  > Successfully saved final .npy embedding files.")
    
    # --- 6. Create info.dat file ---
    print("\n--- Step 5: Creating info.dat for schema definition ---")
    info_file_out = os.path.join(output_path, 'info.dat')
    info_content = {"node.dat": {"0": ["cell"], "1": ["drug"], "2": ["gene"]}, "link.dat": {"0": ["cell", "drug", "c-d"], "1": ["drug", "cell", "d-c"]}}
    with open(info_file_out, 'w') as f:
        json.dump(info_content, f, indent=4)
    print(f"  > Successfully created {os.path.basename(info_file_out)}")
    
    print("\n" + "="*80)
    print(f"Preprocessing Complete! New dataset at: {output_path}")
    print("="*80)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(script_dir)
    original_data_path = os.path.join(PROJECT_ROOT, 'data', 'ours')
    embeddings_path = os.path.join(script_dir, 'embeddings')
    
    # We'll create a new, clean directory for this experiment
    output_path = os.path.join(PROJECT_ROOT, 'data', 'ours_filtered_moleculestm_vae')

    preprocess_data_with_embeddings(original_data_path, embeddings_path, output_path)