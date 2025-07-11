import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import json

def get_valid_cell_names(node_file, expression_file, mapping_file):
    """Helper function to get the final list of valid cell names."""
    original_node_cells = {
        parts[1].upper() for parts in (line.strip().split('\t') for line in open(node_file))
        if len(parts) == 3 and int(parts[2]) == 0
    }
    map_df = pd.read_csv(mapping_file)
    id_to_name_map = {str(row['depmap_id']): str(row['ccle_name']).upper() for _, row in map_df.iterrows() if pd.notna(row['depmap_id']) and pd.notna(row['ccle_name'])}
    df_expr = pd.read_csv(expression_file)
    expression_cell_names = {id_to_name_map.get(dep_id, '') for dep_id in df_expr.iloc[:, 0].astype(str)}
    return original_node_cells.intersection(expression_cell_names)


def preprocess_data_with_embeddings(original_data_path, output_path):
    """
    Filters the graph to include only nodes with valid pre-trained embeddings
    for all types and generates a new, clean dataset directory.
    """
    print("="*80)
    print("Starting Self-Contained Data Preprocessing")
    print("="*80)

    # --- 1. Define File Paths ---
    node_file_in = os.path.join(original_data_path, 'node.dat')
    link_file_in = os.path.join(original_data_path, 'link.dat')
    link_test_file_in = os.path.join(original_data_path, 'link.dat.test')
    
    cell_embedding_file = '/data/luis/HetDrugCellNet/model/embeddings/final_vae_cell_embeddings.npy'
    drug_embedding_file = '/data/luis/MoleculeSTM/drugs_with_embeddings.csv'
    gene_embedding_file = '/data/luis/HetDrugCellNet/model/embeddings/gene_embeddings_esm_by_symbol.pkl'
    expression_file = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    mapping_file = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'

    os.makedirs(output_path, exist_ok=True)
    print(f"Output Path: {output_path}\n")

    # --- 2. Load Node "Whitelists" and Original Counts ---
    print("--- Step 1: Identifying nodes with embeddings and getting original counts ---")
    
    # **NEW**: Get original counts from node.dat first
    original_node_counts = Counter()
    with open(node_file_in, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    original_node_counts[int(parts[2])] += 1
            except: continue
    
    cell_names_with_embeddings = get_valid_cell_names(node_file_in, expression_file, mapping_file)
    print(f"  > Identified {len(cell_names_with_embeddings)} valid cell lines to keep.")

    drug_df = pd.read_csv(drug_embedding_file, index_col=0)
    drug_df.index = drug_df.index.str.upper()
    drug_df = drug_df.groupby(drug_df.index).mean()
    drug_names_with_embeddings = set(drug_df.index)
    print(f"  > Identified {len(drug_names_with_embeddings)} drugs with embeddings.")

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
                g_id, n_type, name_upper = int(g_id), int(n_type_str), name.upper()
                if n_type == 0 and name_upper in cell_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': 0}
                elif n_type == 1 and name_upper in drug_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': 1}
                elif n_type == 2 and name_upper in gene_names_with_embeddings:
                    allowed_nodes[g_id] = {'name': name, 'type': 2}
            except (ValueError, IndexError):
                continue
    print(f"  > Total nodes to keep in the new graph: {len(allowed_nodes)}")

    # --- 4. NEW: Detailed Filtering Report ---
    print("\n" + "-"*40)
    print("Filtering Summary Report")
    print("-" * 40)
    final_kept_counts = Counter(info['type'] for info in allowed_nodes.values())
    node_type_names = {0: 'Cells', 1: 'Drugs', 2: 'Genes'}
    for type_id, type_name in node_type_names.items():
        original_count = original_node_counts.get(type_id, 0)
        kept_count = final_kept_counts.get(type_id, 0)
        percentage = (kept_count / original_count) * 100 if original_count > 0 else 0
        print(f"\n--- {type_name} ---")
        print(f"  Original count in node.dat: {original_count}")
        print(f"  Final count to be kept:     {kept_count}")
        print(f"  Percentage Kept:            {percentage:.2f}%")
    print("-" * 40 + "\n")

    # --- 5. Create New, Filtered Graph Files ---
    print("--- Step 3: Generating new, filtered graph files ---")
    # (The rest of the script is unchanged)
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
        links_written = 0
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                try:
                    h_id, t_id, r_id, weight = line.strip().split('\t')
                    if int(h_id) in id_map and int(t_id) in id_map:
                        f_out.write(f"{id_map[int(h_id)]}\t{id_map[int(t_id)]}\t{r_id}\t{weight}\n")
                        links_written += 1
                except (ValueError, IndexError): continue
        print(f"  > Wrote {links_written} links to new {os.path.basename(output_file)}.")
    
    filter_link_file(link_file_in, os.path.join(output_path, 'link.dat'), old_to_new_id_map)
    filter_link_file(link_test_file_in, os.path.join(output_path, 'link.dat.test'), old_to_new_id_map)

    # --- 6. Create and Save Final, Filtered Embedding Matrices ---
    print("\n--- Step 4: Creating and saving final filtered embedding matrices ---")
    
    valid_cell_names_ordered = sorted(list(cell_names_with_embeddings))
    cell_embeddings_loaded = np.load(cell_embedding_file)
    cell_embed_dict = {name: vec for name, vec in zip(valid_cell_names_ordered, cell_embeddings_loaded)}
    
    filtered_node_counts = Counter(info['type'] for info in new_filtered_nodes)
    final_cell_embeds = np.zeros((filtered_node_counts.get(0, 0), cell_embeddings_loaded.shape[1]))
    final_drug_embeds = np.zeros((filtered_node_counts.get(1, 0), drug_df.shape[1]))
    final_gene_embeds = np.zeros((filtered_node_counts.get(2, 0), next(iter(gene_embedding_dict.values())).shape[0]))

    local_id_counters = Counter()
    for node_info in new_filtered_nodes:
        node_type = node_info['type']
        node_name_upper = node_info['name'].upper()
        local_id = local_id_counters[node_type]
        
        if node_type == 0:
            final_cell_embeds[local_id] = cell_embed_dict[node_name_upper]
        elif node_type == 1:
            final_drug_embeds[local_id] = drug_df.loc[node_name_upper].values
        elif node_type == 2:
            final_gene_embeds[local_id] = gene_embedding_dict[node_name_upper]
            
        local_id_counters[node_type] += 1

    np.save(os.path.join(output_path, 'cell_embeddings.npy'), final_cell_embeds)
    np.save(os.path.join(output_path, 'drug_embeddings.npy'), final_drug_embeds)
    np.save(os.path.join(output_path, 'gene_embeddings.npy'), final_gene_embeds)
    print("  > Successfully saved final .npy embedding files.")

    # --- 7. Create info.dat file ---
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
    output_path = os.path.join(PROJECT_ROOT, 'data', 'ours_filtered_moleculestm_new_vae')

    preprocess_data_with_embeddings(original_data_path, output_path)