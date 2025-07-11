import pandas as pd
import os
import pickle
from collections import Counter

def analyze_all_coverage():
    """
    Analyzes and reports the overlap between nodes defined in the project's node.dat
    and the nodes available in the various pre-trained embedding files for all types.
    """
    print("="*80)
    print("--- Starting Full Data Coverage Analysis ---")
    print("="*80)

    # --- 1. Define Absolute File Paths ---
    PROJECT_ROOT = '/data/luis/HetDrugCellNet'
    
    # Project data files
    node_file_path = os.path.join(PROJECT_ROOT, 'data', 'ours', 'node.dat')
    
    # Embedding files (assuming 'embeddings' folder is inside 'model')
    embeddings_dir = os.path.join(PROJECT_ROOT, 'model', 'embeddings')
    cell_embedding_file = os.path.join(embeddings_dir, 'cell_embeddings_vae.csv')
    cell_line_map_file = os.path.join(embeddings_dir, 'cell_line_mapping.csv')
    
    # --- THIS IS THE FIX: Point to the NEW pickle file with symbol keys ---
    gene_embedding_file = os.path.join(embeddings_dir, 'gene_embeddings_esm_by_symbol.pkl')
    # --- END OF FIX ---
    
    # New path for MoleculeSTM drug embeddings
    drug_embedding_file = '/data/luis/MoleculeSTM/drugs_with_embeddings.csv'

    # --- 2. Load All Project Nodes from node.dat ---
    project_nodes = {
        0: set(), # Cells
        1: set(), # Drugs
        2: set()  # Genes
    }
    node_type_names = {0: 'Cell', 1: 'Drug', 2: 'Gene'}

    print(f"Reading all nodes from: {node_file_path}")
    with open(node_file_path, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split('\t')
                node_name = parts[1]
                node_type = int(parts[2])
                if node_type in project_nodes:
                    # Standardize drug and gene names to uppercase for consistent matching
                    if node_type == 1 or node_type == 2:
                        project_nodes[node_type].add(node_name.upper())
                    else:
                        project_nodes[node_type].add(node_name)
            except (ValueError, IndexError):
                continue
    print("Finished reading project nodes.\n")

    # --- 3. Analyze Each Node Type ---

    def print_report(node_type_name, project_set, embedding_set):
        """Helper function to print a standardized coverage report."""
        overlap_count = len(project_set.intersection(embedding_set))
        project_count = len(project_set)
        embedding_count = len(embedding_set)
        coverage = (overlap_count / project_count) * 100 if project_count > 0 else 0

        print(f"\n--- {node_type_name} Coverage Report ---")
        print(f"Total unique {node_type_name.lower()}s in your project (node.dat): {project_count}")
        print(f"Total unique {node_type_name.lower()}s in your embedding file:      {embedding_count}")
        print("-" * 50)
        print(f"Number of overlapping {node_type_name.lower()}s:                    {overlap_count}")
        print(f"Coverage Percentage:                            {coverage:.2f}%")
        print("--------------------------------------------------")

    # --- Cell Analysis ---
    try:
        print(f"Reading cell embeddings from: {cell_embedding_file}")
        cell_df = pd.read_csv(cell_embedding_file, index_col=0)
        map_df = pd.read_csv(cell_line_map_file)
        depmap_to_ccle = pd.Series(map_df.ccle_name.values, index=map_df.depmap_id).to_dict()
        cell_df.rename(index=depmap_to_ccle, inplace=True)
        embedding_cell_names = set(cell_df.index)
        print_report("Cell Line", project_nodes[0], embedding_cell_names)
    except FileNotFoundError as e:
        print(f"\nERROR loading cell data: {e}")
    
    # --- Drug Analysis ---
    try:
        print(f"\nReading drug embeddings from: {drug_embedding_file}")
        drug_df = pd.read_csv(drug_embedding_file, index_col=0)
        drug_df.index = drug_df.index.str.upper() # Standardize to uppercase
        drug_df = drug_df[~drug_df.index.duplicated(keep='first')] # Remove duplicates
        embedding_drug_names = set(drug_df.index)
        print_report("Drug", project_nodes[1], embedding_drug_names)
    except FileNotFoundError as e:
        print(f"\nERROR loading drug data: {e}")

    # --- Gene Analysis ---
    try:
        print(f"\nReading gene embeddings from: {gene_embedding_file}")
        with open(gene_embedding_file, 'rb') as f:
            gene_embedding_dict = pickle.load(f)
        
        # Convert keys from the pickle file to uppercase for case-insensitive matching
        embedding_gene_names = {str(key).upper() for key in gene_embedding_dict.keys()}

        print_report("Gene/Protein", project_nodes[2], embedding_gene_names)
    except FileNotFoundError as e:
        print(f"\nERROR loading gene data: {e}")


if __name__ == '__main__':
    analyze_all_coverage()
