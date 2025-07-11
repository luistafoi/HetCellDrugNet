import os
import pickle
import pandas as pd
import numpy as np

def load_cell_names_with_embeddings(node_file, expression_file, mapping_file):
    """
    Helper function to determine the exact list of cell lines that have
    expression data available.
    """
    # Get all cell lines from the original node.dat
    original_node_cells = {
        parts[1].upper() for parts in (line.strip().split('\t') for line in open(node_file))
        if len(parts) == 3 and int(parts[2]) == 0
    }
    
    # Get all cell lines from the expression data after mapping IDs
    map_df = pd.read_csv(mapping_file)
    id_to_name_map = {str(row['depmap_id']): str(row['ccle_name']).upper() for _, row in map_df.iterrows() if pd.notna(row['depmap_id']) and pd.notna(row['ccle_name'])}
    
    df_expr = pd.read_csv(expression_file)
    expression_cell_names = {id_to_name_map.get(dep_id, '') for dep_id in df_expr.iloc[:, 0].astype(str)}
    
    # The final valid cell list is the intersection
    return original_node_cells.intersection(expression_cell_names)


def validate_node_coverage(node_file, cell_embedding_file, gene_embedding_file, drug_embedding_file, expression_file, mapping_file):
    """
    Analyzes and reports the coverage between nodes defined in node.dat
    and the available pre-trained embeddings.
    """
    print("=" * 80)
    print("Validating Node Embedding Coverage")
    print("=" * 80)

    # --- 1. Get Node Lists from All Sources ---
    print("\n--- Step 1: Loading node lists from all sources... ---")

    # Get all nodes from the graph definition file
    graph_nodes = {'cells': set(), 'drugs': set(), 'genes': set()}
    with open(node_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                name, n_type = parts[1].upper(), int(parts[2])
                if n_type == 0:
                    graph_nodes['cells'].add(name)
                elif n_type == 1:
                    graph_nodes['drugs'].add(name)
                elif n_type == 2:
                    graph_nodes['genes'].add(name)
    print(f"  > Found {len(graph_nodes['cells'])} cells, {len(graph_nodes['drugs'])} drugs, and {len(graph_nodes['genes'])} genes in node.dat.")

    # Get nodes with embeddings
    cells_with_embeddings = load_cell_names_with_embeddings(node_file, expression_file, mapping_file)
    
    with open(gene_embedding_file, 'rb') as f:
        genes_with_embeddings = {str(k).upper() for k in pickle.load(f).keys()}
        
    df_drug = pd.read_csv(drug_embedding_file)
    drugs_with_embeddings = {str(name).upper() for name in df_drug.iloc[:, 0]}
    
    print(f"  > Found embeddings for {len(cells_with_embeddings)} cells, {len(drugs_with_embeddings)} drugs, and {len(genes_with_embeddings)} genes.")

    # --- 2. Calculate and Report Coverage ---
    print("\n" + "="*80)
    print("Final Coverage Report")
    print("="*80)

    for node_type in ['cells', 'drugs', 'genes']:
        graph_set = graph_nodes[node_type]
        embedding_set = locals()[f"{node_type}_with_embeddings"]
        
        common_nodes = graph_set.intersection(embedding_set)
        
        coverage = (len(common_nodes) / len(graph_set)) * 100 if len(graph_set) > 0 else 0
        
        print(f"\n--- {node_type.capitalize()} ---")
        print(f"  Nodes in graph definition (node.dat): {len(graph_set)}")
        print(f"  Nodes with available embeddings:      {len(embedding_set)}")
        print(f"  Nodes to be kept in final graph:    {len(common_nodes)}")
        print(f"  Coverage:                           {coverage:.2f}%")
        
    print("\n" + "="*80)
    print("This report shows the exact number of nodes that will be included")
    print("in the final dataset after running the main preprocessing script.")
    print("="*80)


if __name__ == '__main__':
    # --- File Paths ---
    # The main graph definition file
    NODE_FILE = '/data/luis/HetDrugCellNet/data/ours/node.dat'

    # The three sources for embeddings
    CELL_EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/mlp_cell_embeddings_512.npy'
    GENE_EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/gene_embeddings_esm_by_symbol.pkl'
    DRUG_EMBEDDING_FILE = '/data/luis/MoleculeSTM/drugs_with_embeddings.csv' # Confirmed path

    # Helper files needed to identify the cell lines corresponding to the cell embeddings
    EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    MAPPING_FILE = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'

    validate_node_coverage(
        NODE_FILE,
        CELL_EMBEDDING_FILE,
        GENE_EMBEDDING_FILE,
        DRUG_EMBEDDING_FILE,
        EXPRESSION_FILE,
        MAPPING_FILE
    )