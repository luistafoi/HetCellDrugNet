import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

def generate_filtered_cell_embeddings(
    node_file,
    expression_file,
    embedding_file,
    mapping_file,
    output_embedding_file,
    output_cell_list_file  # New argument for the list of names
):
    """
    Generates cell line embeddings ONLY for the intersection of cells found
    in the node file and the expression data.
    """
    print("=" * 80)
    print("Generating Filtered Cell Embeddings from Gene Expression")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load All Necessary Data ---
    print("\n--- Step 1: Loading all required data files ---")
    with open(embedding_file, 'rb') as f:
        embedding_dict = {str(k).upper(): v for k, v in pickle.load(f).items()}
    df_expr = pd.read_csv(expression_file)
    id_col_name = df_expr.columns[0]
    map_df = pd.read_csv(mapping_file)
    
    # --- 2. Identify the Set of Valid, Common Cell Lines ---
    print("\n--- Step 2: Identifying common cell lines to process ---")
    
    # Get all cell lines from the original node.dat
    original_node_cells = {
        parts[1].upper() for parts in (line.strip().split('\t') for line in open(node_file))
        if len(parts) == 3 and int(parts[2]) == 0
    }
    
    # Get all cell lines from the expression data after mapping IDs
    id_to_name_map = {str(row['depmap_id']): str(row['ccle_name']).upper() for _, row in map_df.iterrows() if pd.notna(row['depmap_id']) and pd.notna(row['ccle_name'])}
    expression_cell_names = {id_to_name_map.get(dep_id, '') for dep_id in df_expr.iloc[:, 0].astype(str)}
    
    # **CORRECTED LOGIC**: The final valid cell list is the intersection
    valid_cell_names = original_node_cells.intersection(expression_cell_names)
    final_ordered_cells = sorted(list(valid_cell_names))
    print(f"  > Found {len(final_ordered_cells)} common cell lines to generate embeddings for.")

    # --- 3. Align Data for the 883 Valid Cells ---
    print(f"\n--- Step 3: Aligning data for the {len(final_ordered_cells)} valid cells ---")
    
    # Align Genes
    gene_col_rename_map = {col: col.split(' ')[0].upper() for col in df_expr.columns[1:]}
    df_expr = df_expr.rename(columns=gene_col_rename_map)
    expression_genes = {col for col in df_expr.columns[1:]}
    common_genes = sorted(list(expression_genes & embedding_dict.keys()))
    aligned_gene_embeddings = torch.tensor(np.array([embedding_dict[gene] for gene in common_genes]), dtype=torch.float32, device=device)

    # **CORRECTED LOGIC**: Align Cells using the filtered list of 883
    name_to_id_map_rev = {v: k for k, v in id_to_name_map.items()}
    ordered_depmap_ids = [name_to_id_map_rev.get(name) for name in final_ordered_cells]
    
    df_expr_aligned = df_expr.set_index(id_col_name)
    # Use .loc to select only the valid rows. No reindex/fillna needed for missing cells.
    expression_tensor = torch.tensor(df_expr_aligned.loc[ordered_depmap_ids, common_genes].fillna(0).values, dtype=torch.float32, device=device)
    print(f"  > Final expression matrix created with shape: {expression_tensor.shape}")

    # --- 4. Generate Embeddings via Weighted Average ---
    print("\n--- Step 4: Calculating cell embeddings ---")
    expr_weights = F.softmax(expression_tensor, dim=1)
    cell_embeddings = torch.matmul(expr_weights, aligned_gene_embeddings)
    print(f"  > Generated final cell embeddings with shape: {cell_embeddings.shape}")

    # --- 5. Save the Final Embeddings and the list of cell names ---
    print("\n--- Step 5: Saving new files ---")
    
    # Save the new 883 x 1536 embedding file
    np.save(output_embedding_file, cell_embeddings.cpu().numpy())
    print(f"  > Successfully saved embeddings to: {output_embedding_file}")
    
    # Save the list of corresponding cell names for future use
    with open(output_cell_list_file, 'w') as f:
        for cell_name in final_ordered_cells:
            f.write(f"{cell_name}\n")
    print(f"  > Successfully saved list of cell names to: {output_cell_list_file}")
    print("\n" + "="*80)


if __name__ == '__main__':
    # --- File Paths ---
    NODE_FILE = '/data/luis/HetDrugCellNet/data/ours/node.dat'
    EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/gene_embeddings_esm_by_symbol.pkl'
    MAPPING_FILE = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'
    
    # --- Output Files ---
    # 1. The new cell embeddings file (will have 883 rows)
    OUTPUT_EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/filtered_cell_embeddings.npy'
    # 2. A text file containing the list of the 883 cell names, in order
    OUTPUT_CELL_LIST_FILE = '/data/luis/HetDrugCellNet/model/embeddings/filtered_cell_names.txt'

    generate_filtered_cell_embeddings(
        NODE_FILE,
        EXPRESSION_FILE,
        EMBEDDING_FILE,
        MAPPING_FILE,
        OUTPUT_EMBEDDING_FILE,
        OUTPUT_CELL_LIST_FILE
    )