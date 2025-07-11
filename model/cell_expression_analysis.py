import os
import pickle
import pandas as pd

def analyze_gene_embedding_coverage(expression_file_path, embedding_file_path):
    """
    Analyzes the coverage of pre-trained gene embeddings against a gene
    expression data file from DepMap.
    """
    print("="*80)
    print("Starting Gene Embedding Coverage Analysis")
    print("="*80)

    # --- 1. Load Gene Names from Expression File ---
    if not os.path.exists(expression_file_path):
        print(f"FATAL: Expression file not found at: {expression_file_path}")
        return

    print(f"\n--- Reading genes from: {os.path.basename(expression_file_path)} ---")
    try:
        # The first column is the ModelID/ProfileID, so we skip it [1:].
        # We only need the header to get the column names.
        df_expr = pd.read_csv(expression_file_path, nrows=0)
        expression_gene_columns = df_expr.columns[1:]

        # Genes in the DepMap file are "SYMBOL (EntrezID)".
        # We parse the symbol part and convert to uppercase for consistent matching.
        expression_genes = set()
        for gene_col in expression_gene_columns:
            symbol = gene_col.split(' ')[0]
            expression_genes.add(symbol.upper())

        print(f"  > Found {len(expression_genes)} unique protein-coding genes in the expression file.")

    except Exception as e:
        print(f"FATAL: Could not process the expression file. Error: {e}")
        return


    # --- 2. Load Gene Names from Embedding File ---
    if not os.path.exists(embedding_file_path):
        print(f"FATAL: Embedding file not found at: {embedding_file_path}")
        return

    print(f"\n--- Reading embeddings from: {os.path.basename(embedding_file_path)} ---")
    try:
        with open(embedding_file_path, 'rb') as f:
            embedding_dict = pickle.load(f)
        # The keys of the dictionary are the gene symbols.
        # Convert to uppercase for consistent matching.
        embedding_genes = {str(k).upper() for k in embedding_dict.keys()}
        print(f"  > Found {len(embedding_genes)} unique genes in the embedding file.")

    except Exception as e:
        print(f"FATAL: Could not process the embedding file. Error: {e}")
        return


    # --- 3. Calculate and Report Coverage ---
    print("\n" + "="*80)
    print("Coverage Analysis Results")
    print("="*80)

    if not expression_genes:
        print("No genes were found in the expression file to analyze.")
        return

    # Find the intersection of the two sets of genes
    common_genes = expression_genes.intersection(embedding_genes)
    common_genes_count = len(common_genes)

    # Calculate coverage percentage
    coverage_percentage = (common_genes_count / len(expression_genes)) * 100

    print(f"\nTotal Genes in Expression File: {len(expression_genes)}")
    print(f"Total Genes in Embedding File:    {len(embedding_genes)}")
    print("-" * 40)
    print(f"Number of Common Genes:           {common_genes_count}")
    print(f"Coverage Percentage:              {coverage_percentage:.2f}%")
    print("\nThis means that {:.2f}% of the protein-coding genes in your expression dataset".format(coverage_percentage))
    print("have a corresponding pre-trained embedding.")
    print("-" * 80)


if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THESE FILE PATHS ---
    # Please update these paths to point to the correct files on your system.

    # Path to the DepMap gene expression file
    EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'

    # Path to your pickle file containing the gene embeddings
    EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/gene_embeddings_esm_by_symbol.pkl'

    # Run the analysis
    analyze_gene_embedding_coverage(EXPRESSION_FILE, EMBEDDING_FILE)

    import os
import pandas as pd

def analyze_cell_line_coverage_with_mapping(node_file_path, expression_file_path, mapping_file_path):
    """
    Analyzes cell line coverage by first mapping DepMap IDs from an expression
    file to CCLE names, then comparing against a node.dat file.
    """
    print("="*80)
    print("Starting Cell Line Coverage Analysis with ID Mapping")
    print("="*80)

    # --- 1. Load the DepMap ID to CCLE Name Mapping ---
    if not os.path.exists(mapping_file_path):
        print(f"FATAL: Mapping file not found at: {mapping_file_path}")
        return

    print(f"\n--- Reading ID map from: {os.path.basename(mapping_file_path)} ---")
    try:
        map_df = pd.read_csv(mapping_file_path)
        # Create a dictionary mapping depmap_id -> ccle_name
        # Standardize both to uppercase to ensure consistent matching
        id_to_name_map = {
            str(depmap_id).upper(): str(ccle_name).upper()
            for depmap_id, ccle_name in zip(map_df['depmap_id'], map_df['ccle_name'])
        }
        print(f"  > Created a map for {len(id_to_name_map)} ID-to-Name pairs.")

    except Exception as e:
        print(f"FATAL: Could not process the mapping file. Error: {e}")
        return


    # --- 2. Load Cell Line Names from node.dat File ---
    if not os.path.exists(node_file_path):
        print(f"FATAL: Node file not found at: {node_file_path}")
        return

    print(f"\n--- Reading cell lines from: {os.path.basename(node_file_path)} ---")
    node_dat_cells = set()
    try:
        with open(node_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3 and int(parts[2]) == 0: # Assumes node type '0' is cell
                    node_dat_cells.add(parts[1].upper())
        print(f"  > Found {len(node_dat_cells)} unique cell line nodes in node.dat.")

    except Exception as e:
        print(f"FATAL: Could not process the node.dat file. Error: {e}")
        return


    # --- 3. Load and Translate Cell Lines from Expression File ---
    if not os.path.exists(expression_file_path):
        print(f"FATAL: Expression file not found at: {expression_file_path}")
        return

    print(f"\n--- Reading and mapping IDs from: {os.path.basename(expression_file_path)} ---")
    try:
        df_expr = pd.read_csv(expression_file_path, usecols=[0])
        depmap_ids_from_expr = df_expr.iloc[:, 0].str.upper().unique()

        # Translate DepMap IDs to CCLE names using the map
        expression_file_cells = set()
        unmapped_ids = 0
        for depmap_id in depmap_ids_from_expr:
            if depmap_id in id_to_name_map:
                expression_file_cells.add(id_to_name_map[depmap_id])
            else:
                unmapped_ids += 1
        
        print(f"  > Found {len(depmap_ids_from_expr)} unique DepMap IDs in the expression file.")
        if unmapped_ids > 0:
            print(f"  > WARNING: {unmapped_ids} IDs could not be mapped to a CCLE name.")
        print(f"  > Resulted in {len(expression_file_cells)} unique mapped CCLE names.")

    except Exception as e:
        print(f"FATAL: Could not process the expression file. Error: {e}")
        return


    # --- 4. Calculate and Report Coverage ---
    print("\n" + "="*80)
    print("Coverage Analysis Results")
    print("="*80)

    if not node_dat_cells:
        print("No cell lines were found in the node.dat file to analyze.")
        return

    common_cells = node_dat_cells.intersection(expression_file_cells)
    common_cells_count = len(common_cells)
    coverage_percentage = (common_cells_count / len(node_dat_cells)) * 100 if len(node_dat_cells) > 0 else 0

    print(f"\nTotal Cell Lines in node.dat:   {len(node_dat_cells)}")
    print(f"Total Mapped Cell Lines in Expression File: {len(expression_file_cells)}")
    print("-" * 40)
    print(f"Number of Common Cell Lines:      {common_cells_count}")
    print(f"Coverage Percentage:              {coverage_percentage:.2f}%")
    print("\nThis means that {:.2f}% of the cell lines in your graph".format(coverage_percentage))
    print("have corresponding gene expression data available after ID mapping.")
    print("-" * 80)


if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THESE FILE PATHS ---
    
    # Path to the file that maps depmap_id to ccle_name
    MAPPING_FILE = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'

    # Path to your graph's node definition file
    NODE_FILE = '/data/luis/HetDrugCellNet/data/ours/node.dat'

    # Path to the DepMap gene expression file
    EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'

    # Run the analysis
    analyze_cell_line_coverage_with_mapping(NODE_FILE, EXPRESSION_FILE, MAPPING_FILE)