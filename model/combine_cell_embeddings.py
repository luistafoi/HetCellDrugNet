import os
import numpy as np

def combine_embeddings(embedding_file_1, embedding_file_2, output_file):
    """
    Combines two sets of embeddings into a single embedding file by concatenation.
    
    Args:
        embedding_file_1 (str): Path to the first .npy embedding file.
        embedding_file_2 (str): Path to the second .npy embedding file.
        output_file (str): Path to save the final combined embeddings.
    """
    print("=" * 80)
    print("Combining Cell Embedding Files")
    print("=" * 80)

    # --- 1. Load Both Embedding Files ---
    print("--- Step 1: Loading embedding files... ---")
    try:
        embeds_1 = np.load(embedding_file_1)
        embeds_2 = np.load(embedding_file_2)
        print(f"  > Loaded Embedding Set 1 with shape: {embeds_1.shape}")
        print(f"  > Loaded Embedding Set 2 with shape: {embeds_2.shape}")
    except FileNotFoundError as e:
        print(f"FATAL: Could not find an input file. Please ensure both embedding files exist. Error: {e}")
        return

    # --- 2. Validate and Combine ---
    print("\n--- Step 2: Validating and concatenating embeddings... ---")
    
    # Check if the number of cells (rows) is the same in both files
    if embeds_1.shape[0] != embeds_2.shape[0]:
        print(f"FATAL: The number of rows does not match between files! "
              f"({embeds_1.shape[0]} vs {embeds_2.shape[0]})")
        return

    # Concatenate the arrays side-by-side (along the feature axis)
    combined_embeds = np.concatenate((embeds_1, embeds_2), axis=1)
    
    print(f"  > Successfully concatenated embeddings.")
    print(f"  > Final combined embedding shape: {combined_embeds.shape}")

    # --- 3. Save the Final Combined File ---
    print("\n--- Step 3: Saving final file... ---")
    np.save(output_file, combined_embeds)
    print(f"  > Successfully saved combined embeddings to: {output_file}")
    print("\n" + "="*80)


if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THESE FILE PATHS ---

    # 1. Path to your first cell embedding file (MLP + ESM3)
    EMBEDDING_FILE_1 = '/data/luis/HetDrugCellNet/model/embeddings/mlp_cell_embeddings_512.npy'

    # 2. Path to your second cell embedding file (Expression-Only)
    EMBEDDING_FILE_2 = '/data/luis/HetDrugCellNet/model/embeddings/expression_only_cell_embeddings.npy'
    
    # 3. Desired output file path for the new combined embeddings
    OUTPUT_FILE = '/data/luis/HetDrugCellNet/model/embeddings/combined_cell_embeddings.npy'

    combine_embeddings(EMBEDDING_FILE_1, EMBEDDING_FILE_2, OUTPUT_FILE)