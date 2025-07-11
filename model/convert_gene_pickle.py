import pandas as pd
import pickle
import os

def convert_pickle_keys():
    """
    Converts a pickle file with integer Entrez IDs as keys to a new
    pickle file with gene symbols as keys, using a mapping CSV.
    """
    print("="*80)
    print("--- Starting Gene Embedding Pickle Key Conversion ---")
    print("="*80)

    # --- 1. Define File Paths ---
    # Use absolute paths for reliability
    PROJECT_ROOT = '/data/luis/HetDrugCellNet'
    PROCESSING_DIR = '/data/luis/HetGNN Data Processing'

    # Input files
    mapping_file = os.path.join(PROCESSING_DIR, 'gene_entrez_sequence.csv')
    original_pickle_file = os.path.join(PROJECT_ROOT, 'model', 'embeddings', 'gene_embeddings_esm.pkl')

    # Output file
    output_pickle_file = os.path.join(PROJECT_ROOT, 'model', 'embeddings', 'gene_embeddings_esm_by_symbol.pkl')

    try:
        # --- 2. Create the ID-to-Symbol Map ---
        print(f"Reading mapping file from: {mapping_file}")
        map_df = pd.read_csv(mapping_file)
        
        # --- THIS IS THE FIX ---
        # 1. Clean up column names by removing leading/trailing whitespace.
        map_df.columns = map_df.columns.str.strip()
        
        # 2. Use the correct column name 'entrez' instead of 'entrez_id'.
        id_col_name = 'entrez' 
        # --- END OF FIX ---

        # Create a dictionary: {entrez_id: gene_symbol}
        map_df.dropna(subset=['gene', id_col_name], inplace=True)
        map_df[id_col_name] = map_df[id_col_name].astype(int)
        id_to_symbol_map = pd.Series(map_df.gene.values, index=map_df[id_col_name]).to_dict()
        print(f"  > Created map with {len(id_to_symbol_map)} entries.")

        # --- 3. Load the Original Pickle File ---
        print(f"Loading original embeddings from: {original_pickle_file}")
        with open(original_pickle_file, 'rb') as f:
            original_embedding_dict = pickle.load(f)
        print(f"  > Loaded {len(original_embedding_dict)} embeddings.")

        # --- 4. Create the New Dictionary with Symbol Keys ---
        new_embedding_dict = {}
        keys_not_found = 0
        for gene_id, embedding_vector in original_embedding_dict.items():
            gene_symbol = id_to_symbol_map.get(gene_id)
            
            if gene_symbol:
                new_embedding_dict[gene_symbol] = embedding_vector
            else:
                keys_not_found += 1
        
        print(f"  > Converted {len(new_embedding_dict)} keys to gene symbols.")
        if keys_not_found > 0:
            print(f"  > Warning: {keys_not_found} Entrez IDs did not have a corresponding symbol in the mapping file.")

        # --- 5. Save the New Pickle File ---
        print(f"Saving new embedding file to: {output_pickle_file}")
        with open(output_pickle_file, 'wb') as f_out:
            pickle.dump(new_embedding_dict, f_out)

        print("\n" + "="*80)
        print("Conversion Complete!")
        print(f"New file created at: {output_pickle_file}")
        print("You can now update your analysis/preprocessing scripts to use this new file.")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a file. Please check your paths.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    convert_pickle_keys()
