import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def generate_filtered_cell_embeddings_with_mlp(
    node_file,
    expression_file,
    embedding_file,
    mapping_file,
    output_file,
    # --- Hyperparameters for the MLP ---
    final_embedding_dim=512,
    epochs=150,
    learning_rate=1e-4,
    batch_size=64
):
    """
    Generates cell line embeddings for a filtered list of valid cells by
    training an autoencoder on weighted gene embeddings.
    """
    print("=" * 80)
    print("Generating Filtered Cell Embeddings with MLP Encoder")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Load All Necessary Data ---
    print("\n--- Step 1: Loading all required data files... ---")
    with open(embedding_file, 'rb') as f:
        embedding_dict = {str(k).upper(): v for k, v in pickle.load(f).items()}
    df_expr = pd.read_csv(expression_file)
    id_col_name = df_expr.columns[0]
    map_df = pd.read_csv(mapping_file)

    # --- Step 2: Identify the Set of Valid, Common Cell Lines ---
    print("\n--- Step 2: Identifying common cell lines to process ---")
    original_node_cells = {
        parts[1].upper() for parts in (line.strip().split('\t') for line in open(node_file))
        if len(parts) == 3 and int(parts[2]) == 0
    }
    id_to_name_map = {str(row['depmap_id']): str(row['ccle_name']).upper() for _, row in map_df.iterrows() if pd.notna(row['depmap_id']) and pd.notna(row['ccle_name'])}
    expression_cell_names = {id_to_name_map.get(dep_id, '') for dep_id in df_expr.iloc[:, 0].astype(str)}
    
    # **CORRECTED LOGIC**: The final valid cell list is the intersection
    valid_cell_names = original_node_cells.intersection(expression_cell_names)
    final_ordered_cells = sorted(list(valid_cell_names))
    print(f"  > Found {len(final_ordered_cells)} common cell lines to generate embeddings for.")

    # --- Step 3: Align Data for the 883 Valid Cells ---
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

    # --- Step 4: Create and Normalize Input Features ---
    print("\n--- Step 4: Creating and normalizing input features... ---")
    num_cells = expression_tensor.shape[0]
    input_feature_dim = aligned_gene_embeddings.shape[1]
    initial_cell_features = torch.zeros(num_cells, input_feature_dim, device=device)
    
    for i in range(0, num_cells, batch_size):
        expression_batch = expression_tensor[i:min(i + batch_size, num_cells)]
        weighted_batch = expression_batch.unsqueeze(2) * aligned_gene_embeddings.unsqueeze(0)
        initial_cell_features[i:min(i + batch_size, num_cells)] = weighted_batch.sum(dim=1)
        
    mean = initial_cell_features.mean(dim=0, keepdim=True)
    std = initial_cell_features.std(dim=0, keepdim=True)
    initial_cell_features = (initial_cell_features - mean) / (std + 1e-6)
    print(f"  > Created and normalized initial features with shape: {initial_cell_features.shape}")
    
    # --- Step 5: Define and Train Autoencoder ---
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, encoding_dim))
            self.decoder = nn.Sequential(nn.Linear(encoding_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(input_dim=input_feature_dim, encoding_dim=final_embedding_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(TensorDataset(initial_cell_features), batch_size=batch_size, shuffle=True)

    print(f"\n--- Step 5: Training autoencoder for {epochs} epochs... ---")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(data[0]), data[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  > Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}")
    print("  > Training complete.")

    # --- Step 6: Generate and Save Final Embeddings ---
    print("\n--- Step 6: Generating final embeddings... ---")
    model.eval()
    with torch.no_grad():
        final_cell_embeddings = model.encoder(initial_cell_features)
    
    print(f"  > Generated final cell embeddings with shape: {final_cell_embeddings.shape}")
    
    np.save(output_file, final_cell_embeddings.cpu().numpy())
    print(f"  > Successfully saved embeddings to: {output_file}")
    print("\n" + "="*80)


if __name__ == '__main__':
    # --- File Paths and Hyperparameters ---
    NODE_FILE = '/data/luis/HetDrugCellNet/data/ours/node.dat'
    EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/gene_embeddings_esm_by_symbol.pkl'
    MAPPING_FILE = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'
    OUTPUT_FILE = '/data/luis/HetDrugCellNet/model/embeddings/mlp_cell_embeddings.npy'

    generate_filtered_cell_embeddings_with_mlp(
        NODE_FILE,
        EXPRESSION_FILE,
        EMBEDDING_FILE,
        MAPPING_FILE,
        OUTPUT_FILE,
        # You can tune these hyperparameters
        final_embedding_dim=512,
        epochs=150,
        learning_rate=1e-4,
        batch_size=64
    )