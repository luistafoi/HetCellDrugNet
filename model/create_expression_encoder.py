import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_valid_cell_names(node_file, expression_file, mapping_file):
    """Helper function to get the final list of valid cell names by finding the intersection."""
    original_node_cells = {
        parts[1].upper() for parts in (line.strip().split('\t') for line in open(node_file))
        if len(parts) == 3 and int(parts[2]) == 0
    }
    map_df = pd.read_csv(mapping_file)
    id_to_name_map = {str(row['depmap_id']): str(row['ccle_name']).upper() for _, row in map_df.iterrows() if pd.notna(row['depmap_id']) and pd.notna(row['ccle_name'])}
    df_expr = pd.read_csv(expression_file)
    expression_cell_names = {id_to_name_map.get(dep_id, '') for dep_id in df_expr.iloc[:, 0].astype(str)}
    return original_node_cells.intersection(expression_cell_names)

def train_expression_encoder(
    node_file,
    expression_file,
    mapping_file,
    output_file,
    # --- Hyperparameters ---
    final_embedding_dim=512,
    epochs=200,
    learning_rate=1e-4,
    batch_size=64
):
    """
    Trains an autoencoder on raw gene expression data to create cell embeddings.
    """
    print("=" * 80)
    print("Generating Cell Embeddings from Expression-Only MLP Encoder")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Identify Valid Cells and Prepare Expression Matrix ---
    print("\n--- Step 1: Loading and aligning data... ---")
    valid_cell_names = get_valid_cell_names(node_file, expression_file, mapping_file)
    final_ordered_cells = sorted(list(valid_cell_names))
    print(f"  > Found {len(final_ordered_cells)} common cell lines to process.")

    df_expr = pd.read_csv(expression_file)
    id_col_name = df_expr.columns[0]
    map_df = pd.read_csv(mapping_file)
    name_to_id_map_rev = {str(row['ccle_name']).upper(): str(row['depmap_id']) for _, row in map_df.iterrows() if pd.notna(row['depmap_id']) and pd.notna(row['ccle_name'])}
    
    ordered_depmap_ids = [name_to_id_map_rev.get(name) for name in final_ordered_cells]
    
    # We only need the expression values for the valid cells
    df_expr_aligned = df_expr.set_index(id_col_name)
    expression_tensor = torch.tensor(df_expr_aligned.loc[ordered_depmap_ids].iloc[:, :].values, dtype=torch.float32)
    
    original_dim = expression_tensor.shape[1]
    print(f"  > Original expression dimension per cell: {original_dim}") # Print original dimension

    # --- Step 2: Normalize Input Features and Create DataLoader ---
    print("\n--- Step 2: Normalizing input features... ---")
    mean = expression_tensor.mean(dim=0, keepdim=True)
    std = expression_tensor.std(dim=0, keepdim=True)
    expression_tensor_normalized = (expression_tensor - mean) / (std + 1e-6)
    print(f"  > Input tensor created and normalized with shape: {expression_tensor_normalized.shape}")
    
    dataset = TensorDataset(expression_tensor_normalized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- Step 3: Define and Train Autoencoder ---
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, input_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(input_dim=original_dim, encoding_dim=final_embedding_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n--- Step 3: Training autoencoder for {epochs} epochs... ---")
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            batch_features = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  > Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    print("  > Training complete.")

    # --- Step 4: Generate and Save Final Embeddings ---
    print("\n--- Step 4: Generating final embeddings using the trained encoder... ---")
    model.eval()
    with torch.no_grad():
        final_cell_embeddings = model.encoder(expression_tensor_normalized.to(device))
    
    print(f"  > Generated final cell embeddings with shape: {final_cell_embeddings.shape}")
    
    np.save(output_file, final_cell_embeddings.cpu().numpy())
    print(f"  > Successfully saved embeddings to: {output_file}")
    print("\n" + "="*80)

if __name__ == '__main__':
    # --- File Paths and Hyperparameters ---
    NODE_FILE = '/data/luis/HetDrugCellNet/data/ours/node.dat'
    EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    MAPPING_FILE = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'
    OUTPUT_FILE = '/data/luis/HetDrugCellNet/model/embeddings/expression_only_cell_embeddings.npy'

    train_expression_encoder(
        NODE_FILE,
        EXPRESSION_FILE,
        MAPPING_FILE,
        OUTPUT_FILE,
        final_embedding_dim=512, # You can tune this
        epochs=200
    )