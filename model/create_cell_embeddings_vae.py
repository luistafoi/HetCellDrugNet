# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # --- VAE Model Definition (from the script you found) ---
# class CellLineVAE(nn.Module):
#     def __init__(self, dims, dor=0.4):
#         super(CellLineVAE, self).__init__()
#         encode_list = []
#         decode_list = []
#         for i in range(len(dims)-1):
#             encode_list.append(nn.Linear(dims[i], dims[i+1]))
#             encode_list.append(nn.BatchNorm1d(dims[i+1]))
#             encode_list.append(nn.ReLU())
#             encode_list.append(nn.Dropout(dor))

#         for i in range(len(dims)-1, 0, -1):
#             decode_list.append(nn.Linear(dims[i], dims[i-1]))
#             decode_list.append(nn.BatchNorm1d(dims[i-1]))
#             decode_list.append(nn.ReLU())
#             decode_list.append(nn.Dropout(dor))
        
#         self.encode_net = nn.Sequential(*encode_list)
#         self.decode_net = nn.Sequential(*decode_list)
#         self.fc_mu = nn.Linear(dims[-1], dims[-1])
#         self.fc_logvar = nn.Linear(dims[-1], dims[-1])

#     def encode(self, x):
#         h = self.encode_net(x)
#         return self.fc_mu(h), self.fc_logvar(h)
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z):
#         return self.decode_net(z)
        
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# def loss_function(recon_x, x, mu, logvar, beta=1.0):
#     MSE = F.mse_loss(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return MSE + beta * KLD

# # --- Main Execution ---
# if __name__ == '__main__':
#     # --- 1. File Paths and Hyperparameters ---
#     NODE_FILE = '/data/luis/HetDrugCellNet/data/ours/node.dat'
#     EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
#     MAPPING_FILE = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'
    
#     OUTPUT_EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/final_vae_cell_embeddings.npy'
#     OUTPUT_CELL_LIST_FILE = '/data/luis/HetDrugCellNet/model/embeddings/final_vae_cell_names.txt'
    
#     # Hyperparameters
#     LATENT_DIM = 512
#     LEARNING_RATE = 1e-4
#     EPOCHS = 100
#     BATCH_SIZE = 64
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # --- 2. Load, Filter, and Prepare Data ---
#     print("\n--- Step 1: Loading and filtering data ---")
    
#     # Identify the 883 valid cell lines
#     original_node_cells = {p[1].upper() for p in (l.strip().split('\t') for l in open(NODE_FILE)) if len(p)==3 and int(p[2])==0}
#     map_df = pd.read_csv(MAPPING_FILE)
#     id_to_name_map = {str(r['depmap_id']): str(r['ccle_name']).upper() for _, r in map_df.iterrows() if pd.notna(r['depmap_id']) and pd.notna(r['ccle_name'])}
#     df_expr = pd.read_csv(EXPRESSION_FILE)
#     expression_cell_names = {id_to_name_map.get(id, '') for id in df_expr.iloc[:, 0].astype(str)}
#     valid_cell_names = original_node_cells.intersection(expression_cell_names)
#     final_ordered_cells = sorted(list(valid_cell_names))
#     print(f"  > Identified {len(final_ordered_cells)} common cell lines to process.")

#     # Filter and align the expression DataFrame
#     name_to_id_map_rev = {v: k for k, v in id_to_name_map.items()}
#     ordered_depmap_ids = [name_to_id_map_rev.get(name) for name in final_ordered_cells]
#     df_expr_aligned = df_expr.set_index(df_expr.columns[0])
#     expression_data = df_expr_aligned.loc[ordered_depmap_ids].values

#     # Normalize the data
#     scaler = StandardScaler()
#     expression_data_scaled = scaler.fit_transform(expression_data)
    
#     # Split data and create DataLoaders
#     X_train, X_val = train_test_split(expression_data_scaled, test_size=0.15, random_state=42)
#     train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
#     train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE)
#     print("  > Data prepared and split into training/validation sets.")

#     # --- 3. Initialize and Train the VAE Model ---
#     input_dim = expression_data_scaled.shape[1]
#     dims = [input_dim, 4096, 1024, LATENT_DIM] # Define the architecture layers
    
#     model = CellLineVAE(dims).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     print(f"\n--- Step 2: Training VAE for {EPOCHS} epochs... ---")
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
#         for batch_idx, data in enumerate(train_loader):
#             x = data[0].to(device)
#             optimizer.zero_grad()
#             recon_x, mu, logvar = model(x)
#             loss = loss_function(recon_x, x, mu, logvar)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
        
#         # Validation step
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for data in val_loader:
#                 x = data[0].to(device)
#                 recon_x, mu, logvar = model(x)
#                 val_loss += loss_function(recon_x, x, mu, logvar).item()
        
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader.dataset):.4f}, Val Loss: {val_loss/len(val_loader.dataset):.4f}")

#     print("  > Training complete.")

#     # --- 4. Generate and Save Final Embeddings ---
#     print("\n--- Step 3: Generating and saving final embeddings... ---")
#     model.eval()
#     full_data_tensor = torch.tensor(expression_data_scaled, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         final_embeddings_mu, _ = model.encode(full_data_tensor)
    
#     np.save(OUTPUT_EMBEDDING_FILE, final_embeddings_mu.cpu().numpy())
#     print(f"  > Saved final ({final_embeddings_mu.shape[0]} x {final_embeddings_mu.shape[1]}) embeddings to: {OUTPUT_EMBEDDING_FILE}")

#     with open(OUTPUT_CELL_LIST_FILE, 'w') as f:
#         for cell_name in final_ordered_cells:
#             f.write(f"{cell_name}\n")
#     print(f"  > Saved list of corresponding cell names to: {OUTPUT_CELL_LIST_FILE}")
#     print("\n" + "="*80)

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

# --- VAE Model Definition (Original Structure) ---
class CellLineVAE(nn.Module):
    def __init__(self, dims, dor=0.4):
        super(CellLineVAE, self).__init__()
        encode_list = []
        decode_list = []
        for i in range(len(dims)-1):
            encode_list.append(nn.Linear(dims[i], dims[i+1]))
            encode_list.append(nn.BatchNorm1d(dims[i+1]))
            encode_list.append(nn.ReLU())
            encode_list.append(nn.Dropout(dor))

        for i in range(len(dims)-1, 0, -1):
            decode_list.append(nn.Linear(dims[i], dims[i-1]))
            decode_list.append(nn.BatchNorm1d(dims[i-1]))
            decode_list.append(nn.ReLU())
            decode_list.append(nn.Dropout(dor))
        
        self.encode_net = nn.Sequential(*encode_list)
        self.decode_net = nn.Sequential(*decode_list)
        self.fc_mu = nn.Linear(dims[-1], dims[-1])
        self.fc_logvar = nn.Linear(dims[-1], dims[-1])

    def encode(self, x):
        h = self.encode_net(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decode_net(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- VAE Loss Function (Original Structure) ---
def loss_function( recon_x, x, mu, logvar, beta=1.0):
    # Using 'sum' reduction makes the loss value large, but the trend is what matters.
    MSE = F.mse_loss(recon_x, x, reduction='sum') 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

# --- Main Execution ---
if __name__ == '__main__':
    # --- START: MODIFIED DATA LOADING ---
    
    # 1. Define File Paths and Hyperparameters
    NODE_FILE = '/data/luis/HetDrugCellNet/data/ours/node.dat'
    EXPRESSION_FILE = '/data/luis/CellDrugNet/data/Datasets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv'
    MAPPING_FILE = '/data/luis/CellDrugNet/data/Datasets/Repurposing_Public_24Q2_Cell_Line_Meta_Data (2)_new.csv'
    
    OUTPUT_EMBEDDING_FILE = '/data/luis/HetDrugCellNet/model/embeddings/final_vae_cell_embeddings.npy'
    OUTPUT_CELL_LIST_FILE = '/data/luis/HetDrugCellNet/model/embeddings/final_vae_cell_names.txt'
    
    LATENT_DIM = 512
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Identify the 883 valid cell lines
    original_node_cells = {p[1].upper() for p in (l.strip().split('\t') for l in open(NODE_FILE)) if len(p)==3 and int(p[2])==0}
    map_df = pd.read_csv(MAPPING_FILE)
    id_to_name_map = {str(r['depmap_id']): str(r['ccle_name']).upper() for _, r in map_df.iterrows() if pd.notna(r['depmap_id']) and pd.notna(r['ccle_name'])}
    df_expr = pd.read_csv(EXPRESSION_FILE)
    expression_cell_names = {id_to_name_map.get(dep_id, '') for dep_id in df_expr.iloc[:, 0].astype(str)}
    valid_cell_names = original_node_cells.intersection(expression_cell_names)
    final_ordered_cells = sorted(list(valid_cell_names))
    print(f"Identified {len(final_ordered_cells)} common cell lines to process.")

    # 3. Filter, Normalize, and Split the Data
    name_to_id_map_rev = {v: k for k, v in id_to_name_map.items()}
    ordered_depmap_ids = [name_to_id_map_rev.get(name) for name in final_ordered_cells]
    df_expr_aligned = df_expr.set_index(df_expr.columns[0])
    expression_data = df_expr_aligned.loc[ordered_depmap_ids].values

    scaler = StandardScaler()
    expression_data_scaled = scaler.fit_transform(expression_data)
    
    X_train, X_val = train_test_split(expression_data_scaled, test_size=0.15, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=BATCH_SIZE)
    print("Data prepared and split into training/validation sets.")
    
    # --- END: MODIFIED DATA LOADING ---

    # --- Initialize Model and Optimizer (Using original architecture) ---
    input_dim = expression_data_scaled.shape[1]
    # Using the deeper layer structure from the original script
    dims = [input_dim, 10000, 5000, 1000, 500, LATENT_DIM] 
    
    model = CellLineVAE(dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    scheduler = StepLR(optimizer, step_size=35, gamma=0.1)

    # --- Train and Validate the Model ---
    print(f"\nTraining VAE for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data in train_loader:
            x = data[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                x = data[0].to(device)
                recon_x, mu, logvar = model(x)
                val_loss += loss_function(recon_x, x, mu, logvar).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader.dataset):.4f}, Val Loss: {val_loss/len(val_loader.dataset):.4f}")

    print("Training complete.")

    # --- Generate and Save Final Embeddings ---
    print("\nGenerating final embeddings...")
    model.eval()
    full_data_tensor = torch.tensor(expression_data_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        final_embeddings_mu, _ = model.encode(full_data_tensor)
    
    np.save(OUTPUT_EMBEDDING_FILE, final_embeddings_mu.cpu().numpy())
    print(f"Saved final embeddings to: {OUTPUT_EMBEDDING_FILE}")

    with open(OUTPUT_CELL_LIST_FILE, 'w') as f:
        for cell_name in final_ordered_cells:
            f.write(f"{cell_name}\n")
    print(f"Saved list of cell names to: {OUTPUT_CELL_LIST_FILE}")