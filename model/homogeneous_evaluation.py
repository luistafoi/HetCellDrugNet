import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, to_undirected, negative_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import networkx as nx
import pickle

# =============================================================================
# 1. SETUP AND DEVICE CONFIGURATION
# =============================================================================
# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# 2. DATA AND EMBEDDING LOADING
# =============================================================================
# --- Load Raw Embeddings from Different Sources ---

# Define paths to the embedding files
# IMPORTANT: Update these paths to match their location on your system.
gene_embedding_path = '/data/luis/HetDrugCellNet/model/embeddings/gene_embeddings_esm_by_symbol.pkl'
drug_embedding_path = '/data/luis/MoleculeSTM/drugs_with_embeddings.csv'
cell_embedding_path = '/data/luis/HetDrugCellNet/model/embeddings/cell_embeddings_vae.csv'

# Load Gene Embeddings (from Pickle)
try:
    with open(gene_embedding_path, 'rb') as f:
        # Convert keys to uppercase for consistent matching
        raw_gene_embeddings = {str(k).upper(): torch.tensor(v, dtype=torch.float32) for k, v in pickle.load(f).items()}
    print(f"Loaded {len(raw_gene_embeddings)} raw gene embeddings.")
except FileNotFoundError:
    print(f"ERROR: Gene embedding file not found at {gene_embedding_path}")
    exit()

# Helper function to load CSV embeddings
def load_csv_embeddings(path, name_col=0, start_col=1, header=0):
    try:
        df = pd.read_csv(path, header=header)
        embeddings_dict = {}
        name_key = df.columns[name_col]
        # Convert names to uppercase for consistent matching
        for _, row in df.iterrows():
            name = str(row[name_key]).upper()
            vec = torch.tensor(row.iloc[start_col:].values.astype(np.float32))
            embeddings_dict[name] = vec
        return embeddings_dict
    except FileNotFoundError:
        print(f"ERROR: Embedding file not found at {path}")
        exit()

# Load Drug and Cell Embeddings (from CSV)
raw_drug_embeddings = load_csv_embeddings(drug_embedding_path)
print(f"Loaded {len(raw_drug_embeddings)} raw drug embeddings.")
raw_cell_embeddings = load_csv_embeddings(cell_embedding_path)
print(f"Loaded {len(raw_cell_embeddings)} raw cell embeddings.")


# --- Load Relationship Datasets ---
# IMPORTANT: Update these paths to match their location on your system.
df_gene_cell = pd.read_csv('/data/luis/Datasets/genetocell.csv')[['cell', 'gene']]
df_cell_drug_raw = pd.read_csv('/data/luis/Datasets/drug_SMILES.csv', index_col = 0)
df_gene_drug = pd.read_csv('/data/luis/Datasets/genetodrug.csv')[['Gene', 'SMILES2']]
df_gene_drug.rename(columns={'SMILES2': 'SMILES'}, inplace=True)
ppi_data = pd.read_csv('/data/luis/Datasets/gene_gene_association.csv')

# Convert cell-drug matrix to an edge list
cell_drug_edges_list = []
for cell in df_cell_drug_raw.index:
    for smiles in df_cell_drug_raw.columns:
        if df_cell_drug_raw.loc[cell, smiles] == 1:
            cell_drug_edges_list.append([cell, smiles])
df_cell_drug = pd.DataFrame(cell_drug_edges_list, columns=['cell', 'SMILES'])


# --- Node Mappings and Feature Preparation ---
genes = np.unique(np.concatenate([df_gene_cell['gene'], df_gene_drug['Gene']]))
cells = np.unique(np.concatenate([df_gene_cell['cell'], df_cell_drug['cell']]))
drugs = np.unique(np.concatenate([df_gene_drug['SMILES'], df_cell_drug['SMILES']]))

print(f"\nUnique nodes found: {len(genes)} Genes, {len(cells)} Cells, {len(drugs)} Drugs")

gene2idx = {gene: i for i, gene in enumerate(genes)}
cell2idx = {cell: i + len(genes) for i, cell in enumerate(cells)}
drug2idx = {drug: i + len(genes) + len(cells) for i, drug in enumerate(drugs)}
num_nodes = len(genes) + len(cells) + len(drugs)

# Get the dimensions of the raw embeddings
gene_feat_dim = next(iter(raw_gene_embeddings.values())).shape[0]
drug_feat_dim = next(iter(raw_drug_embeddings.values())).shape[0]
cell_feat_dim = next(iter(raw_cell_embeddings.values())).shape[0]

# --- Create ordered raw feature tensors ---
# Create tensors for each node type, ordered by their index in the graph.
# If a node doesn't have a pre-trained embedding, its features will remain zeros.

# Genes
gene_features_raw = torch.zeros(len(genes), gene_feat_dim, dtype=torch.float32)
for gene, idx in gene2idx.items():
    if gene.upper() in raw_gene_embeddings:
        gene_features_raw[idx] = raw_gene_embeddings[gene.upper()]

# Cells
cell_features_raw = torch.zeros(len(cells), cell_feat_dim, dtype=torch.float32)
for cell, idx in cell2idx.items():
    local_idx = idx - len(genes)
    if cell.upper() in raw_cell_embeddings:
        cell_features_raw[local_idx] = raw_cell_embeddings[cell.upper()]

# Drugs
drug_features_raw = torch.zeros(len(drugs), drug_feat_dim, dtype=torch.float32)
for drug, idx in drug2idx.items():
    local_idx = idx - len(genes) - len(cells)
    if drug.upper() in raw_drug_embeddings:
        drug_features_raw[local_idx] = raw_drug_embeddings[drug.upper()]

# Move raw features to the correct device
gene_features_raw = gene_features_raw.to(device)
cell_features_raw = cell_features_raw.to(device)
drug_features_raw = drug_features_raw.to(device)

print(f"Raw feature dimensions: Gene={gene_feat_dim}, Cell={cell_feat_dim}, Drug={drug_feat_dim}\n")


# =============================================================================
# 3. GRAPH CONSTRUCTION
# =============================================================================
edges = []

# Gene-cell edges
for _, row in df_gene_cell.iterrows():
    edges.append([gene2idx[row['gene']], cell2idx[row['cell']]])

# Cell-drug edges
for _, row in df_cell_drug.iterrows():
    edges.append([cell2idx[row['cell']], drug2idx[row['SMILES']]])

# Drug-Gene edges
for _, row in df_gene_drug.iterrows():
    edges.append([drug2idx[row['SMILES']], gene2idx[row['Gene']]])

# Add PPI (Gene-Gene) Edges using NetworkX
G = nx.Graph()
for _, row in ppi_data.iterrows():
    if row['Gene1'] in gene2idx and row['Gene2'] in gene2idx:
        G.add_edge(gene2idx[row['Gene1']], gene2idx[row['Gene2']])
G.add_nodes_from(gene2idx.values())
ppi_edge_index = from_networkx(G).edge_index
ppi_edges = ppi_edge_index.cpu().numpy().T
edges.extend(ppi_edges.tolist())

# Convert all edges to a single tensor
edges = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
edges = to_undirected(edges) # Ensure graph is undirected for GCN


# =============================================================================
# 4. DATA SPLITTING (for Link Prediction Task)
# =============================================================================
# Isolate the cell-drug edges, which we want to predict
cell_indices_set = set(cell2idx.values())
drug_indices_set = set(drug2idx.values())

# Create tensors from the sets AND move them to the correct device
cell_indices_tensor = torch.tensor(list(cell_indices_set), device=device)
drug_indices_tensor = torch.tensor(list(drug_indices_set), device=device)

# Now, use these GPU tensors to create the mask
cell_drug_mask = (torch.isin(edges[0], cell_indices_tensor) & torch.isin(edges[1], drug_indices_tensor)) | \
                 (torch.isin(edges[0], drug_indices_tensor) & torch.isin(edges[1], cell_indices_tensor))

cell_drug_edge_index = edges[:, cell_drug_mask]
other_edge_index = edges[:, ~cell_drug_mask]

# Split the cell-drug links into training and testing sets
train_pos_cd_edges, test_pos_cd_edges = train_test_split(cell_drug_edge_index.t().cpu().numpy(), test_size=0.2, random_state=1234)
train_pos_cd_edges = torch.tensor(train_pos_cd_edges, dtype=torch.long).t().to(device)
test_pos_cd_edges = torch.tensor(test_pos_cd_edges, dtype=torch.long).t().to(device)
print(f"Positive cell-drug links: {train_pos_cd_edges.size(1)} Train, {test_pos_cd_edges.size(1)} Test")

# The training graph consists of all non-cell-drug links PLUS the training cell-drug links
train_edge_index = torch.cat([other_edge_index, train_pos_cd_edges], dim=1)
print(f"Total edges in training graph: {train_edge_index.size(1)}")

# Generate negative samples for training and testing
def generate_negative_samples(pos_edges, num_neg_ratio=1.0):
    num_neg_samples = int(pos_edges.size(1) * num_neg_ratio)
    # Using torch_geometric's negative_sampling is efficient
    neg_edge_index = negative_sampling(
        edge_index=edges, # Sample from all possible edges to avoid picking existing ones
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
    )
    return neg_edge_index

train_neg_cd_edges = generate_negative_samples(train_pos_cd_edges)
test_neg_cd_edges = generate_negative_samples(test_pos_cd_edges)
print(f"Negative cell-drug links: {train_neg_cd_edges.size(1)} Train, {test_neg_cd_edges.size(1)} Test\n")


# =============================================================================
# 5. MODEL DEFINITION
# =============================================================================
class GCN(torch.nn.Module):
    def __init__(self, gene_dim, cell_dim, drug_dim, out_dim=128):
        super(GCN, self).__init__()

        # Projection layers remain the same
        self.gene_proj = torch.nn.Linear(gene_dim, out_dim)
        self.cell_proj = torch.nn.Linear(cell_dim, out_dim)
        self.drug_proj = torch.nn.Linear(drug_dim, out_dim)

        # A SINGLE GCN layer
        self.conv1 = GCNConv(out_dim, out_dim)

    def forward(self, raw_feats, node_maps, edge_index):
        raw_gene_feats, raw_cell_feats, raw_drug_feats = raw_feats
        gene_map, cell_map, drug_map = node_maps

        # Project raw features
        gene_embeds = self.gene_proj(raw_gene_feats)
        cell_embeds = self.cell_proj(raw_cell_feats)
        drug_embeds = self.drug_proj(raw_drug_feats)

        # Build the feature matrix
        num_total_nodes = len(gene_map) + len(cell_map) + len(drug_map)
        x = torch.zeros(num_total_nodes, self.conv1.in_channels, device=edge_index.device)

        x[list(gene_map.values())] = gene_embeds
        x[list(cell_map.values())] = cell_embeds
        x[list(drug_map.values())] = drug_embeds

        # Apply the single GCN convolution
        x = self.conv1(x, edge_index)
        
        return x # Return the embeddings after one layer

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

# =============================================================================
# 6. TRAINING AND EVALUATION
# =============================================================================
# Initialize the model
# Update the model initialization
model = GCN(gene_dim=gene_feat_dim, 
            cell_dim=cell_feat_dim, 
            drug_dim=drug_feat_dim,
            out_dim=256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    
    # Pack the raw features and maps to pass to the model
    raw_features_tuple = (gene_features_raw, cell_features_raw, drug_features_raw)
    node_maps_tuple = (gene2idx, cell2idx, drug2idx)
    
    # Get final node embeddings after GCN layers
    z = model(raw_features_tuple, node_maps_tuple, train_edge_index)
    
    # Decode for the link prediction task
    logits = model.decode(z, train_pos_cd_edges, train_neg_cd_edges)
    
    # Create labels
    pos_labels = torch.ones(train_pos_cd_edges.size(1))
    neg_labels = torch.zeros(train_neg_cd_edges.size(1))
    labels = torch.cat([pos_labels, neg_labels]).to(device)
    
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate():
    model.eval()
    
    raw_features_tuple = (gene_features_raw, cell_features_raw, drug_features_raw)
    node_maps_tuple = (gene2idx, cell2idx, drug2idx)

    # Note: We still use the training graph structure for message passing during evaluation
    z = model(raw_features_tuple, node_maps_tuple, train_edge_index)
    
    logits = model.decode(z, test_pos_cd_edges, test_neg_cd_edges)
    
    pos_labels = torch.ones(test_pos_cd_edges.size(1))
    neg_labels = torch.zeros(test_neg_cd_edges.size(1))
    labels = torch.cat([pos_labels, neg_labels]).cpu()
    
    predictions = torch.sigmoid(logits).cpu()
    
    acc = accuracy_score(labels, (predictions > 0.5).int())
    auc = roc_auc_score(labels, predictions)
    
    return acc, auc, labels, predictions

# --- Training Loop ---
best_val_auc = 0
patience_counter = 0
patience = 20

print("--- Starting Training ---")
for epoch in range(1, 501):
    loss = train()
    
    if epoch % 10 == 0:
        val_acc, val_auc, _, _ = evaluate()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Optional: Save best model checkpoint
            # torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in validation AUC.")
            break

# --- Final Evaluation and ROC Curve ---
print("\n--- Final Evaluation on Test Set ---")
final_acc, final_auc, test_labels, test_predictions = evaluate()
print(f'Final Test Accuracy: {final_acc:.4f}')
print(f'Final Test AUC: {final_auc:.4f}')

# Plot ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {final_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()