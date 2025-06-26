import string
import re
import numpy as np
import os
import random
from itertools import islice # Make sure islice is imported if used, or remove if not. Assuming it was intended earlier.
import argparse
import pickle
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import json
import sys
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle # Added for sub-sampling

sys.path.append('../../') # To find scripts folder

# Assuming data_loader is in scripts/data_loader.py relative to benchmark/
from scripts.data_loader import data_loader


parser = argparse.ArgumentParser(description='application data process')
parser.add_argument('--embed_d', type=int, default=128,
                    help='embedding dimension')
parser.add_argument('--data', type=str, default='amazon', # This will be 'ours' from your command
                    help='select dataset')
parser.add_argument('--random_seed', type=int, default=42, help='random seed for shuffling') # Added for reproducibility
args = parser.parse_args()
print(args)

data_name = args.data
# temp_dir should be relative to this script's location if node_embedding files are saved there by main.py
# e.g., LP/benchmark/methods/HetGNN/{data_name}-temp/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(current_script_dir, f'{data_name}-temp')

# data_path points to where the HGB dataset (node.dat, link.dat, info.dat) is located
# e.g., LP/data/{data_name}/
data_path = os.path.join(current_script_dir, f'../../data/{data_name}')


dl_pickle_f = os.path.join(data_path, 'dl_pickle') # Path to the pickled data_loader object
if os.path.exists(dl_pickle_f):
    dl = pickle.load(open(dl_pickle_f, 'rb'))
    print(f'Info: load {data_name} from {dl_pickle_f}')
else:
    dl = data_loader(data_path) # data_path is 'LP/data/ours/'
    pickle.dump(dl, open(dl_pickle_f, 'wb'))
    print(f'Info: load {data_name} from original data and generate {dl_pickle_f}')

node_n_total = dl.nodes['total']
node_shift = dl.nodes['shift']
node_counts_per_type = dl.nodes['count']


info_dat_path = os.path.join(data_path, "info.dat")
try:
    with open(info_dat_path, 'r') as f_info:
        data_info = json.load(f_info)
except FileNotFoundError:
    print(f"FATAL ERROR: info.dat not found at {info_dat_path}. Exiting.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"FATAL ERROR: Could not decode JSON from info.dat at {info_dat_path}. Exiting.")
    sys.exit(1)

f_node_type_info_dict = data_info.get('node.dat')
if not isinstance(f_node_type_info_dict, dict):
    print(f"FATAL ERROR: 'node.dat' key missing or not a dictionary in {info_dat_path}. Exiting.")
    sys.exit(1)

node_type2name, node_name2type = dict(), dict()
for type_id_str, type_details_list in f_node_type_info_dict.items():
    try:
        type_id_int = int(type_id_str)
        if isinstance(type_details_list, list) and len(type_details_list) > 0:
            name_str = str(type_details_list[0])
            node_type2name[type_id_int] = name_str
            node_name2type[name_str] = type_id_int
        else:
            print(f"Warning: Invalid format for type ID {type_id_str} in info.dat: {type_details_list}. Skipping this type.")
    except ValueError:
        print(f"Warning: Could not convert type ID {type_id_str} to int. Skipping this type.")
        continue

if not node_name2type:
    print(f"FATAL ERROR: node_name2type map is empty after parsing info.dat. Check info.dat structure and content.")
    sys.exit(1)

print(f"Successfully populated node_name2type: {node_name2type}")
print(f"Successfully populated node_type2name: {node_type2name}")


# Only evaluate cell-drug and drug-cell relations (IDs 0 and 1)
cell_drug_relation_ids = [0, 1]

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import tools

# --- Helper functions ---
def load_model_and_embeddings(model_path, embedding_path=None, temp_dir=None, dl=None, node_name2type=None):
    # Model loading is optional if only using embeddings for dot product scoring
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location='cpu')
    else:
        model = None
    # Find latest node_embedding*.txt file
    embedding_files = [f for f in os.listdir(temp_dir) if f.startswith('node_embedding') and f.endswith('.txt')]
    if not embedding_files:
        raise FileNotFoundError(f"No node_embedding*.txt files found in {temp_dir}")
    def extract_epoch(fname):
        m = re.match(r'node_embedding(\d+)\.txt', fname)
        return int(m.group(1)) if m else -1
    latest_file = max(embedding_files, key=extract_epoch)
    embedding_path = os.path.join(temp_dir, latest_file)
    print(f"Loading embeddings from {embedding_path}")
    # Parse embedding file: lines: node_name <vec>
    # Example: cell0 0.1 0.2 ...
    # Map node_name (e.g., cell0) to integer node ID
    # Build embedding matrix of shape (num_nodes, embed_dim)
    # Get total number of nodes and embedding dim
    num_nodes = dl.nodes['total']
    embed_dim = None
    embeddings = None
    with open(embedding_path, 'r') as f:
        lines = f.readlines()
        # Determine embedding dim from first line
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            embed_dim = len(parts) - 1
            break
        if embed_dim is None:
            raise ValueError("Could not determine embedding dimension from file.")
        embeddings = np.zeros((num_nodes, embed_dim), dtype=np.float32)
        for line in lines:
            parts = line.strip().split()
            if len(parts) != embed_dim + 1:
                continue
            node_name = parts[0]  # e.g., cell0, drug1
            # Parse node type and index
            m = re.match(r'([a-zA-Z]+)(\d+)', node_name)
            if not m:
                continue
            ntype, nidx = m.group(1), int(m.group(2))
            ntype_id = node_name2type.get(ntype)
            if ntype_id is None:
                continue
            node_id = dl.nodes['shift'][ntype_id] + nidx
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[node_id] = vec
    return model, embeddings

def score_links(embeddings, pairs):
    # Dot product scoring; replace with model-specific scoring if needed
    return np.array([np.dot(embeddings[u], embeddings[v]) for u, v in pairs])

def load_test_pairs(link_dat_test_path, rel_ids):
    pos_pairs = []
    with open(link_dat_test_path, 'r') as f:
        for line in f:
            s, t, r, w = line.strip().split('\t')
            if int(r) in rel_ids:
                pos_pairs.append((int(s), int(t)))
    return pos_pairs

def generate_negative_samples(node_type2name, node_name2type, dl, rel_ids, pos_pairs, num_samples):
    # Only sample negatives of the correct type (cell-drug or drug-cell)
    pos_set = set(pos_pairs)
    neg_pairs = set()
    rng = np.random.default_rng(42)
    # Get node id ranges for each type
    cell_type = node_name2type.get('cell', 0)
    drug_type = node_name2type.get('drug', 1)
    cell_start = dl.nodes['shift'][cell_type]
    cell_end = cell_start + dl.nodes['count'][cell_type]
    drug_start = dl.nodes['shift'][drug_type]
    drug_end = drug_start + dl.nodes['count'][drug_type]
    while len(neg_pairs) < num_samples:
        # Randomly pick direction (cell->drug or drug->cell)
        rel = rng.choice(rel_ids)
        if rel == 0:
            u = rng.integers(cell_start, cell_end)
            v = rng.integers(drug_start, drug_end)
        else:
            u = rng.integers(drug_start, drug_end)
            v = rng.integers(cell_start, cell_end)
        if (u, v) not in pos_set:
            neg_pairs.add((u, v))
    return list(neg_pairs)

if __name__ == "__main__":
    # Paths to model and embeddings (update as needed)
    model_path = os.path.join(temp_dir, "model_final.pth")
    link_dat_test_path = os.path.join(data_path, "link.dat.test")
    rel_ids = cell_drug_relation_ids

    # Load model and embeddings
    # Instead of loading only state_dict, instantiate model and load weights
    feature_list = dl.nodes.get('attr', None)
    # Rebuild feature_list as in training
    input_data = None
    try:
        import data_generator
        input_data = data_generator.input_data(args, dl)
        feature_list = input_data.feature_list
        for node_type in feature_list.keys():
            feature_list[node_type] = feature_list[node_type].to('cpu')
    except Exception as e:
        print(f"Warning: Could not rebuild feature_list/input_data: {e}")
    model = tools.HetAgg(args, feature_list, neigh_list_train=None, train_id_list=None, dl=dl, input_data=input_data, device='cpu')
    cell_type_id = node_name2type.get('cell', 0)
    drug_type_id = node_name2type.get('drug', 1)
    model.build_merge_and_lp_layers(cell_type_id, drug_type_id)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    # Prepare initial embeddings
    initial_embeds = []
    for node_type in sorted(feature_list.keys()):
        initial_embeds.append(feature_list[node_type])
    initial_embeds = torch.cat(initial_embeds, dim=0)
    initial_embeds = initial_embeds.to('cpu')

    # Load positive test pairs
    pos_pairs = load_test_pairs(link_dat_test_path, rel_ids)
    num_pos = len(pos_pairs)
    print(f"Loaded {num_pos} positive test pairs for relations 0 and 1.")

    # Generate negative samples (matched by type)
    neg_pairs = generate_negative_samples(node_type2name, node_name2type, dl, rel_ids, pos_pairs, num_pos)
    print(f"Generated {len(neg_pairs)} negative samples.")

    # Score positive and negative pairs using the model's link prediction head
    def score_with_model(model, pairs, u_type, v_type, initial_embeds, batch_size=512):
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            u_idx = torch.tensor([x[0] for x in batch], dtype=torch.long)
            v_idx = torch.tensor([x[1] for x in batch], dtype=torch.long)
            with torch.no_grad():
                s = model.score_link_batch(u_idx, v_idx, u_type, v_type, initial_embeds).cpu().numpy()
            scores.append(s)
        return np.concatenate(scores)

    # Score cell->drug and drug->cell separately
    pos_scores = []
    neg_scores = []
    for rel_id, (u_type, v_type) in zip([0, 1], [(cell_type_id, drug_type_id), (drug_type_id, cell_type_id)]):
        pos_pairs_rel = [p for p in pos_pairs if (rel_id == 0 and p[0] >= dl.nodes['shift'][cell_type_id] and p[1] >= dl.nodes['shift'][drug_type_id]) or (rel_id == 1 and p[0] >= dl.nodes['shift'][drug_type_id] and p[1] >= dl.nodes['shift'][cell_type_id])]
        neg_pairs_rel = [p for p in neg_pairs if (rel_id == 0 and p[0] >= dl.nodes['shift'][cell_type_id] and p[1] >= dl.nodes['shift'][drug_type_id]) or (rel_id == 1 and p[0] >= dl.nodes['shift'][drug_type_id] and p[1] >= dl.nodes['shift'][cell_type_id])]
        pos_scores.append(score_with_model(model, pos_pairs_rel, u_type, v_type, initial_embeds))
        neg_scores.append(score_with_model(model, neg_pairs_rel, u_type, v_type, initial_embeds))
    pos_scores = np.concatenate(pos_scores)
    neg_scores = np.concatenate(neg_scores)

    y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
    y_scores = np.concatenate([pos_scores, neg_scores])

    # Compute metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")

    # Optionally, print ROC and PR curves (first 5 points)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    print(f"ROC curve (first 5 points): {list(zip(fpr, tpr))[:5]} ...")
    print(f"PR curve (first 5 points): {list(zip(precision, recall))[:5]} ...")