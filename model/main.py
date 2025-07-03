import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import itertools
from sklearn.metrics import roc_auc_score, f1_score
import os
import sys
import numpy as np
import random
import copy
from args import read_args

import data_generator
import tools
from utils.data_loader import data_loader

torch.set_num_threads(2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_drug_cell_data_loader(dl, input_data, drug_type, cell_type, batch_size, device, max_samples=None):
    """
    Prepares a DataLoader for the drug-cell link prediction task for TRAINING.
    Includes optional sub-sampling for faster epochs.
    """
    drug_type_id = input_data.node_name2type.get(drug_type)
    cell_type_id = input_data.node_name2type.get(cell_type)

    if drug_type_id is None: raise ValueError(f"Node type name '{drug_type}' not found.")
    if cell_type_id is None: raise ValueError(f"Node type name '{cell_type}' not found.")

    drug_cell_r_id = -1
    u_type_id, v_type_id = None, None
    for r_id, (s_type, d_type) in dl.links['meta'].items():
        if (s_type == drug_type_id and d_type == cell_type_id):
            drug_cell_r_id, u_type_id, v_type_id = r_id, drug_type_id, cell_type_id
            break
        if (s_type == cell_type_id and d_type == drug_type_id):
            drug_cell_r_id, u_type_id, v_type_id = r_id, cell_type_id, drug_type_id
            break
    if drug_cell_r_id == -1: raise ValueError(f"Could not find relation for '{drug_type}'-'{cell_type}'.")

    u_type_name = input_data.node_type2name[u_type_id]
    v_type_name = input_data.node_type2name[v_type_id]

    pos_links = dl.train_pos[drug_cell_r_id]
    neg_links = dl.train_neg[drug_cell_r_id]

    u_nodes_pos_full, v_nodes_pos_full = pos_links[0], pos_links[1]
    u_nodes_neg_full, v_nodes_neg_full = neg_links[0], neg_links[1]

    # --- NEW: Sub-sampling Logic ---
    if max_samples and max_samples < (len(u_nodes_pos_full) + len(u_nodes_neg_full)):
        num_pos = max_samples // 2
        num_neg = max_samples - num_pos
        
        pos_indices = np.random.choice(len(u_nodes_pos_full), num_pos, replace=False)
        neg_indices = np.random.choice(len(u_nodes_neg_full), num_neg, replace=False)

        u_nodes = [u_nodes_pos_full[i] for i in pos_indices] + [u_nodes_neg_full[i] for i in neg_indices]
        v_nodes = [v_nodes_pos_full[i] for i in pos_indices] + [v_nodes_neg_full[i] for i in neg_indices]
        labels = [1.0] * num_pos + [0.0] * num_neg
    else:
        u_nodes = u_nodes_pos_full + u_nodes_neg_full
        v_nodes = v_nodes_pos_full + v_nodes_neg_full
        labels = [1.0] * len(u_nodes_pos_full) + [0.0] * len(u_nodes_neg_full)

    dataset = TensorDataset(
        torch.LongTensor(u_nodes).to(device),
        torch.LongTensor(v_nodes).to(device),
        torch.FloatTensor(labels).to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0), u_type_name, v_type_name


def get_validation_data_loader(dl, input_data, drug_type, cell_type, batch_size, device):
    """Prepares a DataLoader for the VALIDATION set."""
    drug_type_id = input_data.node_name2type.get(drug_type)
    cell_type_id = input_data.node_name2type.get(cell_type)
    if drug_type_id is None or cell_type_id is None: raise ValueError("Validation: Drug or cell type not found.")
    
    drug_cell_r_id = -1
    u_type_id, v_type_id = None, None
    for r_id, (s_type, d_type) in dl.links_init['meta'].items():
        if (s_type == drug_type_id and d_type == cell_type_id):
            drug_cell_r_id = r_id
            u_type_id, v_type_id = drug_type_id, cell_type_id
            break
        if (s_type == cell_type_id and d_type == drug_type_id):
            drug_cell_r_id = r_id
            u_type_id, v_type_id = cell_type_id, drug_type_id
            break
    
    if drug_cell_r_id == -1: raise ValueError("Validation relation not found.")
    u_type_name = input_data.node_type2name[u_type_id]
    v_type_name = input_data.node_type2name[v_type_id]

    pos_links = dl.valid_pos.get(drug_cell_r_id, [[], []])
    neg_links = dl.valid_neg.get(drug_cell_r_id, [[], []])
    u_nodes = pos_links[0] + neg_links[0]
    v_nodes = pos_links[1] + neg_links[1]
    labels = [1.0] * len(pos_links[0]) + [0.0] * len(neg_links[0])

    # --- FIX: Move all tensors to the specified device ---
    dataset = TensorDataset(
        torch.LongTensor(u_nodes).to(device),
        torch.LongTensor(v_nodes).to(device),
        torch.FloatTensor(labels).to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), u_type_name, v_type_name

def get_test_data_loader(dl, input_data, drug_type, cell_type, batch_size, device):
    """Prepares a DataLoader for the TEST set with robust data validation."""
    drug_type_id = input_data.node_name2type.get(drug_type)
    cell_type_id = input_data.node_name2type.get(cell_type)
    if drug_type_id is None or cell_type_id is None: raise ValueError("Test: Drug or cell type not found.")

    # Find the relation ID for the drug-cell interaction
    drug_cell_r_id = -1
    is_reversed = False # Flag to check if the relation in the file is (cell, drug)
    for r_id, (s_type, d_type) in dl.links_test['meta'].items():
        if (s_type == drug_type_id and d_type == cell_type_id):
            drug_cell_r_id = r_id
            is_reversed = False
            break
        if (s_type == cell_type_id and d_type == drug_type_id):
            drug_cell_r_id = r_id
            is_reversed = True
            print("INFO: Test relation is in (cell, drug) order. Loader will swap them correctly.")
            break
    
    if drug_cell_r_id == -1: raise ValueError("Test relation not found.")
    
    drug_nodes, cell_nodes, labels = [], [], []
    
    # --- Process Positive Links ---
    pos_links_matrix = dl.links_test['data'][drug_cell_r_id]
    pos_rows, pos_cols = pos_links_matrix.nonzero()
    print(f"INFO: Processing {len(pos_rows)} positive test links.")
    
    # The model's link_prediction_forward always expects (drug, cell) order.
    # We must ensure the lists are populated correctly regardless of the file's order.
    if is_reversed:
        # The file has (cell, drug), so rows are cells and cols are drugs.
        cell_nodes.extend(list(pos_rows))
        drug_nodes.extend(list(pos_cols))
    else:
        # The file has (drug, cell), so rows are drugs and cols are cells.
        drug_nodes.extend(list(pos_rows))
        cell_nodes.extend(list(pos_cols))
    labels.extend([1.0] * len(pos_rows))

    # --- Process Negative Links ---
    neg_links = dl.test_neg.get(drug_cell_r_id, [[], []])
    neg_u, neg_v = neg_links[0], neg_links[1] # u corresponds to rows, v to cols
    print(f"INFO: Processing {len(neg_u)} negative test links.")

    if is_reversed:
        # Negative samples were generated for (cell, drug), so u are cells, v are drugs.
        cell_nodes.extend(neg_u)
        drug_nodes.extend(neg_v)
    else:
        # Negative samples were generated for (drug, cell), so u are drugs, v are cells.
        drug_nodes.extend(neg_u)
        cell_nodes.extend(neg_v)
    labels.extend([0.0] * len(neg_u))

    print(f"INFO: Total test samples prepared: {len(labels)}")
    
    # The model expects (drug, cell), so we pass drug_nodes as the first tensor
    # and cell_nodes as the second.
    dataset = TensorDataset(
        torch.LongTensor(drug_nodes).to(device),
        torch.LongTensor(cell_nodes).to(device),
        torch.FloatTensor(labels).to(device)
    )
    
    # Return the type names in the order the model expects.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), drug_type, cell_type

def evaluate_model(model, dataloader, u_type_eval, v_type_eval, drug_type_name, device):
    """Evaluates the model on a given dataloader and returns performance metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    # A dictionary to group predictions by their head node for MRR
    # It will look like: { head_node_id: [ (pred_score, label), ... ], ... }
    preds_by_head = {}

    with torch.no_grad():
        for u_nodes_batch, v_nodes_batch, labels_batch in dataloader:
            u_nodes = u_nodes_batch.to(device)
            v_nodes = v_nodes_batch.to(device)
            labels = labels_batch.to(device)

            preds = model.link_prediction_forward(u_nodes, v_nodes)
            
            # Store predictions and labels for overall AUC and F1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Group predictions and labels by head node for MRR calculation
            head_nodes = u_nodes.cpu().numpy()
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()

            for i in range(len(head_nodes)):
                head_id = head_nodes[i]
                if head_id not in preds_by_head:
                    preds_by_head[head_id] = []
                preds_by_head[head_id].append( (batch_preds[i], batch_labels[i]) )

    # --- Calculate overall metrics ---
    if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
        return 0.0, 0.0, 0.0
        
    roc_auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(np.array(all_labels), np.array(all_preds) > 0.5)
    
    # --- Calculate true Mean Reciprocal Rank (MRR) ---
    reciprocal_ranks = []
    for head_id, predictions in preds_by_head.items():
        # Only consider queries that have at least one true positive link
        if any(label == 1 for score, label in predictions):
            # Sort predictions for this head node by score, descending
            predictions.sort(key=lambda x: x[0], reverse=True)
            
            # Find the rank of the first true positive
            for rank, (score, label) in enumerate(predictions):
                if label == 1:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break # Move to the next head node
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    return roc_auc, f1, mrr

if __name__ == '__main__':
    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    
    print(f'Using device: {device}')
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.data}-temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', args.data)
    dl = data_loader(data_path)
    input_data = data_generator.input_data(args, dl, temp_dir)
    
    het_neigh_train_f = os.path.join(temp_dir, 'het_neigh_train.txt')
    if not os.path.exists(het_neigh_train_f):
        input_data.gen_het_w_walk_restart(het_neigh_train_f)
    else:
        if not any(any(n_list for n_list in v) for v in input_data.neigh_list_train.values()):
            input_data.load_het_neigh_train(het_neigh_train_f)

    het_random_walk_f = os.path.join(temp_dir, 'het_random_walk.txt')
    if not os.path.exists(het_random_walk_f):
        input_data.gen_het_w_walk(het_random_walk_f)
    
    input_data.gen_embeds_w_neigh()
    
    drug_type_name = 'drug'
    cell_type_name = 'cell'
    
    valid_dataloader, u_type_valid, v_type_valid = get_validation_data_loader(dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device)
    test_dataloader, u_type_test, v_type_test = get_test_data_loader(dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device)

    model = tools.HetAgg(args, dl=dl, input_data=input_data, device=device).to(device)
    model.init_weights()
    model.setup_link_prediction(drug_type_name=drug_type_name, cell_type_name=cell_type_name)
    
    # Optimizer now only takes the model's parameters
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    
    best_valid_auc, best_epoch, best_model_state = 0.0, 0, None
    patience, patience_counter = 50, 0

    print('\n--- Starting End-to-End Multi-Task Training ---')
    for epoch in range(args.train_iter_n):
        model.train()
        print(f'\nINFO: Epoch {epoch+1} / {args.train_iter_n}')

        # --- Phase 1: Link Prediction Training on a SUBSET of data ---
        print("--- Phase 1: Training on Link Prediction ---")
        # Create a new, smaller, random subset of data for each epoch.
        # This makes each epoch much faster.
        lp_dataloader, u_type_lp, v_type_lp = get_drug_cell_data_loader(
            dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device, max_samples=200000
        )
        
        total_lp_loss = 0.0
        for i, (u_nodes, v_nodes, labels) in enumerate(lp_dataloader):
            optimizer.zero_grad()
            loss_lp = model.link_prediction_loss(u_nodes, v_nodes, labels)
            loss_lp.backward()
            optimizer.step()
            total_lp_loss += loss_lp.item()
        
        if len(lp_dataloader) > 0:
            print(f"  Avg Link Prediction Loss for Epoch: {total_lp_loss / len(lp_dataloader):.4f}")

        # --- Phase 2: Self-Supervised Training ---
        print("--- Phase 2: Training on Random Walks ---")
        triple_list = input_data.sample_het_walk_triple()
        if not triple_list or not any(triple_list.values()):
            print("  Warning: No triples sampled for this phase. Skipping.")
        else:
            total_rw_loss = 0.0
            num_rw_batches = 0
            for triple_pair, triples in triple_list.items():
                if not triples: continue
                num_batches_for_pair = (len(triples) + args.mini_batch_s - 1) // args.mini_batch_s
                for k in range(num_batches_for_pair):
                    optimizer.zero_grad()
                    triple_list_batch = triples[k * args.mini_batch_s: (k + 1) * args.mini_batch_s]
                    c_embeds, p_embeds, n_embeds = model(triple_list_batch, triple_pair)
                    loss_rw = tools.cross_entropy_loss(c_embeds, p_embeds, n_embeds)
                    loss_rw.backward()
                    optimizer.step()
                    total_rw_loss += loss_rw.item()
                    num_rw_batches += 1
            
            if num_rw_batches > 0:
                print(f"  Avg Random Walk Loss for Epoch: {total_rw_loss / num_rw_batches:.4f}")

        # --- Periodic Validation ---
        if (epoch + 1) % 5 == 0:
            print("--- Running Validation ---")
            valid_auc, valid_f1, valid_mrr = evaluate_model(model, valid_dataloader, u_type_valid, v_type_valid, drug_type_name, device)
            print(f"Validation Results | ROC-AUC: {valid_auc:.4f} | F1: {valid_f1:.4f} | MRR: {valid_mrr:.4f}")

            if valid_auc > best_valid_auc:
                best_valid_auc, best_epoch = valid_auc, epoch + 1
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"*** New best validation AUC found. Saving model state from epoch {best_epoch}. ***")
            else:
                patience_counter += 1
                print(f"Validation AUC did not improve. Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Stopping early as validation has not improved for {patience*5} epochs.")
                break

    print("\n--- Training Finished ---")
    
    print(f"\n--- Loading best model from epoch {best_epoch} (AUC: {best_valid_auc:.4f}) and running final evaluation on Test Set ---")
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    else:
        print("Warning: No best model was saved. Evaluating the final model state.")

    test_auc, test_f1, test_mrr = evaluate_model(model, test_dataloader, u_type_test, v_type_test, drug_type_name, device)
    print("\n--- Final Test Set Evaluation Results ---")
    print(f"ROC-AUC: {test_auc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {test_mrr:.4f}")
    print("-----------------------------------------")