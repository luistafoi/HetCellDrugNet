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


def get_drug_cell_data_loader(dl, input_data, drug_type, cell_type, batch_size, device):
    """
    Prepares a DataLoader for the drug-cell link prediction task for TRAINING.
    """
    drug_type_id = input_data.node_name2type.get(drug_type)
    cell_type_id = input_data.node_name2type.get(cell_type)

    if drug_type_id is None:
        raise ValueError(f"Node type name '{drug_type}' not found in dataset's type mapping.")
    if cell_type_id is None:
        raise ValueError(f"Node type name '{cell_type}' not found in dataset's type mapping.")

    drug_cell_r_id = -1
    u_type_id, v_type_id = None, None
    for r_id, (s_type, d_type) in dl.links['meta'].items():
        if (s_type == drug_type_id and d_type == cell_type_id):
            drug_cell_r_id = r_id
            u_type_id, v_type_id = drug_type_id, cell_type_id
            break
        if (s_type == cell_type_id and d_type == drug_type_id):
            drug_cell_r_id = r_id
            u_type_id, v_type_id = cell_type_id, drug_type_id
            break
    
    if drug_cell_r_id == -1:
        raise ValueError(f"Could not find a relation between '{drug_type}' and '{cell_type}' in the training data.")

    u_type_name = input_data.node_type2name[u_type_id]
    v_type_name = input_data.node_type2name[v_type_id]

    pos_links = dl.train_pos[drug_cell_r_id]
    neg_links = dl.train_neg[drug_cell_r_id]

    u_nodes = pos_links[0] + neg_links[0]
    v_nodes = pos_links[1] + neg_links[1]
    labels = [1.0] * len(pos_links[0]) + [0.0] * len(neg_links[0])

    # --- FIX: Move all tensors to the specified device ---
    dataset = TensorDataset(
        torch.LongTensor(u_nodes).to(device),
        torch.LongTensor(v_nodes).to(device),
        torch.FloatTensor(labels).to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True), u_type_name, v_type_name


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
    """Prepares a DataLoader for the TEST set."""
    drug_type_id = input_data.node_name2type.get(drug_type)
    cell_type_id = input_data.node_name2type.get(cell_type)
    if drug_type_id is None or cell_type_id is None: raise ValueError("Test: Drug or cell type not found.")

    drug_cell_r_id = -1
    u_type_id, v_type_id = None, None
    for r_id, (s_type, d_type) in dl.links_test['meta'].items():
        if (s_type == drug_type_id and d_type == cell_type_id):
            drug_cell_r_id = r_id
            u_type_id, v_type_id = drug_type_id, cell_type_id
            break
        if (s_type == cell_type_id and d_type == drug_type_id):
            drug_cell_r_id = r_id
            u_type_id, v_type_id = cell_type_id, drug_type_id
            break
    
    if drug_cell_r_id == -1: raise ValueError("Test relation not found.")
    u_type_name = input_data.node_type2name[u_type_id]
    v_type_name = input_data.node_type2name[v_type_id]
    
    pos_links_matrix = dl.links_test['data'][drug_cell_r_id]
    pos_rows, pos_cols = pos_links_matrix.nonzero()
    neg_links = dl.test_neg.get(drug_cell_r_id, [[], []])
    u_nodes = list(pos_rows) + neg_links[0]
    v_nodes = list(pos_cols) + neg_links[1]
    labels = [1.0] * len(pos_rows) + [0.0] * len(neg_links[0])
    
    # --- FIX: Move all tensors to the specified device ---
    dataset = TensorDataset(
        torch.LongTensor(u_nodes).to(device),
        torch.LongTensor(v_nodes).to(device),
        torch.FloatTensor(labels).to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), u_type_name, v_type_name


def calculate_mrr(preds, labels):
    """Calculates the Mean Reciprocal Rank (MRR)."""
    if len(labels) == 0 or sum(labels) == 0:
        return 0.0
    sorted_results = sorted(zip(preds, labels), key=lambda x: x[0], reverse=True)
    rank = 0
    for i, (pred, label) in enumerate(sorted_results):
        if label == 1:
            rank = i + 1
            break
    return 1.0 / rank if rank > 0 else 0.0


def evaluate_model(model, dataloader, u_type_eval, v_type_eval, drug_type_name, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        # The u_nodes and v_nodes tensors are already on the GPU now
        for u_nodes, v_nodes, labels in dataloader:
            preds = model.link_prediction_forward(u_nodes, v_nodes)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) # labels is also on the GPU, so .cpu() is good practice here
    
    if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
        return 0.0, 0.0, 0.0
        
    roc_auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(np.array(all_labels), np.array(all_preds) > 0.5)
    mrr = calculate_mrr(all_preds, all_labels)
    
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

    # --- 1. Load All Data ---
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', args.data)
    dl = data_loader(data_path)
    input_data = data_generator.input_data(args, dl, temp_dir)
    
    het_neigh_train_f = os.path.join(temp_dir, 'het_neigh_train.txt')
    if not os.path.exists(het_neigh_train_f):
        print(f"Generating training neighbors file: {het_neigh_train_f}")
        input_data.gen_het_w_walk_restart(het_neigh_train_f)
    else:
        print(f"Loading existing training neighbors file: {het_neigh_train_f}")
        # We need to load the file if it exists but is not yet in memory
        if not any(any(n_list for n_list in v) for v in input_data.neigh_list_train.values()):
            input_data.load_het_neigh_train(het_neigh_train_f)

    het_random_walk_f = os.path.join(temp_dir, 'het_random_walk.txt')
    if not os.path.exists(het_random_walk_f):
        print(f"Generating random walk file: {het_random_walk_f}")
        input_data.gen_het_w_walk(het_random_walk_f)
    
    input_data.gen_embeds_w_neigh()
    
    drug_type_name = 'drug'
    cell_type_name = 'cell'
    
    # --- 2. Prepare All DataLoaders (Train, Validation, Test) ---
    lp_dataloader, u_type_lp, v_type_lp = get_drug_cell_data_loader(dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device)
    valid_dataloader, u_type_valid, v_type_valid = get_validation_data_loader(dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device)
    test_dataloader, u_type_test, v_type_test = get_test_data_loader(dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device)

    # --- 3. Initialize Model and Optimizer ---
    model = tools.HetAgg(args, dl=dl, input_data=input_data, device=device).to(device)
    model.init_weights()
    model.setup_link_prediction(drug_type_name=drug_type_name, cell_type_name=cell_type_name)
    loss_wrapper = tools.MultiTaskLossWrapper(n_tasks=2).to(device)
    optimizer = optim.Adam(itertools.chain(model.parameters(), loss_wrapper.parameters()), lr=args.lr, weight_decay=0)
    
    # --- 4. Variables for Tracking Best Model ---
    best_valid_auc = 0.0
    best_epoch = 0
    best_model_state = None

    print('\n--- Starting End-to-End Multi-Task Training ---')
    for epoch in range(args.train_iter_n):
        model.train()
        print(f'\nINFO: Epoch {epoch+1} / {args.train_iter_n}')
        
        lp_iter = iter(lp_dataloader)
        triple_list = input_data.sample_het_walk_triple()
        
        if not triple_list or not any(triple_list.values()):
            print("Warning: No triples were sampled for this epoch. Skipping.")
            continue
            
        batch_n = min(len(lp_dataloader), int(len(list(triple_list.values())[0]) / args.mini_batch_s))
        if batch_n == 0:
            print("Warning: Not enough data to create a single batch. Skipping epoch.")
            continue
        print(f'INFO: Processing {batch_n} batches for this epoch.')

        # --- 1. Initialize accumulators for all losses ---
        total_epoch_loss = 0.0
        total_epoch_rw_loss = 0.0
        total_epoch_lp_loss = 0.0

        for k in range(batch_n):
            optimizer.zero_grad()

            # --- Task A: Random Walk Loss ---
            c_out_rw, p_out_rw, n_out_rw = [], [], []
            for triple_pair in triple_list.keys():
                triple_list_batch = triple_list[triple_pair][k * args.mini_batch_s: (k + 1) * args.mini_batch_s]
                if not triple_list_batch: continue
                c, p, n = model(triple_list_batch, triple_pair)
                c_out_rw.append(c); p_out_rw.append(p); n_out_rw.append(n)
            
            if not c_out_rw: continue
            loss_rw = tools.cross_entropy_loss(torch.cat(c_out_rw), torch.cat(p_out_rw), torch.cat(n_out_rw))
            
            # --- Task B: Link Prediction Loss ---
            u_nodes, v_nodes, labels = next(lp_iter)
            loss_lp = model.link_prediction_loss(u_nodes, v_nodes, labels)
            
            # --- Combine, Backpropagate, and Accumulate ---
            total_loss = loss_wrapper([loss_rw, loss_lp])
            total_loss.backward()
            optimizer.step()

            # --- 2. Accumulate all three loss values ---
            total_epoch_loss += total_loss.item()
            total_epoch_rw_loss += loss_rw.item()
            total_epoch_lp_loss += loss_lp.item()

            # Batch-level logging is still useful
            weights = torch.exp(-loss_wrapper.log_vars).detach().cpu().numpy()
            print(f"  Batch {k}/{batch_n} | Total Loss: {total_loss.item():.4f} | "
                    f"RW Loss: {loss_rw.item():.4f} (w={weights[0]:.4f}) | "
                    f"LP Loss: {loss_lp.item():.4f} (w={weights[1]:.4f})")
        
        # --- 3. Calculate and print all three average losses ---
        avg_total = total_epoch_loss / batch_n
        avg_rw = total_epoch_rw_loss / batch_n
        avg_lp = total_epoch_lp_loss / batch_n
        print(f"--- Epoch {epoch+1} Avg Loss: {avg_total:.4f} | Avg RW Loss: {avg_rw:.4f} | Avg LP Loss: {avg_lp:.4f} ---")
        
        # --- PERIODIC VALIDATION ---
        if (epoch + 1) % 5 == 0:
            print("--- Running Validation ---")
            valid_auc, valid_f1, valid_mrr = evaluate_model(model, valid_dataloader, u_type_valid, v_type_valid, drug_type_name, device)
            print(f"Validation Results | ROC-AUC: {valid_auc:.4f} | F1: {valid_f1:.4f} | MRR: {valid_mrr:.4f}")

            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"*** New best validation AUC found. Saving model state from epoch {best_epoch}. ***")

    print("\n--- Training Finished ---")
    
    # --- 6. FINAL EVALUATION ON TEST SET ---
    print(f"\n--- Loading best model from epoch {best_epoch} (AUC: {best_valid_auc:.4f}) and running final evaluation on Test Set ---")
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    else:
        print("Warning: No best model was saved. Evaluating the final model state.")
        final_model_path = os.path.join(temp_dir, "final_model_state.pt")
        torch.save(model.state_dict(), final_model_path)

    test_auc, test_f1, test_mrr = evaluate_model(model, test_dataloader, u_type_test, v_type_test, drug_type_name, device)
    print("\n--- Final Test Set Evaluation Results ---")
    print(f"ROC-AUC: {test_auc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {test_mrr:.4f}")
    print("-----------------------------------------")