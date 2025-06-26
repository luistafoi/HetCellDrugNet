import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import itertools
from sklearn.metrics import roc_auc_score, f1_score

torch.set_num_threads(2)
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
import pickle
import os
import data_generator
import tools
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils.data_loader import data_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Use device: {device}')

def get_drug_cell_data_loader(dl, input_data, drug_type, cell_type, batch_size, device):
    """
    Prepares a DataLoader for the drug-cell link prediction task.
    It finds the drug-cell relation, gathers positive and negative examples
    from the training set, and creates a DataLoader to serve them in batches.
    """
    drug_type_id = input_data.node_name2type.get(drug_type)
    cell_type_id = input_data.node_name2type.get(cell_type)

    if drug_type_id is None:
        raise ValueError(f"Node type name '{drug_type}' not found in dataset's type mapping.")
    if cell_type_id is None:
        raise ValueError(f"Node type name '{cell_type}' not found in dataset's type mapping.")

    # Find the relation ID for (drug, cell) or (cell, drug) links
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
        raise ValueError(f"Could not find a relation between '{drug_type}' (id:{drug_type_id}) and '{cell_type}' (id:{cell_type_id}) in the training data.")

    u_type_name = input_data.node_type2name[u_type_id]
    v_type_name = input_data.node_type2name[v_type_id]
    print(f"INFO: Found drug-cell relation for LP: ID={drug_cell_r_id}, Types=({u_type_name}, {v_type_name})")

    # Get positive and negative links from the data_loader's training sets
    pos_links = dl.train_pos[drug_cell_r_id]
    neg_links = dl.train_neg[drug_cell_r_id]

    # Combine positive and negative samples
    u_nodes = pos_links[0] + neg_links[0]
    v_nodes = pos_links[1] + neg_links[1]
    labels = [1.0] * len(pos_links[0]) + [0.0] * len(neg_links[0])

    # Create a TensorDataset and a DataLoader
    dataset = TensorDataset(
        torch.LongTensor(u_nodes).to(device),
        torch.LongTensor(v_nodes).to(device),
        torch.FloatTensor(labels).to(device)
    )
    # Drop last batch if it's smaller than the rest, to maintain batch size consistency
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True), u_type_name, v_type_name

def get_test_data_loader(dl, input_data, drug_type, cell_type, batch_size, device):
    """
    Prepares a DataLoader for the test set.
    """
    drug_type_id = input_data.node_name2type.get(drug_type)
    cell_type_id = input_data.node_name2type.get(cell_type)

    if drug_type_id is None:
        raise ValueError(f"Node type name '{drug_type}' not found in dataset's type mapping.")
    if cell_type_id is None:
        raise ValueError(f"Node type name '{cell_type}' not found in dataset's type mapping.")

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
    
    if drug_cell_r_id == -1:
        raise ValueError(f"Could not find a relation between '{drug_type}' and '{cell_type}' in the test data.")

    pos_links_matrix = dl.links_test['data'][drug_cell_r_id]
    pos_rows, pos_cols = pos_links_matrix.nonzero()
    
    neg_links = dl.test_neg[drug_cell_r_id]

    u_nodes = list(pos_rows) + neg_links[0]
    v_nodes = list(pos_cols) + neg_links[1]
    labels = [1.0] * len(pos_rows) + [0.0] * len(neg_links[0])

    dataset = TensorDataset(
        torch.LongTensor(u_nodes).to(device),
        torch.LongTensor(v_nodes).to(device),
        torch.FloatTensor(labels).to(device)
    )
    u_type_name = input_data.node_type2name[u_type_id]
    v_type_name = input_data.node_type2name[v_type_id]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), u_type_name, v_type_name


def calculate_mrr(preds, labels):
    """
    Calculates the Mean Reciprocal Rank (MRR).
    Assumes preds and labels are sorted by prediction score in descending order.
    """
    # Combine predictions and labels and sort by prediction score
    sorted_results = sorted(zip(preds, labels), key=lambda x: x[0], reverse=True)
    
    # Find the rank of the first true positive
    rank = 0
    for i, (pred, label) in enumerate(sorted_results):
        if label == 1:
            rank = i + 1
            break
    
    return 1.0 / rank if rank > 0 else 0.0

if __name__ == '__main__':
    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    
    data_name = args.data
    temp_dir = os.path.join(sys.path[0], f'{data_name}-temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # fix random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # --- 1. Load Data ---
    # The data_loader class is now the single source of truth for data.
    # It handles finding, unzipping, and splitting data into train/valid/test sets.
    data_path = f'data/{data_name}'
    dl = data_loader(data_path)

    # --- 2. Prepare Data for Both Tasks ---
    # Task A (Self-supervised): Random walk data generation
    input_data = data_generator.input_data(args, dl, temp_dir)
    # The following lines generate the necessary walk files for the self-supervised task
    het_neigh_train_f = os.path.join(temp_dir, 'het_neigh_train.txt')
    if not os.path.exists(het_neigh_train_f):
        input_data.gen_het_w_walk_restart(het_neigh_train_f)
    het_random_walk_f = os.path.join(temp_dir, 'het_random_walk.txt')
    if not os.path.exists(het_random_walk_f):
        input_data.gen_het_w_walk(het_random_walk_f)
    input_data.gen_embeds_w_neigh()
    
    # Task B (Supervised): Link prediction data
    # NOTE: Change 'drug' and 'cell_line' if your node type names are different in your dataset
    drug_type_name = 'drug'
    cell_type_name = 'cell'
    lp_dataloader, u_type_lp, v_type_lp = get_drug_cell_data_loader(dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device)

    # --- 3. Initialize Model, Loss Wrapper, and Optimizer ---
    feature_list = {k: v.to(device) for k, v in input_data.feature_list.items()}
    model = tools.HetAgg(args, feature_list, neigh_list_train=input_data.neigh_list_train,
                         dl=dl, input_data=input_data, device=device).to(device)
    model.init_weights()
    # IMPORTANT: Setup the model for the link prediction task
    model.setup_link_prediction(drug_type_name=drug_type_name, cell_type_name=cell_type_name)
    
    loss_wrapper = tools.MultiTaskLossWrapper(n_tasks=2).to(device)
    
    # The optimizer needs to manage the parameters of BOTH the model and the loss wrapper
    optimizer = optim.Adam(
        itertools.chain(model.parameters(), loss_wrapper.parameters()), 
        lr=args.lr, 
        weight_decay=0
    )

    print('--- Starting End-to-End Multi-Task Training ---')
    model.train()
    
    # --- 4. Multi-Task Training Loop ---
    for iter_i in range(args.train_iter_n):
        print(f'INFO: Iteration {iter_i+1} / {args.train_iter_n}')
        
        # Prepare iterators for both tasks for this epoch
        lp_iter = iter(lp_dataloader)
        triple_list = input_data.sample_het_walk_triple()
        
        # Determine number of batches based on the smaller of the two tasks
        batch_n = min(len(lp_dataloader), int(len(list(triple_list.values())[0]) / args.mini_batch_s))
        print(f'INFO: Processing {batch_n} batches for this iteration.')

        for k in range(batch_n):
            optimizer.zero_grad()

            # --- Task A: Random Walk Loss ---
            # This part remains similar to the original, getting a batch of triples
            c_out_rw, p_out_rw, n_out_rw = [], [], []
            for triple_pair_index, triple_pair in enumerate(triple_list.keys()):
                triple_list_batch = triple_list[triple_pair][k * args.mini_batch_s: (k + 1) * args.mini_batch_s]
                if not triple_list_batch: continue
                c, p, n = model(triple_list_batch, triple_pair)
                c_out_rw.append(c)
                p_out_rw.append(p)
                n_out_rw.append(n)
            
            if not c_out_rw: continue # Skip if no valid RW batches
            loss_rw = tools.cross_entropy_loss(torch.cat(c_out_rw), torch.cat(p_out_rw), torch.cat(n_out_rw), args.embed_d)

            # --- Task B: Link Prediction Loss ---
            # Get the next batch of drug-cell links
            u_nodes, v_nodes, labels = next(lp_iter)
            
            # Ensure the node order matches what the model expects (drug, cell)
            if u_type_lp == drug_type_name:
                loss_lp = model.link_prediction_loss(u_nodes, v_nodes, labels)
            else: # The relation was (cell, drug), so we swap them
                loss_lp = model.link_prediction_loss(v_nodes, u_nodes, labels)

            # --- Combine Losses and Backpropagate ---
            total_loss = loss_wrapper([loss_rw, loss_lp])
            
            total_loss.backward()
            optimizer.step()

            if k % 50 == 0:
                # Get the learned weights (1/sigma^2) for logging
                weights = torch.exp(-loss_wrapper.log_vars).detach().cpu().numpy()
                print(f"  Batch {k}/{batch_n} | Total Loss: {total_loss.item():.4f} | "
                      f"RW Loss: {loss_rw.item():.4f} (w={weights[0]:.2f}) | "
                      f"LP Loss: {loss_lp.item():.4f} (w={weights[1]:.2f})")

        if iter_i % args.save_model_freq == 0 and iter_i > 0:
            model_path = os.path.join(temp_dir, f"model_iter_{iter_i}.pt")
            torch.save(model.state_dict(), model_path)
            print(f'INFO: Saved model to {model_path}')

    print("--- Training Finished ---")
    final_model_path = os.path.join(temp_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final trained model to {final_model_path}")

    # --- 5. Evaluation ---
    print("\n--- Starting Evaluation on Test Set ---")
    
    # Load the final model
    model.load_state_dict(torch.load(final_model_path))
    model.eval()

    # Prepare test data loader
    test_dataloader, u_type_test, v_type_test = get_test_data_loader(dl, input_data, drug_type_name, cell_type_name, args.mini_batch_s, device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for u_nodes, v_nodes, labels in test_dataloader:
            if u_type_test == drug_type_name:
                preds = model.link_prediction_forward(u_nodes, v_nodes)
            else:
                preds = model.link_prediction_forward(v_nodes, u_nodes)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    roc_auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    mrr = calculate_mrr(all_preds, all_labels)

    print("--- Evaluation Results ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print("--------------------------")
