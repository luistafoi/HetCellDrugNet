import os
import torch
import torch.optim as optim
import optuna
import random
import numpy as np

# Import your existing code from other files
import data_generator
import tools
from utils.data_loader import data_loader
from main import get_drug_cell_data_loader, get_validation_data_loader, evaluate_model

# --- Global Settings ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'ours_filtered_moleculestm_vae' # The dataset to use
N_TRIALS = 60  # Total number of hyperparameter combinations to test
N_EPOCHS_PER_TRIAL = 30 # Number of epochs to run for each trial

def objective(trial, dl, input_data):
    """
    This function is called by Optuna for each trial.
    It trains the model with a set of hyperparameters and returns a score.
    """
    # --- 1. Define the Hyperparameter Search Space ---
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'embed_d': trial.suggest_categorical('embed_d', [128, 256, 512]),
        'n_layers': trial.suggest_int('n_layers', 1, 3),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    # Use a dummy args object to pass parameters
    class Args:
        def __init__(self, p):
            self.embed_d = p['embed_d']
            self.n_layers = p['n_layers']
            self.mini_batch_s = 256

    args = Args(params)
    
    # --- 2. Initialize and Train the Model ---
    valid_dataloader, u_type_valid, _ = get_validation_data_loader(dl, input_data, 'drug', 'cell', args.mini_batch_s, DEVICE)

    model = tools.HetAgg(args, dl=dl, input_data=input_data, device=DEVICE).to(DEVICE)
    model.init_weights()
    model.setup_link_prediction(drug_type_name='drug', cell_type_name='cell')
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    best_valid_auc = 0.0

    print(f"--- Starting Trial {trial.number} with params: {trial.params} ---")
    for epoch in range(N_EPOCHS_PER_TRIAL):
        model.train()
        
        # We will ONLY train on the link prediction task to make the search faster
        lp_dataloader, _, _ = get_drug_cell_data_loader(dl, input_data, 'drug', 'cell', args.mini_batch_s, DEVICE, max_samples=100000)
        for i, (u_nodes, v_nodes, labels) in enumerate(lp_dataloader):
            optimizer.zero_grad()
            loss_lp = model.link_prediction_loss(u_nodes, v_nodes, labels)
            loss_lp.backward()
            optimizer.step()
        
        # Periodic Validation
        if (epoch + 1) % 5 == 0:
            valid_auc, _, _ = evaluate_model(model, valid_dataloader, u_type_valid, 'cell', 'drug', DEVICE)
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc

            # Print the intermediate results for monitoring
            print(f"  > Trial {trial.number}, Epoch {epoch + 1}: Val AUC = {valid_auc:.4f} (Best for this trial: {best_valid_auc:.4f})")
            
            # Optuna Pruning: Stop unpromising trials early
            trial.report(best_valid_auc, epoch)
            if trial.should_prune():
                print(f"  > Trial {trial.number} pruned.")
                raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} Finished ---")
    return best_valid_auc


if __name__ == '__main__':
    # --- Load Data ONCE Before Starting the Study ---
    print("--- Loading and preparing data once... ---")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', DATASET_NAME)
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{DATASET_NAME}-temp')
    
    class TempArgs: # Dummy args needed for data_generator
        data = DATASET_NAME
        batch_s = 20000
        window = 5
    
    dl = data_loader(data_path)
    input_data = data_generator.input_data(TempArgs(), dl, temp_dir)

    # Ensure random walk files exist so they don't need to be generated
    het_neigh_train_f = os.path.join(temp_dir, 'het_neigh_train.txt')
    if not os.path.exists(het_neigh_train_f):
        print("ERROR: `het_neigh_train.txt` not found. Please run main.py once to generate temp files.")
        exit()
    else:
        input_data.load_het_neigh_train(het_neigh_train_f)
    print("--- Data loading complete. Starting optimization. ---")

    # --- Run the Optimization Study ---
    print(f"\nStarting hyperparameter optimization with Optuna for {N_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, dl, input_data), n_trials=N_TRIALS)

    print("\n" + "="*80)
    print("Optimization finished.")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Best Validation AUC): {trial.value:.4f}")
    print("  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*80)