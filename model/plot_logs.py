import re
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def parse_log_file(log_file_path):
    """
    Parses the training log file to extract metrics and returns them as a list of dicts.
    """
    print(f"Parsing log file: {log_file_path}")
    
    # Regex patterns to find the lines with our metrics
    epoch_regex = re.compile(r"Epoch (\d+)")
    lp_loss_regex = re.compile(r"LP Loss: (\d+\.\d+)|Avg Link Prediction Loss for Epoch: (\d+\.\d+)")
    
    # --- THIS IS THE FIX ---
    # The definition for rw_regex was missing. It is now added.
    rw_loss_regex = re.compile(r"RW Loss: (\d+\.\d+)|Avg Random Walk Loss for Epoch: (\d+\.\d+)")
    # --- END OF FIX ---

    val_regex = re.compile(r"Validation Results.*?ROC-AUC: (\d+\.\d+)")

    records = []
    current_epoch_data = {}
    current_epoch = 0

    with open(log_file_path, 'r') as f:
        for line in f:
            epoch_match = epoch_regex.search(line)
            if epoch_match:
                if current_epoch_data and 'epoch' in current_epoch_data:
                    records.append(current_epoch_data)
                current_epoch = int(epoch_match.group(1))
                current_epoch_data = {'epoch': current_epoch}

            lp_match = lp_loss_regex.search(line)
            if lp_match:
                current_epoch_data['lp_loss'] = float(lp_match.group(1) or lp_match.group(2))

            rw_match = rw_loss_regex.search(line)
            if rw_match:
                current_epoch_data['rw_loss'] = float(rw_match.group(1) or rw_match.group(2))
            
            val_match = val_regex.search(line)
            if val_match:
                current_epoch_data['val_auc'] = float(val_match.group(1))
    
    if current_epoch_data:
        records.append(current_epoch_data)
        
    return records

def save_and_plot_metrics(records, output_prefix='training_run'):
    """
    Saves the parsed metrics to a CSV file and generates plots.
    """
    if not records:
        print("No data to plot or save.")
        return

    df = pd.DataFrame(records)
    
    csv_filename = f"{output_prefix}_metrics.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSuccessfully saved metrics to: {os.path.abspath(csv_filename)}")

    sns.set_theme(style="whitegrid")

    # Plot 1: Training Losses
    plt.figure(figsize=(12, 7))
    # Melt the DataFrame to plot both losses on the same axes with different colors
    loss_df = df.melt(id_vars=['epoch'], value_vars=['lp_loss', 'rw_loss'], var_name='loss_type', value_name='loss_value')
    sns.lineplot(data=loss_df, x='epoch', y='loss_value', hue='loss_type', palette=['#4c72b0', '#dd8452'])
    plt.title('Training Losses vs. Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(title='Loss Type')
    loss_plot_filename = f"{output_prefix}_losses_plot.pdf"
    plt.savefig(loss_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Successfully saved loss plot to: {os.path.abspath(loss_plot_filename)}")
    plt.close()

    # Plot 2: Validation AUC
    auc_df = df.dropna(subset=['val_auc'])
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=auc_df, x='epoch', y='val_auc', marker='o', label='Validation ROC-AUC', color='g')
    plt.title('Validation ROC-AUC vs. Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.legend()
    auc_plot_filename = f"{output_prefix}_auc_plot.pdf"
    plt.savefig(auc_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Successfully saved AUC plot to: {os.path.abspath(auc_plot_filename)}")
    plt.close()


if __name__ == '__main__':
    try:
        import pandas
        import matplotlib
        import seaborn
    except ImportError:
        print("\nERROR: Missing required libraries for plotting.")
        print("Please install them by running the following command:")
        print("pip install pandas matplotlib seaborn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Parse and plot GNN training logs.")
    parser.add_argument('log_file', type=str, help="Path to the training log file.")
    args = parser.parse_args()
    
    parsed_data = parse_log_file(args.log_file)
    save_and_plot_metrics(parsed_data)
