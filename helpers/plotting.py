import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import MaxNLocator

def plot_metrics(history, model_name, save_path):
    all_records = []
    for run_idx, run_data in enumerate(history):
        num_epochs = len(run_data['training_loss'])
        for epoch in range(num_epochs):
            record = {
                'run': run_idx,
                'epoch': epoch + 1,
                'training_loss': run_data['training_loss'][epoch],
                'validation_loss': run_data['val_loss'][epoch],
                'training_accuracy': run_data['training_accuracy'][epoch],
                'validation_accuracy': run_data['val_accuracy'][epoch],
            }
            all_records.append(record)
        
    df = pd.DataFrame(all_records)
    max_epochs = df['epoch'].max()
    epochs = np.arange(1, max_epochs + 1)

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Training and Validation Metrics of {model_name} over {len(history)} folds", fontsize=16)

    def format_ax(axes, title, y_label):
        axes.set_title(title, fontsize=14)
        axes.set_ylabel(y_label)
        axes.set_xlabel('Epoch')
        axes.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    sns.lineplot(
        data = df,
        x = 'epoch',
        y = 'training_loss',
        errorbar='sd',
        ax = ax[0, 0],
        color = 'blue'
    )
    format_ax(ax[0, 0], 'Training Loss', 'Loss')

    sns.lineplot(
        data = df,
        x = 'epoch',
        y = 'validation_loss',
        errorbar='sd',
        ax = ax[0, 1],
        color = 'orange'
    )
    format_ax(ax[0, 1], 'Validation Loss', 'Loss')

    sns.lineplot(
        data = df,
        x = 'epoch',
        y = 'training_accuracy',
        errorbar='sd',
        ax = ax[1, 0],
        color = 'green'
    )
    format_ax(ax[1, 0], 'Training Accuracy', 'Accuracy (%)')

    sns.lineplot(
        data = df,
        x = 'epoch',
        y = 'validation_accuracy',
        errorbar='sd',
        ax = ax[1, 1],
        color = 'red'
    )
    format_ax(ax[1, 1], 'Validation Accuracy', 'Accuracy (%)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()