import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['axes.edgecolor'] = '0.8'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

SCRIPT_DIR = Path(__file__).resolve().parent

client_failures_tabvfl_cache_filename = "client_failure_f1_score_tabvfl_cache_plus_pretrain.xlsx"
client_failures_tabvfl_zeros_filename = "client_failure_f1_score_tabvfl_zeros_plus_pretrain.xlsx"

if __name__ == "__main__":
    # load latent eval mean results
    client_failure_tabvfl_cache_df = pd.read_excel(f"{SCRIPT_DIR}/{client_failures_tabvfl_cache_filename}")
    client_failure_tabvfl_zeros_df = pd.read_excel(f"{SCRIPT_DIR}/{client_failures_tabvfl_zeros_filename}")
    
    dataset_names = client_failure_tabvfl_cache_df["Dataset"].to_numpy()
    failure_probabilities_x_axis = ["0.2", "0.35", "0.5"]
    x_axis_point_without_failure_sim = ["0.2"]
    methods_df_list = [
        client_failure_tabvfl_zeros_df,
        client_failure_tabvfl_cache_df,
    ]

    fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(15, 4))
    
    ticks = np.arange(len(failure_probabilities_x_axis))
    
    labels = [
        "TabVFL w/ zeros method",
        "TabVFL w/ cached method",
    ]
    colors = [
        "r",
        "g",
    ]
    markers_linestyle = [
        ("s", "-"),
        ("*", "-"),
    ]

    for idx, dataset_name in enumerate(dataset_names):
        min_val = np.inf
        max_val = -np.inf
        for method_idx, method_df in enumerate(methods_df_list):
            axes[idx].plot(
                failure_probabilities_x_axis,
                [round(number, 3) for number in method_df.iloc[idx, 2:]],
                color=colors[method_idx],
                linestyle=markers_linestyle[method_idx][1],
                marker=markers_linestyle[method_idx][0],
                label=labels[method_idx],  # Use the design name as the label
                lw=2
            )

        axes[idx].axhline(
            y=methods_df_list[0].iloc[idx, 1], # both method dfs have the save starting values as no client failure were simulated
            color="b",
            linestyle='--',
            label="TabVFL w/o failures",
            markersize=6
        )

        if idx == 0:
            axes[idx].set_ylabel('F1-score')
        
        axes[idx].set_xlabel('p', fontstyle="italic")
        axes[idx].set_title(f'{dataset_name}')

        if idx == len(dataset_names)-1:
            handles, labels = axes[idx].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower right", frameon=True, bbox_to_anchor=(1.17, 0.4))
    
    plt.tight_layout()
    fig.savefig(str(SCRIPT_DIR / f"client_failure_methods_plus_pretrain_tabvfl_plots.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()