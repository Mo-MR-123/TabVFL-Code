import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.style.use('seaborn-whitegrid')
plt.rcParams['axes.edgecolor'] = '0.8'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

SCRIPT_DIR = Path(__file__).resolve().parent

latent_eval_mean_results_filename = "latent_eval_means_per_design.xlsx"

tabnet_vfl_column_names = ["TabNet VFL-Accuracy", "TabNet VFL-F1-score", "TabNet VFL-ROC-AUC"]
local_tabnets_column_names = ["Local TabNets-Accuracy", "Local TabNets-F1-score", "Local TabNets-ROC-AUC"]
central_tabnet_column_names = ["Central TabNet-Accuracy", "Central TabNet-F1-score", "Central TabNet-ROC-AUC"]
tabnet_vfl_le_column_names = ["TabNet VFL-LE-Accuracy", "TabNet VFL-LE-F1-score", "TabNet VFL-LE-ROC-AUC"]

if __name__ == "__main__":
    # load latent eval mean results
    latent_eval_mean_results_df = pd.read_excel(f"{SCRIPT_DIR}/{latent_eval_mean_results_filename}")
    dataset_names = latent_eval_mean_results_df["Dataset"].to_numpy()
    designs_names = [
        "CT",
        "LT",
        "TabVFL-LE",
        "TabVFL",
    ]
    patterns_designs = [
        '\\',
        '/',
        'x',
        '|'
    ]

    # intrusion dataset accuracies of designs
    eval_accs_per_dataset = [
        latent_eval_mean_results_df[central_tabnet_column_names[0]].to_numpy(),
        latent_eval_mean_results_df[local_tabnets_column_names[0]].to_numpy(),
        latent_eval_mean_results_df[tabnet_vfl_le_column_names[0]].to_numpy(),
        latent_eval_mean_results_df[tabnet_vfl_column_names[0]].to_numpy(),
    ]
    print(f"{eval_accs_per_dataset=}")
    eval_f1_scores_per_dataset = [
        latent_eval_mean_results_df[central_tabnet_column_names[1]].to_numpy(),
        latent_eval_mean_results_df[local_tabnets_column_names[1]].to_numpy(),
        latent_eval_mean_results_df[tabnet_vfl_le_column_names[1]].to_numpy(),
        latent_eval_mean_results_df[tabnet_vfl_column_names[1]].to_numpy(),
    ]
    eval_roc_aucs_per_dataset = [
        latent_eval_mean_results_df[central_tabnet_column_names[2]].to_numpy(),
        latent_eval_mean_results_df[local_tabnets_column_names[2]].to_numpy(),
        latent_eval_mean_results_df[tabnet_vfl_le_column_names[2]].to_numpy(),
        latent_eval_mean_results_df[tabnet_vfl_column_names[2]].to_numpy(),
    ]

    fig, axes = plt.subplots(3, len(dataset_names), figsize=(15, 8), sharey='row')

    design_name_color = [
        "crimson", 
        "lightsalmon", 
        "#1A76FF",
        "g", 
    ]
    
    bar_width = 0.03
    bar_gap = 0.01  # Adjust the gap between bars
    ticks = np.arange(len(designs_names))
    
    for idx, dataset_name in enumerate(dataset_names):
        x_pos = np.arange(len(designs_names)) * (bar_width + bar_gap)  # Calculate x-positions once per dataset

        eval_res_per_design_curr_dataset = []
        for design_idx, design_name in enumerate(designs_names):
            eval_res_per_design_curr_dataset.append(eval_f1_scores_per_dataset[design_idx][idx])
            
            # Add labels to each point in the scatter plot
            acc = eval_accs_per_dataset[design_idx][idx]
            axes[0][idx].bar(
                x_pos[design_idx],
                acc,
                bar_width,
                color=design_name_color[design_idx],
                label=design_name,  # Use the design name as the label
                hatch=patterns_designs[design_idx],
                lw=2
            )
            axes[0][idx].set_xticks([])
            axes[0][idx].set_ylim([50,100])

            f1_score = eval_f1_scores_per_dataset[design_idx][idx]
            axes[1][idx].bar(
                x_pos[design_idx],
                f1_score,
                bar_width,
                color=design_name_color[design_idx],
                label=design_name,  # Use the design name as the label
                hatch=patterns_designs[design_idx],
                lw=2
            )
            axes[1][idx].set_xticks([])
            axes[1][idx].set_ylim([0.5,1])

            roc_auc = eval_roc_aucs_per_dataset[design_idx][idx]
            axes[2][idx].bar(
                x_pos[design_idx],
                roc_auc,
                bar_width,
                color=design_name_color[design_idx],
                label=design_name,  # Use the design name as the label
                hatch=patterns_designs[design_idx],
                lw=2
            )
            axes[2][idx].set_xticks([])
            axes[2][idx].set_ylim([0.5,1])

        print(f"{eval_res_per_design_curr_dataset=}")

        if idx == 0:
            axes[0][0].set_ylabel('Accuracy')
            axes[1][0].set_ylabel('F1-score')
            axes[2][0].set_ylabel('ROC-AUC')

        axes[0][idx].set_title(f'{dataset_name}')

        if idx == len(dataset_names)-1:
            handles, labels = axes[0][idx].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right', frameon=True, fontsize='large', bbox_to_anchor=(1.12, 0))
        
    plt.tight_layout()
    fig.savefig(str(SCRIPT_DIR / f"latent_eval_plot.png"), format='png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()

    