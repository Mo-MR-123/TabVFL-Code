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
    failure_probabilities_x_axis = ["TabVFL-0.2", "TabVFL-0.35", "TabVFL-0.5"]
    

    for idx, dataset_name in enumerate(dataset_names):
        print(f"Current dataset name: {dataset_name}")
        curr_dataset_baseline_value = client_failure_tabvfl_cache_df.loc[idx, "TabVFL"]
        max_improvement = -1
        avg_improvement_percentages = 0
        min_val_zeros = np.inf
        min_val_cache = np.inf
        for fp in failure_probabilities_x_axis:
            curr_p_val_zeros = client_failure_tabvfl_zeros_df.loc[idx, fp]
            curr_p_val_cache = client_failure_tabvfl_cache_df.loc[idx, fp]
            min_val_zeros = min(min_val_zeros, curr_p_val_zeros)
            min_val_cache = min(min_val_cache, curr_p_val_cache)
            improvement = ((curr_p_val_cache - curr_p_val_zeros) / curr_p_val_zeros) * 100
            max_improvement = max(max_improvement, round(improvement, 2))
            avg_improvement_percentages += improvement 
        avg_improvement_percentages /= len(failure_probabilities_x_axis)
        print(f"average_improvement_cache_method: {round(avg_improvement_percentages, 2)}")
        print(f"max_improvement_cache_method: {max_improvement}")
        max_perf_decrease_zeros = ((min_val_zeros - curr_dataset_baseline_value) / curr_dataset_baseline_value) * 100
        max_perf_decrease_cache = ((min_val_cache - curr_dataset_baseline_value) / curr_dataset_baseline_value) * 100
        print(f"max_performance_decrease_zeros: {round(max_perf_decrease_zeros, 2)}")
        print(f"max_performance_decrease_cache: {round(max_perf_decrease_cache, 2)}")

