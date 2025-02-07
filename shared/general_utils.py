import numpy as np
import pandas as pd
from pathlib import Path
import time
import sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import log_loss, r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.utils.multiclass import type_of_target
from typing import Any, Dict, List, Tuple, Union, TypeVar
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import io
import subprocess
import tempfile

from xgboost import XGBModel

from shared.tabnet_logger import TabNetLogger

def round_to_n_decimals(arr: Union[torch.Tensor, np.ndarray], decimal_places: int = 3) -> Union[torch.Tensor, np.ndarray]:
    """
    This function rounds the float to the specified number of decimal places
    """
    if isinstance(arr, torch.Tensor):
        return torch.round(arr * 10**decimal_places) / 10**decimal_places
    elif isinstance(arr, np.ndarray):
        return np.round(arr, decimal_places)
    else:
        raise TypeError("Input must be a PyTorch tensor or NumPy array")

def plot_tsne_of_latent(X_latent_data: np.ndarray, y_labels: np.ndarray, experiment_name: str, script_dir: Path):
    with plt.style.context("ggplot"):
        # Check if the array is 1D
        if X_latent_data.ndim == 1:
            # Convert the array to a 2D array with a single column
            X_latent_data = X_latent_data.reshape(-1, 1)
        
        # Apply t-SNE to transform the data into a 2D space
        tsne = TSNE(n_components=2, random_state=42, n_jobs=4)
        X_tsne = tsne.fit_transform(X_latent_data)
        
        # Create a new figure with a specific size
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create an array of transparency values
        unique_labels, counts_vals = np.unique(y_labels, return_counts=True)
        if len(unique_labels) == 2:
            if counts_vals[0] < counts_vals[1]:
                alphas = np.where(y_labels == counts_vals[0], 0.2, 0.8)   
            else:
                alphas = np.where(y_labels == counts_vals[1], 0.2, 0.8)
        elif len(unique_labels) > 2 and np.issubdtype(unique_labels.dtype, np.integer) and len(unique_labels) <= 20:
            initial_alpha = 0.95
            alphas = np.zeros(len(unique_labels))
            sorted_indexes_counts_desc = reversed(sorted(range(len(counts_vals)), key=lambda i: counts_vals[i]))
            for idx in sorted_indexes_counts_desc:
                alphas[idx] = initial_alpha
                initial_alpha -= 0.1
                if initial_alpha < 0.05:
                    initial_alpha = 0.5
        else:
            alphas = [0.5] * len(unique_labels)

        # Plot the transformed data, color-coded by the target variable per labels
        for label_idx, label in enumerate(unique_labels):
            if label_idx >= 20:
                break
            idx = np.where(y_labels == label)[0]
            ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label, alpha=alphas[label_idx])

        # Add labels to the plot
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        ax.set_title(f't-SNE plot of the latent space from trained TabNet model, experiment: {experiment_name}')

        # legend
        ax.legend(loc='lower right', title='Target variable')

        fig.savefig(str(script_dir / f"tsne_plot_{experiment_name}.pdf"), format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.5)
        
        # Show the plot
        plt.show()

        plt.close(fig)

def preprocessing_label_data(
    label_data_to_preprocess: pd.DataFrame,
    logger=None,
) -> pd.DataFrame:
    # Make sure only one column is present, which is the label column
    if len(label_data_to_preprocess.columns.tolist()) > 1:
        raise ValueError("Only one column is expected in the label data for preprocessing")
    
    # For now, only label encode the labels
    _label_encoder_labels = LabelEncoder()
    labels_preprocessed = _label_encoder_labels.fit_transform(label_data_to_preprocess.to_numpy().ravel())
    return labels_preprocessed

def infer_optimizer(optimizer_name: str):
    """
    This function returns the optimizer class based on the optimizer name
    """
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

def log_if_available(logger: TabNetLogger, message: str, log_level: str = "info"):
    """
    This function logs the message if the logger is available, otherwise it prints the message
    """
    if logger is not None:
        if log_level.lower() == "info":
            logger.info(message)
        elif log_level.lower() == "warning":
            logger.warning(message)
        elif log_level.lower() == "error":
            logger.error(message)
        elif log_level.lower() == "critical":
            logger.critical(message)
        else:
            raise ValueError(f"Log level {log_level} not supported")
    else:
        print(message)

T = TypeVar('T')
def train_valid_test_split(
    X_data_to_split: T, 
    y_data_to_split: T,
    test_ratio: float,
    valid_ratio: float,
    random_state: int = None,
    shuffle: bool = False,
    stratify: T = None
) -> Tuple[T, T, T, T, T, T]:
    """
    This function splits the data into train, validation, and test sets
    """
    if test_ratio > 0.2:
        raise ValueError("test_ratio must be less than or equal to 0.25. Otherwise, the training set will be too small.") 
    if valid_ratio > 0.2:
        raise ValueError("valid_ratio must be less than or equal to 0.2. Otherwise, the training set will be too small.")
    
    train_data, temp_data, y_train, y_temp = train_test_split(
        X_data_to_split, 
        y_data_to_split,
        test_size=(test_ratio + valid_ratio), 
        shuffle=shuffle, 
        random_state=random_state, 
        stratify=stratify
    )

    # Calculate the size of the validation set based on the valid_ratio
    valid_size = valid_ratio / (test_ratio + valid_ratio)

    if stratify is None:
        test_data, valid_data, y_test, y_val = train_test_split(
            temp_data, 
            y_temp,
            test_size=valid_size, 
            shuffle=shuffle, 
            random_state=random_state, 
            stratify=None
        )
    else:
        test_data, valid_data, y_test, y_val = train_test_split(
            temp_data, 
            y_temp,
            test_size=valid_size, 
            shuffle=shuffle, 
            random_state=random_state, 
            stratify=y_temp
        )
    return train_data, valid_data, test_data, y_train, y_val, y_test
    

def open_df_temp(df: pd.DataFrame, exp_name: str):
    # write DataFrame to string buffer using tab separator
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, sep='\t')

    # write contents of string buffer to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix=exp_name+"_", delete=False) as f:
        f.write(csv_buffer.getvalue())
        tmp_file_path = f.name
    
    print(f"Saving tmp file of evaluation results to {tmp_file_path=}")

    # open temporary file in default program that opens .csv files. Works only on Linux.
    subprocess.run(['xdg-open', tmp_file_path], check=True)


def evaluate_predictor(
    predictor: Union[BaseEstimator, XGBModel],
    eval_results_path: Path,
    task_type: str = None,
    logger: TabNetLogger = None,
    X_test: np.ndarray = None, 
    y_test: np.ndarray = None,
):
    if hasattr(predictor, 'estimator'):
        predictor_name = type(predictor.estimator).__name__
    else:
        predictor_name = type(predictor).__name__
    log_if_available(None, f"------------------{predictor_name} EVALUATION RESULTS START----------------------")

    if hasattr(predictor, 'predict_proba') or hasattr(predictor, 'estimator'):
        y_proba = predictor.predict_proba(X_test)
    else:
        y_proba = None

    y_pred = predictor.predict(X_test)

    unique, counts = np.unique(y_pred, return_counts=True)
    unique_true, counts_true = np.unique(y_test, return_counts=True)
    log_if_available(logger, f"Unique values in y_pred: {unique}, counts: {counts} ")   
    log_if_available(logger, f"Unique values in y_true: {unique_true}, counts: {counts_true} ")   

    # storing the evaluation results in a dictionary
    results_dict = {}

    if task_type.startswith('binary'):  
        accuracy = accuracy_score(
            y_true=y_test,
            y_pred=y_pred
        )
        log_if_available(logger, f"Accuracy: {accuracy * 100:.6f}%")

        # Calculate precision, recall, and F1 score using scikit-learn
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_test, 
            y_pred=y_pred, 
            average='weighted'
        )

        log_if_available(logger, f"F1 score: {np.around(f1, decimals=6)}")

        # Calculate roc_auc_score using scikit-learn
        test_roc_auc = roc_auc_score(
            y_true=y_test, 
            y_score=y_proba[:, 1],
            average='weighted', 
            multi_class='ovr'
        )
        log_if_available(logger, f"ROC AUC SCORE: {np.around(test_roc_auc, decimals=6)}")

        results_dict['accuracy'] = np.around(accuracy * 100, decimals=4)
        results_dict['f1_score'] = np.around(f1, decimals=4)
        results_dict['roc_auc'] = np.around(test_roc_auc, decimals=4)

    elif task_type.startswith('multiclass') or task_type == 'multilabel-indicator':
        accuracy = accuracy_score(
            y_true=y_test,
            y_pred=y_pred
        )
        log_if_available(logger, f"Accuracy: {accuracy * 100:.6f}%")

        # Calculate precision, recall, and F1 score using scikit-learn
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_test, 
            y_pred=y_pred, 
            average='weighted'
        )
        log_if_available(logger, f"F1 score: {np.around(f1, decimals=6)}")

        log_if_available(logger, f"{y_proba=}")

        # Calculate roc_auc_score using scikit-learn
        test_roc_auc = roc_auc_score(
            y_true=y_test, 
            y_score=y_proba,
            average='weighted',
            multi_class='ovr'
        )
        log_if_available(logger, f"ROC AUC SCORE: {test_roc_auc}")

        results_dict['accuracy'] = np.around(accuracy * 100, decimals=4)
        results_dict['f1_score'] = np.around(f1, decimals=4)
        results_dict['roc_auc'] = np.around(test_roc_auc, decimals=4)

    elif task_type.startswith('continuous'):
        """ 
            - Mean Absolute Error (MAE): This metric measures the average absolute difference between the
            predicted and actual values of the target variable. MAE takes values between 0 and infinity,
            with lower values indicating a better fit of the model to the data. 
            MAE is less sensitive to outliers than MSE, but it does not penalize large errors as heavily.

            - Mean Squared Error (MSE): This metric measures the average squared difference between the
            predicted and actual values of the target variable. MSE takes values between 0 and infinity,
            with lower values indicating a better fit of the model to the data. 
            MSE penalizes large errors more heavily than MAE, which can make it more sensitive to outliers. However, MSE can be more difficult to interpret since it is in squared units of the target variable. 
        """
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        log_if_available(logger, f"R^2 score: {r2:.6f}")
        log_if_available(logger, f"MAE score: {mae:.6f}")
        log_if_available(logger, f"MSE score: {mse:.6f}")

        results_dict['r2'] = np.around(r2, decimals=4)
        results_dict['mae'] = np.around(mae, decimals=4)
        results_dict['mse'] = np.around(mse, decimals=4)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    log_if_available(None, f"------------------{predictor_name} EVALUATION RESULTS END----------------------")

    save_test_results(
        y_true=y_test,
        y_probs=y_proba,
        y_pred=y_pred,
        name_npz_file=eval_results_path,
        predictor_name=predictor_name
    )
    
    return results_dict

def evaluate_multiple_predictors(
    predictors: List[Union[BaseEstimator, XGBModel]],
    hyperparams: Dict[str, Any],
    eval_results_path: Path,
    experiment_name: str,
    task_type: str,
    logger: TabNetLogger = None,
    X_test: np.ndarray = None, 
    y_test: np.ndarray = None,  # should already be numpy array
    store_eval_results_tmp_folder: bool = True,
):
    log_if_available(logger, f"TYPE OF TARGET (TASK TYPE) IN EVALUATION: {task_type}")

    if task_type.startswith('multiclass') or task_type.startswith('binary'):
        results_pd = pd.DataFrame(columns=['accuracy', 'f1_score', 'roc_auc'])
    elif task_type.startswith('continuous'):
        results_pd = pd.DataFrame(columns=['r2', 'mae', 'mse'])

    index_predictors_list = []
    eval_predictors_results_list = []
    for predictor in predictors:
        index_predictors_list.append(predictor.__class__.__name__)

    # Add a blank row to separate the predictors from experiment name
    index_predictors_list.append(' ')
    index_predictors_list.append('AVERAGE')
    index_predictors_list.append(experiment_name)
    index_predictors_list.append(' ')
    for k,v in hyperparams.items():
        index_predictors_list.append(str(k)+": "+str(v))

    # Loop through all the predictors and evaluate them
    for predictor in predictors:
        eval_predictor_results_dict = evaluate_predictor(
            predictor=predictor,
            eval_results_path=eval_results_path,
            task_type=task_type,
            logger=logger,
            X_test=X_test,
            y_test=y_test
        )
        eval_predictors_results_list.append(eval_predictor_results_dict)

    # add dummy empty rows to make len of index and data rows equal
    eval_predictors_results_list.append({})
    
    # calculate average of all the predictors for each evaluation metric
    dict_avg_eval_vals = {}
    for dict_eval in eval_predictors_results_list:
        for key, value in dict_eval.items():
            if dict_avg_eval_vals.get(key, None):
                dict_avg_eval_vals[key] += value
            else:
                dict_avg_eval_vals[key] = value
    for k in dict_avg_eval_vals.keys():
        dict_avg_eval_vals[k] = np.round(dict_avg_eval_vals[k] / len(predictors), decimals=3)
    eval_predictors_results_list.append({ k:v for k,v in dict_avg_eval_vals.items() })

    eval_predictors_results_list.append({})
    eval_predictors_results_list.append({})

    # add empty rows having the same length as the number of tabnet hyperparams
    for _ in range(len(hyperparams)):
        eval_predictors_results_list.append({})

    # create a new DataFrame from the new rows and index labels
    results_predictors_pd = pd.DataFrame(eval_predictors_results_list, index=index_predictors_list)

    results_pd = results_pd.append(results_predictors_pd)

    if store_eval_results_tmp_folder:
        open_df_temp(results_pd, experiment_name)

def save_test_results(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    y_pred: np.ndarray, 
    name_npz_file: Path,
    predictor_name: str
) -> None:
    """Saves compressed npz file with the test results containing the predictions and true target labels.
    
    When loading the npz file, the following keys data can be fetched:
        {
            y_true: true target labels (1D array),
            y_probs: predicted target labels (1D or multi-dimensional array, obtained after softmax),
            y_pred: predicted target labels (1D array, obtained after softmax and argmax)
        }

    Args:
        y_true (np.ndarray): The true target labels
        y_probs (np.ndarray): The probabilities of the labels e.g. [0.2, 0.8] after applying softmax
        y_pred (np.ndarray): The predicted target labels after applying softmax and argmax
        name_npz_file (Path): Name of the npz file to save the results to
    """

    # add current time in milliseconds to the file name
    # filename_with_curr_time = name_npz_file.stem + "_" + predictor_name + "_" + str(current_milli_time()) + name_npz_file.suffix
    filename_with_curr_time = name_npz_file.stem + "_" + predictor_name + name_npz_file.suffix
    name_npz_file = name_npz_file.parent / filename_with_curr_time

    # Save the fpr and tpr to a compressed npz file
    np.savez_compressed(name_npz_file, y_true=y_true, y_probs=y_probs, y_pred=y_pred)
