import os
from pathlib import Path
import logging

from sklearn.multiclass import OneVsRestClassifier


os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parent)

import argparse
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import yaml
import multiprocessing as mp
# The following is added to support CUDA being used in forked processes
from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
from shared.vertical_dataset_splitter import DataSplitter 
from shared.general_utils import train_valid_test_split

from tabnet_vfl.server.server import run as run_tabnet_vfl
from tabnet_vfl_local_encoder.server.server import run as run_tabnet_local_encoder_vfl
from tabnet_vfl.one_client.client import run as run_tabnet_vfl_client
from tabnet_vfl_local_encoder.one_client.client import run as run_tabnet_local_encoder_vfl_client
from local_tabnets.server.server import run as run_local_tabnets
from local_tabnets.one_client.client import run as run_local_tabnets_client
from central_baseline_tabnet.central_tabnet import run as run_central_tabnet

from sklearn.utils.multiclass import type_of_target
from sklearn.base import BaseEstimator
from xgboost import XGBModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.calibration import CalibratedClassifierCV

# Logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log format
)
logger = logging.getLogger(__name__)

# This is here to ensure values in yaml files that are prefixed with '!!python/tuple' are parsed as tuples
# use it by calling yaml.load(yaml_file, Loader=TupleSafeLoader)
class TupleSafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

TupleSafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    TupleSafeLoader.construct_python_tuple
)

# Global Constants
CHOICES_EXPERIMENT_TYPE: List[str] = ['tabnet_vfl', 'local_tabnets', 'central_tabnet', 'tabnet_vfl_local_encoder']
CHOICES_TASK_TYPE: List[str] = ['binary', 'multiclass', 'continuous']
PREDICTORS_CHOICES: List[str] = [
    'randomforest',
    'decisiontree',
    'svm',
    'mlp',
    'logisticreg',
    'linearreg',
    'xgboost',
]

def path_to_dataset(path_to_dataset_folder: Path, dataset_name: str) -> str:
    """Return the path to the dataset.

    Args:
        path_to_dataset_folder (Path): Absolute path to the folder containing the dataset.
        dataset_name (str): Name of the dataset to use.

    Returns:
        Path: Absolute path to the dataset.
    """
    folder_path = Path(path_to_dataset_folder)
    dataset_path = list(folder_path.glob(dataset_name + '.*'))
    if not dataset_path:
        raise ValueError(f'Invalid/Non-existing dataset name. {dataset_path=} empty.')
    return str(dataset_path[0].resolve())

def make_list_of_predictors(predictor_names: List[str], task_type: str) -> List[Union[BaseEstimator, XGBModel]]:
    """Make a list of predictors given a list of names.

    Args:
        predictor_names (List[str]): List of names of the predictors to use.
        task_type (str): Type of task to perform. Can be either 'classification' or 'regression'.

    Returns:
        List[Union[BaseEstimator, XGBModel]]: List of predictors.
    """
    # double checking before continuing
    if task_type not in CHOICES_TASK_TYPE:
        raise ValueError(f'Invalid task type. Got {task_type} instead.')
    return [infer_predictor_model_given_name(predictor_name, task_type) for predictor_name in predictor_names]

def infer_predictor_model_given_name(predictor_name: str, task_type: str) -> Union[BaseEstimator, XGBModel]:
    if task_type == 'multiclass' or task_type == 'binary':
        if predictor_name == 'logisticreg':
            return LogisticRegression(random_state=42)
        elif predictor_name == 'xgboost':
            return XGBClassifier(random_state=42)
        elif predictor_name == 'randomforest':
            return RandomForestClassifier(random_state=42)
        elif predictor_name == 'decisiontree':
            return DecisionTreeClassifier(random_state=42)
        elif predictor_name == 'mlp':
            return MLPClassifier(random_state=42)
        elif predictor_name == 'svm':
            # to be able to use predict_proba, but should make sure that for cv=5 (default) each class has at least 5 samples
            # not using SVC anymore since it does not support multiclass one-hot encoded labels for training or evaluation which is needed to calculate roc_auc
            return CalibratedClassifierCV(LinearSVC(random_state=42), method='sigmoid') 
        else:
            raise ValueError(f'Invalid predictor name. Got {predictor_name} instead.')
    elif task_type == 'continuous':
        if predictor_name == 'linearreg':
            return LinearRegression()
        elif predictor_name == 'xgboost':
            return XGBRegressor(random_state=42)
        elif predictor_name == 'randomforest':
            return RandomForestRegressor(random_state=42)
        elif predictor_name == 'decisiontree':
            return DecisionTreeRegressor(random_state=42)
        elif predictor_name == 'mlp':
            return MLPRegressor(random_state=42)
        elif predictor_name == 'svm':
            return LinearSVR(random_state=42)
        else:
            raise ValueError(f'Invalid predictor name. Got {predictor_name} instead.')
    else:
        raise ValueError(f'Invalid task type. Got {task_type} instead.')

def infer_dataypes_for_split_cols(
    split_cols_df: List[pd.DataFrame], 
    mixed_cols: list, 
    integer_cols: list, 
    categorical_cols: list
) -> List[dict]:
    """Infer the datatypes of the columns in the split dataframes.
    Return example (NOTICE THE DIFFERENT DICT VALUES):
    [
        {mixed_columns: {'col1': [0.0], 'col2': [0.0]}, integer_columns: ['col1', 'col3'], categorical_columns: ['col4']},
        {mixed_columns: {}, integer_columns: ['col5'], categorical_columns: ['col6']},
        ...
    ]
    """
    cols_of_split_df = []
    for split_df in split_cols_df:
        cols_used = {'mixed_columns': {}, 'integer_columns': [], 'categorical_columns': []}
        col_names = split_df.columns.tolist()
        for col_name in col_names:
            if col_name in mixed_cols:
                cols_used['mixed_columns'][col_name] = [0.0]
            if col_name in integer_cols:
                cols_used['integer_columns'].append(col_name)
            if col_name in categorical_cols:
                cols_used['categorical_columns'].append(col_name)
            if col_name not in mixed_cols and col_name not in integer_cols and col_name not in categorical_cols:
                raise ValueError(f'Invalid column name. {col_name=} not in {mixed_cols=} or {integer_cols=} or {categorical_cols=}.')
        cols_of_split_df.append(cols_used)
    return cols_of_split_df

def listify_comma_separated_string(comma_separated_string: str) -> List[str]:
    """Convert a comma separated string to a list of strings.

    Args:
        comma_separated_string (str): String containing comma separated values.

    Returns:
        List[str]: List of strings.
    """
    return [x.strip() for x in comma_separated_string.split(',')]

def filter_df(raw_df: pd.DataFrame, cols_to_keep: list) -> pd.DataFrame:
    """Filter the dataframe to keep only the columns of interest.

    Args:
        raw_df (pd.DataFrame): Raw dataframe.
        cols_to_keep (list): List of columns to keep.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    return raw_df.drop(columns=[col for col in raw_df.columns if col not in cols_to_keep])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_folder_path', type=Path, required=True, help='Relative/Absolute path to the folder containing the datasets.')
    parser.add_argument('--config_path', type=Path, required=True, help='Relative/Absolute path to the YAML file containing config info, e.g., use_cuda, num_clients, precentages_columns_each_client etc... .')
    parser.add_argument('--hyperparams_path', type=Path, required=True, help='Relative/Absolute path to the YAML file containing tabnet hyperparameters info, e.g., batch_size, n_d, n_a etc... .')
    parser.add_argument('--experiment_type', type=str, required=True, choices=CHOICES_EXPERIMENT_TYPE, default='tabnet_vfl', help='What experiment to run. Default is "tabnet_vfl".')
    parser.add_argument('--task_type', type=str, choices=CHOICES_TASK_TYPE, help='Overwrites the automatic task inference if needed.')
    parser.add_argument('--predictors', type=str, required=True, help='What predictors to use for evaluation. Multiple predictors can be specified by separating them with a comma.')
    parser.add_argument('--eval_out', type=str, default='eval_data.npz', help='The file name where evaluation results are stored. Default is "eval_data.npz"')
    args = parser.parse_args()

    # Add .npz extension to eval_out if not present
    if not args.eval_out.endswith('.npz'):
        args.eval_out += '.npz'
    
    logger.info(f"{args=}")
    
    # Load config and hyperparams from YAML files
    with open(args.hyperparams_path, "r") as f:
        hyperparams_dict = yaml.load(f, Loader=TupleSafeLoader)
    with open(args.config_path, "r") as f:
        config_dict = yaml.load(f, Loader=TupleSafeLoader)

    # fetch globally used vars
    seed = config_dict["seed"]
    batch_size = hyperparams_dict["batch_size"]
    epochs = hyperparams_dict["epochs"]
    experiment_type = args.experiment_type
    num_clients = None

    # Load in the dataset to use for the experiment
    path_to_dataset = path_to_dataset(args.datasets_folder_path, config_dict['dataset']) # type: ignore
    if path_to_dataset.endswith('.csv'):
        raw_df = pd.read_csv(path_to_dataset) # type: ignore
    elif path_to_dataset.endswith('.xlsx'):
        raw_df = pd.read_excel(path_to_dataset)
    else:
        raise ValueError(f'Invalid dataset format. Expected .csv or .xlsx but got {path_to_dataset} instead.')
    
    raw_df_columns_original = raw_df.columns.tolist()

    # Keep only the feature columns that are specificed in the config file and label column
    # This enables us to filter out the columns that we don't need for the experiments via yaml config file
    # This feature is optional, since we always use the full dataset columns.
    # NOTE: atleast one of the attributes in yaml file of config (mixed_columns, integer_columns, categorical_columns) must be non-empty
    # if one of the attributes is non-empty, then all the columns of the dataset to use for the experiment must be specified 
    raw_df = filter_df(
        raw_df=raw_df,
        cols_to_keep=config_dict["mixed_columns"]+config_dict["integer_columns"]+config_dict["categorical_columns"]+[config_dict['column_label_name']]
    )

    assert set(raw_df_columns_original) == set(raw_df.columns.tolist()), f"Columns mismatch after filtering. Diff: {set(raw_df_columns_original)-set(raw_df.columns.tolist())=}"

    logger.info(f"BEFORE : {raw_df.columns.tolist()=}")
    logger.info(f"BEFORE shape: {raw_df.shape=}")

    # Fetch label column and drop it from the dataset
    # this is done so that the columns that are split among clients are feature columns
    # and not including label info
    server_labels_raw_df = raw_df[[ config_dict['column_label_name'] ]]
    raw_df_features = filter_df(
        raw_df=raw_df,
        cols_to_keep=list(set(config_dict["mixed_columns"]+config_dict["integer_columns"]+config_dict["categorical_columns"]))
    )

    # Infer the type of the label column
    if args.task_type is not None:
        label_task_type = args.task_type
    else:
        label_task_type = type_of_target(server_labels_raw_df)
    logger.info(f"{label_task_type=}")

    if label_task_type != "continuous" and label_task_type != "binary" and label_task_type != "multiclass":
        raise ValueError(f"Invalid label task type given: {label_task_type}. Must be one of: continuous, binary, multiclass")

    df_features_columns = raw_df_features.columns.tolist()

    logger.info(f"AFTER : {df_features_columns=}")
    logger.info(f"AFTER shape: {raw_df_features.shape=}")

    assert config_dict['column_label_name'] in set(raw_df.columns.tolist())
    assert config_dict['column_label_name'] in set(server_labels_raw_df.columns.tolist())
    assert config_dict['column_label_name'] not in set(raw_df_features.columns.tolist())
   
    # Infer which predictor to used based on the given classification/regression task type
    predictor_models = make_list_of_predictors(listify_comma_separated_string(args.predictors), label_task_type)

    # NOTE: Make sure labels for all datasets are preprocessed with label encoding
    preprocessed_y_labels = server_labels_raw_df.to_numpy() 

    # SPLIT FEATURES into train and test sets. 
    # NOTE: this does not work with regression datasets since there is no stratification possible
    # if a label is continuous as there are not enough indentical labels to stratify on
    if label_task_type == "continuous":
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
            raw_df_features, 
            preprocessed_y_labels,
            test_ratio=config_dict["test_ratio"], 
            valid_ratio=config_dict["valid_ratio"],
            random_state=seed, 
            shuffle=True,
        )
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    else:
        # SPLIT LABELS into train and test sets.
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
            raw_df_features,
            preprocessed_y_labels, 
            test_ratio=config_dict["test_ratio"],
            valid_ratio=config_dict["valid_ratio"], 
            random_state=seed, 
            shuffle=True, 
            stratify=preprocessed_y_labels
        )

    logger.info(f"[BOOTSTRAPPER] - {y_train=}")
    logger.info(f"[BOOTSTRAPPER] - {y_train.shape=}")
    logger.info(f"[BOOTSTRAPPER] - {y_valid=}")
    logger.info(f"[BOOTSTRAPPER] - {y_valid.shape=}")
    logger.info(f"[BOOTSTRAPPER] - {y_test=}")
    logger.info(f"[BOOTSTRAPPER] - {y_test.shape=}")

    # convert to numpy arrays
    X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()

    if label_task_type != "continuous":
        # DEBUG checking label distribution
        # Compute the histogram
        hist, bins = np.histogram(y_train, bins=np.arange(y_train.min(), y_train.max() + 2))
        # logger.info the frequency counts for each label
        for label, count in zip(bins[:-1], hist):
            logger.info(f"y_train Label {label}: {count} occurrences") 
        # Compute the histogram
        hist, bins = np.histogram(y_valid, bins=np.arange(y_valid.min(), y_valid.max() + 2))
        # logger.info the frequency counts for each label
        for label, count in zip(bins[:-1], hist):
            logger.info(f"y_valid Label {label}: {count} occurrences") 
        # Compute the histogram
        hist, bins = np.histogram(y_test, bins=np.arange(y_test.min(), y_test.max() + 2))
        # logger.info the frequency counts for each label
        for label, count in zip(bins[:-1], hist):
            logger.info(f"y_test Label {label}: {count} occurrences") 


    # setup and start experiment and split features of dataset according to config 
    if args.experiment_type == 'tabnet_vfl':
        tabnet_vfl_conf = config_dict['tabnet_vfl']
        tabnet_vfl_hyperparams = hyperparams_dict['tabnet_vfl']

        num_clients = tabnet_vfl_conf['num_clients']

        ds_split_cols_X_train = DataSplitter(array_to_split=X_train, columns=df_features_columns).split_data(tabnet_vfl_conf["data_column_split"])
        ds_split_cols_X_valid = DataSplitter(array_to_split=X_valid, columns=df_features_columns).split_data(tabnet_vfl_conf["data_column_split"])
        ds_split_cols_X_test = DataSplitter(array_to_split=X_test, columns=df_features_columns).split_data(tabnet_vfl_conf["data_column_split"])
        
        logger.info(f"[BOOTSTRAPPER] - {ds_split_cols_X_train[0]=}")
        logger.info(f"[BOOTSTRAPPER] - {ds_split_cols_X_valid[0]=}")
        logger.info(f"[BOOTSTRAPPER] - {ds_split_cols_X_test[0]=}")

        inferred_cols_of_split_df = infer_dataypes_for_split_cols(
            split_cols_df=ds_split_cols_X_train, # does not matter whether to use X_train or X_test here, since both have the same columns
            mixed_cols=config_dict["mixed_columns"], 
            integer_cols=config_dict["integer_columns"], 
            categorical_cols=config_dict["categorical_columns"]
        )
        zipped_df_cols_and_inferred_cols = list(zip(inferred_cols_of_split_df, ds_split_cols_X_train))

        # Start server process with rank 0
        rank = 0
        server_proc = mp.Process(
            target=run_tabnet_vfl, 
            kwargs=dict(
                rank=rank,
                y_train=y_train,
                y_valid=y_valid,
                y_test=y_test,
                X_train_per_client=zipped_df_cols_and_inferred_cols,
                splitted_X_test=ds_split_cols_X_test,
                splitted_X_valid=ds_split_cols_X_valid,
                task_type=label_task_type,
                master_addr=tabnet_vfl_conf["master_addr"],
                master_port=tabnet_vfl_conf["master_port"],
                num_clients=tabnet_vfl_conf["num_clients"],
                use_cuda=tabnet_vfl_conf["use_cuda"],
                epoch_failure_probability=tabnet_vfl_conf["epoch_failure_probability"],
                eval_out=args.eval_out,
                column_label_name=config_dict['column_label_name'],
                train_ratio=config_dict["train_ratio"],
                valid_ratio=config_dict["valid_ratio"],
                test_ratio=config_dict["test_ratio"],
                seed=seed,
                batch_size=batch_size,
                epochs=epochs,
                predictors=predictor_models,
                tabnet_hyperparams=tabnet_vfl_hyperparams["tabnet_hyperparams"],
                decoder_split_ratios=tabnet_vfl_hyperparams["decoder_split_ratios"],
                optimizer=tabnet_vfl_hyperparams["optimizer"],
                optimizer_params=tabnet_vfl_hyperparams["optimizer_params"],
            )
        )
        server_proc.start()

        # Start client processes each with a different rank
        client_procs = []
        for _ in range(tabnet_vfl_conf["num_clients"]):
            rank += 1
            new_client_proc = mp.Process(
                target=run_tabnet_vfl_client, 
                kwargs=dict(
                    rank=rank,
                    master_addr=tabnet_vfl_conf["master_addr"], 
                    master_port=tabnet_vfl_conf["master_port"], 
                    num_clients=tabnet_vfl_conf["num_clients"], 
                )
            )
            new_client_proc.start()
            client_procs.append(new_client_proc)

        # Wait for all processes to finish
        server_proc.join()
        for client_proc in client_procs:
            client_proc.join()

    elif args.experiment_type == 'tabnet_vfl_local_encoder':
        tabvfl_local_encoder_conf = config_dict['tabnet_vfl']
        tabvfl_local_encoder_hyperparams = hyperparams_dict['tabvfl_local_encoder']

        num_clients = tabvfl_local_encoder_conf['num_clients']

        ds_split_cols_X_train = DataSplitter(array_to_split=X_train, columns=df_features_columns).split_data(tabvfl_local_encoder_conf["data_column_split"])
        ds_split_cols_X_valid = DataSplitter(array_to_split=X_valid, columns=df_features_columns).split_data(tabvfl_local_encoder_conf["data_column_split"])
        ds_split_cols_X_test = DataSplitter(array_to_split=X_test, columns=df_features_columns).split_data(tabvfl_local_encoder_conf["data_column_split"])
        
        logger.info(f"[BOOTSTRAPPER] - {ds_split_cols_X_train[0]=}")
        logger.info(f"[BOOTSTRAPPER] - {ds_split_cols_X_test[0]=}")

        inferred_cols_of_split_df = infer_dataypes_for_split_cols(
            split_cols_df=ds_split_cols_X_train, # does not matter whether to use X_train or X_test here, since both have the same columns
            mixed_cols=config_dict["mixed_columns"], 
            integer_cols=config_dict["integer_columns"], 
            categorical_cols=config_dict["categorical_columns"]
        )
        zipped_df_cols_and_inferred_cols = list(zip(inferred_cols_of_split_df, ds_split_cols_X_train))

        # Start server process with rank 0
        rank = 0
        server_proc = mp.Process(
            target=run_tabnet_local_encoder_vfl, 
            args=(
                rank,
                y_train,
                y_valid,
                y_test,
                zipped_df_cols_and_inferred_cols,
                ds_split_cols_X_valid,
                ds_split_cols_X_test,
                label_task_type,
                tabvfl_local_encoder_conf["master_addr"],
                tabvfl_local_encoder_conf["master_port"],
                tabvfl_local_encoder_conf["num_clients"],
                tabvfl_local_encoder_conf["use_cuda"],
                args.eval_out,
                config_dict['column_label_name'],
                config_dict["train_ratio"],
                config_dict["valid_ratio"],
                config_dict["test_ratio"],
                seed,
                batch_size,
                epochs,
                predictor_models,
                tabvfl_local_encoder_hyperparams["tabnet_hyperparams"],
                tabvfl_local_encoder_hyperparams["decoder_split_ratios"],
                tabvfl_local_encoder_hyperparams["optimizer"],
                tabvfl_local_encoder_hyperparams["optimizer_params"],
            )
        )
        server_proc.start()

        # Start client processes each with a different rank
        client_procs = []
        for _ in range(tabvfl_local_encoder_conf["num_clients"]):
            rank += 1
            new_client_proc = mp.Process(
                target=run_tabnet_local_encoder_vfl_client, 
                args=(
                    rank,
                    tabvfl_local_encoder_conf["master_addr"], 
                    tabvfl_local_encoder_conf["master_port"], 
                    tabvfl_local_encoder_conf["num_clients"], 
                )
            )
            new_client_proc.start()
            client_procs.append(new_client_proc)

        # Wait for all processes to finish
        server_proc.join()
        for client_proc in client_procs:
            client_proc.join()

    elif args.experiment_type == 'local_tabnets':
        local_tabnets_conf = config_dict['local_tabnets']
        local_tabnets_hyperparams = hyperparams_dict['local_tabnets']

        num_clients = local_tabnets_conf['num_clients']

        ds_split_cols_X_train = DataSplitter(array_to_split=X_train, columns=df_features_columns).split_data(local_tabnets_conf["data_column_split"])
        ds_split_cols_X_valid = DataSplitter(array_to_split=X_valid, columns=df_features_columns).split_data(local_tabnets_conf["data_column_split"])
        ds_split_cols_X_test = DataSplitter(array_to_split=X_test, columns=df_features_columns).split_data(local_tabnets_conf["data_column_split"])

        inferred_cols_of_split_df = infer_dataypes_for_split_cols(
            split_cols_df=ds_split_cols_X_train, 
            mixed_cols=config_dict["mixed_columns"], 
            integer_cols=config_dict["integer_columns"], 
            categorical_cols=config_dict["categorical_columns"]
        )
        logger.info(f"[BOOTSTRAPPER] - {ds_split_cols_X_train[0]=}")
        logger.info(f"[BOOTSTRAPPER] - {ds_split_cols_X_test[0]=}")

        zipped_df_cols_and_inferred_cols = list(zip(inferred_cols_of_split_df, ds_split_cols_X_train))

        # Start server process with rank 0
        rank = 0
        server_proc = mp.Process(
            target=run_local_tabnets, 
            args=(
                rank,
                y_train,
                y_valid,
                y_test,
                zipped_df_cols_and_inferred_cols,
                ds_split_cols_X_test,
                ds_split_cols_X_valid,
                label_task_type,
                local_tabnets_conf["master_addr"],
                local_tabnets_conf["master_port"],
                local_tabnets_conf["num_clients"],
                local_tabnets_conf["use_cuda"],
                args.eval_out,
                config_dict['column_label_name'],
                config_dict["train_ratio"],
                config_dict["valid_ratio"],
                config_dict["test_ratio"],
                seed,
                batch_size,
                epochs,
                predictor_models,
                local_tabnets_hyperparams["tabnet_pretrainer_params"],
                local_tabnets_hyperparams["tabnet_pretrainer_fit_params"],
                local_tabnets_hyperparams["optimizer"],
                local_tabnets_hyperparams["optimizer_params"],
            )
        )
        server_proc.start()

        # Start client processes each with a different rank
        client_procs = []
        for _ in range(local_tabnets_conf["num_clients"]):
            rank += 1
            new_client_proc = mp.Process(
                target=run_local_tabnets_client, 
                args=(
                    rank,
                    local_tabnets_conf["master_addr"],
                    local_tabnets_conf["master_port"], 
                    local_tabnets_conf["num_clients"], 
                )
            )
            new_client_proc.start()
            client_procs.append(new_client_proc)           

        # Wait for all processes to finish
        server_proc.join()
        for client_proc in client_procs:
            client_proc.join()

    else: # centralized tabnet experiment
        central_tabnet_conf = config_dict['central_tabnet']
        cental_tabnet_hyperparams = hyperparams_dict['central_tabnet']

        # this is done to ensure that the column order is preserved when mixed, integer and categorical columns are appended
        # the order in which the columns are appended to each column type is important and does effect the preprocessing
        # so this is done to ensure consistency with other experiments
        inferred_datatypes_for_raw_df = infer_dataypes_for_split_cols(
            [raw_df_features], 
            config_dict["mixed_columns"], 
            config_dict["integer_columns"], 
            config_dict["categorical_columns"]
        )
        categorical_columns = inferred_datatypes_for_raw_df[0]["categorical_columns"]
        integer_columns = inferred_datatypes_for_raw_df[0]["integer_columns"]
        mixed_columns = inferred_datatypes_for_raw_df[0]["mixed_columns"]
        cols_info_dict = {
            "categorical_columns": categorical_columns,
            "integer_columns": integer_columns,
            "mixed_columns": mixed_columns
        }
        logger.info(f"[BOOTSTRAPPER] Central TabNet: {cols_info_dict=}")
        
        # start central tabnet experiment
        central_tabnet_proc = mp.Process(
            target=run_central_tabnet, 
            kwargs=dict(
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                y_train=y_train,
                y_valid=y_valid,
                y_test=y_test,
                label_task_type=label_task_type,
                predictors=predictor_models, 
                column_label_name=config_dict['column_label_name'],
                train_ratio=config_dict["train_ratio"],
                val_ratio=config_dict["valid_ratio"],
                test_ratio=config_dict["test_ratio"],
                eval_out=args.eval_out,
                tabnet_hyperparams=cental_tabnet_hyperparams["tabnet_pretrainer_params"],
                tabnet_pretrainer_fit_params=cental_tabnet_hyperparams["tabnet_pretrainer_fit_params"],
                optimizer=cental_tabnet_hyperparams["optimizer"],
                optimizer_params=cental_tabnet_hyperparams["optimizer_params"],
                cols_info_dict=cols_info_dict,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                use_cuda=central_tabnet_conf["use_cuda"],
            )
        )
        central_tabnet_proc.start()
        central_tabnet_proc.join()
    
    # logger.info information about the experiment to keep track of the experiments
    experiment_metadata = {}
    experiment_metadata["columns_used"] = df_features_columns
    experiment_metadata["dataset"] = config_dict['dataset'] 
    experiment_metadata["experiment"] = args.experiment_type
    experiment_metadata["seed"] = seed
    experiment_metadata["num_clients"] = "" if not num_clients else num_clients
    experiment_metadata["predictors"] = args.predictors
    logger.info(f"[BOOTSTRAPPER] Experiment Information: ")
    logger.info(experiment_metadata)

    
    