import pandas as pd
import numpy as np
import torch
from shared.general_utils import evaluate_multiple_predictors, infer_optimizer, round_to_n_decimals
from shared.general_utils import plot_tsne_of_latent
from sklearn.model_selection import train_test_split
from typing import Any, List, Tuple, Dict, Union
from sklearn.base import BaseEstimator
from xgboost import XGBModel
from pathlib import Path
from torchinfo import summary
import warnings
warnings.filterwarnings("ignore")
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from shared.tabnet_logger import TabNetLogger
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from matplotlib import pyplot as plt

random_seed = 42

# Set the random seed for NumPy
np.random.seed(random_seed)

# Set the random seed for PyTorch CPU operations
torch.manual_seed(random_seed)

# Set the random seed for PyTorch GPU operations (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem

logger = TabNetLogger(SCRIPT_NAME).get_logger()

def save_epoch_times_and_plot(values, training_phase: str):
    plot_path = str(SCRIPT_DIR / f"epoch_time_values_{training_phase}_central_plot.pdf")
    file_path = SCRIPT_DIR / f"epoch_time_values_{training_phase}_central.npz"

    with plt.style.context("ggplot"):
        # Save the values to a file
        np.savez_compressed(file_path, epoch_times=values)

        # Create a figure and subplots
        fig, ax = plt.subplots(figsize=(9, 11))

        # Plot the values
        ax.plot(values, color='blue', linestyle='-', marker='o', markersize=4)

        # Set plot title and axis labels
        ax.set_title("Epoch Times (In Seconds)", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Time (in seconds)", fontsize=12)

        # Set x-axis tick labels font size
        ax.tick_params(axis='x', labelsize=10)

        # Set y-axis tick labels font size
        ax.tick_params(axis='y', labelsize=10)

        # Set grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save the plot to a file
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

def save_train_and_valid_metrics(
    list_train_losses: list,
    list_valid_f1_scores: list, 
    list_valid_accuracy_scores: list, 
    list_valid_roc_auc_scores: list, 
    training_phase: str, 
    title: str
) -> None:
    assert training_phase == "finetuning" or training_phase == "pretraining", "training phase string must be either finetuning or pretraining"

    if list_train_losses and list_valid_f1_scores and list_valid_accuracy_scores and list_valid_roc_auc_scores:
        assert len(list_train_losses) == len(list_valid_f1_scores) == len(list_valid_accuracy_scores) == len(list_valid_roc_auc_scores), "The validation lists should have the same length."

    with plt.style.context("ggplot"):
        # for testing whether the training loss is decreasing and converging to a certain value 
        fig, ax = plt.subplots(figsize=(9, 11))

        file_path_loss = str(SCRIPT_DIR / f"{training_phase}_train_loss_valid_metrics_values_ct.npz")
        name_image_file = str(SCRIPT_DIR / f"{training_phase}_train_loss_valid_metrics_plots_ct.pdf")

        np.savez_compressed(
            file_path_loss, 
            train_losses=list_train_losses,
            valid_f1_scores=list_valid_f1_scores, 
            valid_accuracy_scores=list_valid_accuracy_scores,
            valid_roc_auc_scores=list_valid_roc_auc_scores
        )

        ax.set_title(title)
        if training_phase == "finetuning":
            ax.plot(list_train_losses, label='Train Losses (cross-entropy)', color='green')
            ax.plot(list_valid_f1_scores, label='Valid F1-Score', color='blue')
            ax.plot(list_valid_accuracy_scores, label='Valid Accuracy', color='orange')
            ax.plot(list_valid_roc_auc_scores, label='Valid ROC-AUC', color='red')
        else:
            ax.plot(list_train_losses, label='Train Losses (UnsupervisedLoss)', color='green')
            ax.plot(list_valid_f1_scores, label='Valid UnsupervisedLoss', color='blue')


        # Add labels and title
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        plt.legend()  # Show legend

        fig.savefig(name_image_file, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
        
        plt.show()
        plt.close(fig)

class CustomF1Metric(Metric):
    def __init__(self):
        self._name = "f1_score"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred, 
            average='weighted'
        )
        return f1
    
class CustomROCAUCMetric(Metric):
    def __init__(self):
        self._name = "roc_auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        num_classes = len(np.unique(y_true))
        if num_classes == 2:
            weighted_auc = roc_auc_score(y_true, y_score[:, 1], average='weighted', multi_class='ovr')
        else:
            weighted_auc = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
        return weighted_auc

class CentralTabnet:
    def __init__(
        self,
        X_train: np.ndarray,
        X_valid: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        predictors: List[Union[BaseEstimator, XGBModel]], 
        column_label_name: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        eval_out: str,
        tabnet_hyperparams: Dict,
        tabnet_pretrainer_fit_params: Dict,
        optimizer: str,
        optimizer_params: Dict,
        cols_info_dict: Dict,
        seed: int,
        epochs: int,
        batch_size: int,
        use_cuda: bool, 
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.predictors = predictors
        self.column_label_name = column_label_name
        self.train_ratio = train_ratio
        self.valid_ratio = val_ratio
        self.test_ratio = test_ratio
        self.eval_out = SCRIPT_DIR / eval_out
        self.tabnet_hyperparams = tabnet_hyperparams
        self.tabnet_pretrainer_fit_params = tabnet_pretrainer_fit_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.cols_info_dict = cols_info_dict
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.task_type = task_type

        if self.task_type != "continuous":
            self.y_train = y_train.ravel()
            self.y_valid = y_valid.ravel()
            self.y_test = y_test.ravel()
        else:
            self.y_train = y_train
            self.y_valid = y_valid
            self.y_test = y_test

        logger.info(f"Using seed: {self.seed}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using epochs: {self.epochs}")
        logger.info(f"Using batch_size: {self.batch_size}")
        logger.info(f"Using use_cuda: {self.use_cuda}")
        logger.info(f"Using train_ratio: {self.train_ratio}")
        logger.info(f"Using valid_ratio: {self.valid_ratio}")
        logger.info(f"Using test_ratio: {self.test_ratio}")
        logger.info(f"Using eval_out: {self.eval_out}")
        logger.info(f"Using tabnet_hyperparams: {self.tabnet_hyperparams}")
        logger.info(f"Using tabnet_pretrainer_fit_params: {self.tabnet_pretrainer_fit_params}")
        logger.info(f"Using optimizer: {self.optimizer}")
        logger.info(f"Using optimizer_params: {self.optimizer_params}")
        logger.info(f"Using cols_info_dict: {self.cols_info_dict}")
        logger.info(f"Target type: {self.task_type}")

        print(f"X_train: {self.X_train=}")
        print(f"X_valid: {self.X_valid=}")
        print(f"X_test: {self.X_test=}")

        print(f"y_train: {self.y_train=}")
        print(f"y_valid: {self.y_valid=}")
        print(f"y_test: {self.y_test=}")

        self.tabnet_pretrainer_model = TabNetPretrainer(
            input_dim=self.X_train.shape[1],
            **self.tabnet_hyperparams,
            optimizer_fn=infer_optimizer(optimizer),
            optimizer_params=optimizer_params,
            seed=self.seed,
            device_name=self.device,
        )

        print(f"TabNet Pretrainer model: {self.tabnet_pretrainer_model=}")

    def get_latent_representation(self, input_data) -> np.ndarray:
        assert self.tabnet_finetuner_model.network.training == False, "TabNet Finetuner model network is still in training mode!"
        assert self.tabnet_finetuner_model.network.tabnet.training == False, "TabNet Finetuner model tabnet is still in training mode!"
        assert self.tabnet_finetuner_model.network.tabnet.encoder.training == False, "TabNet Finetuner model encoder is still in training mode!"

        # DEBUGGING
        with torch.no_grad():
            debug_steps_output_train, _ = self.tabnet_finetuner_model.network.tabnet.encoder(torch.from_numpy(self.X_train).to(self.device).float())
            debug_steps_summed_train = torch.sum(torch.stack(debug_steps_output_train, dim=0), dim=0)
            print(f"First train sample latent: {debug_steps_summed_train[0:2,:]=}")
            debug_steps_output_test, _ = self.tabnet_finetuner_model.network.tabnet.encoder(torch.from_numpy(self.X_test).to(self.device).float())
            debug_steps_summed_test = torch.sum(torch.stack(debug_steps_output_test, dim=0), dim=0)
            print(f"First test sample latent: {debug_steps_summed_test[0:2,:]=}")

        with torch.no_grad():
            steps_output, _ = self.tabnet_finetuner_model.network.tabnet.encoder(torch.from_numpy(input_data).to(self.device).float())
            latent_representation = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        logger.info(f"{latent_representation.requires_grad=}")
        return latent_representation.cpu().numpy()

    def fit(self):
        logger.info("Fitting TabNet Pretrainer...")
        self.tabnet_pretrainer_model.fit(
            X_train=self.X_train,
            eval_set=[self.X_valid],
            eval_name=["valid"],
            max_epochs=self.epochs, 
            batch_size=self.batch_size, 
            **self.tabnet_pretrainer_fit_params
        )
        logger.info(f"TabNet Pretrainer {self.tabnet_pretrainer_model} fitted!")

        pretraining_epoch_training_times = self.tabnet_pretrainer_model.epoch_training_times

        if self.task_type == "continuous":
            self.tabnet_finetuner_model = TabNetRegressor(
                seed=self.seed,
                device_name=self.device,
                optimizer_fn=infer_optimizer(self.optimizer),
                optimizer_params=self.optimizer_params,
                epsilon=self.tabnet_pretrainer_model.network.epsilon,
                momentum=self.tabnet_pretrainer_model.momentum,
                gamma=self.tabnet_pretrainer_model.gamma,
                clip_value=self.tabnet_pretrainer_model.clip_value,
                lambda_sparse=self.tabnet_hyperparams["lambda_sparse"],
                n_shared_decoder=self.tabnet_pretrainer_model.network.n_shared_decoder, 
                n_indep_decoder=self.tabnet_pretrainer_model.network.n_indep_decoder,
            )
        else:
            self.tabnet_finetuner_model = TabNetClassifier(
                seed=self.seed,
                device_name=self.device,
                optimizer_fn=infer_optimizer(self.optimizer),
                optimizer_params=self.optimizer_params,
                epsilon=self.tabnet_pretrainer_model.network.epsilon,
                momentum=self.tabnet_pretrainer_model.momentum,
                gamma=self.tabnet_pretrainer_model.gamma,
                clip_value=self.tabnet_pretrainer_model.clip_value,
                lambda_sparse=self.tabnet_hyperparams["lambda_sparse"],
                n_shared_decoder=self.tabnet_pretrainer_model.network.n_shared_decoder, 
                n_indep_decoder=self.tabnet_pretrainer_model.network.n_indep_decoder,
            )
        
        print(f"TabNet Finetuner model: {self.tabnet_finetuner_model=}")

        logger.info("Fitting TabNet Finetuner...")
        # Create a new dictionary without the pretraining_ratio key-value pair
        filtered_fit_dict = {k: v for k, v in self.tabnet_pretrainer_fit_params.items() if k != "pretraining_ratio"}
        print(f"{filtered_fit_dict=}")
        
        if self.task_type == "continuous":
            self.tabnet_finetuner_model.fit(
                X_train=self.X_train,
                y_train=self.y_train,
                eval_set=[(self.X_valid, self.y_valid)],
                eval_name=["valid"],
                from_unsupervised=self.tabnet_pretrainer_model,
                max_epochs=self.epochs,
                batch_size=self.batch_size,
                **filtered_fit_dict
            )
        else:
            self.tabnet_finetuner_model.fit(
                X_train=self.X_train,
                y_train=self.y_train,
                eval_set=[(self.X_valid, self.y_valid)],
                eval_name=["valid"],
                eval_metric=["accuracy", CustomROCAUCMetric, CustomF1Metric],
                from_unsupervised=self.tabnet_pretrainer_model,
                max_epochs=self.epochs,
                batch_size=self.batch_size,
                **filtered_fit_dict
            )
        logger.info(f"TabNet Finetuner {self.tabnet_finetuner_model} fitted!")

        finetuning_epoch_training_times = self.tabnet_finetuner_model.epoch_training_times

        self.tabnet_finetuner_model.network.eval()
        self.tabnet_finetuner_model.network.tabnet.eval()
        self.tabnet_finetuner_model.network.tabnet.encoder.eval()

        # Concatenate along axis 0 (rows)
        X_total_dataset = np.concatenate((
            self.get_latent_representation(self.X_train),
            self.get_latent_representation(self.X_valid),
            self.get_latent_representation(self.X_test),
        ), axis=0)
        y_total_dataset = np.concatenate((self.y_train, self.y_valid, self.y_test), axis=0)

        logger.info(f"{X_total_dataset=}")
        logger.info(f"{X_total_dataset.shape=}")
        logger.info(f"{y_total_dataset=}")
        logger.info(f"{y_total_dataset.shape=}")

        # split the dataset into train and test sets
        if self.task_type == "continuous":
            X_train_downstream, X_test_downstream, y_train_downstream, y_test_downstream = train_test_split(
                X_total_dataset,
                y_total_dataset,
                test_size=(self.test_ratio + self.valid_ratio),
                random_state=self.seed,
                shuffle=True,
            )
        else:
            X_train_downstream, X_test_downstream, y_train_downstream, y_test_downstream = train_test_split(
                X_total_dataset,
                y_total_dataset,
                test_size=(self.test_ratio + self.valid_ratio),
                random_state=self.seed,
                shuffle=True,
                stratify=y_total_dataset,
            )

        self.X_test_downstream = X_test_downstream
        self.y_test_downstream = y_test_downstream

        logger.info(f"{X_train_downstream=}")
        logger.info(f"{X_train_downstream.shape=}")
        logger.info(f"{X_test_downstream=}")
        logger.info(f"{X_test_downstream.shape=}")
        logger.info(f"{y_train_downstream=}")
        logger.info(f"{y_train_downstream.shape=}")
        logger.info(f"{y_test_downstream=}")
        logger.info(f"{y_test_downstream.shape=}")

        for predictor in self.predictors:
            logger.info(f"Fitting {predictor}...")
            predictor.fit(X_train_downstream, y_train_downstream)
            logger.info(f"{predictor} fitted!")
        
        return X_train_downstream, y_train_downstream, pretraining_epoch_training_times, finetuning_epoch_training_times

    def evaluate(self) -> Tuple[list, list, list, dict]:
        hyperparams_log: Dict[str, Any] = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            **self.tabnet_hyperparams,
            "optimizer": self.optimizer,
            **self.optimizer_params,
            **self.tabnet_pretrainer_fit_params
        }

        evaluate_multiple_predictors(
            predictors=self.predictors,
            hyperparams=hyperparams_log,
            experiment_name="central_tabnet",
            eval_results_path=self.eval_out,
            logger=None,
            task_type=self.task_type,
            X_test=self.X_test_downstream,
            y_test=self.y_test_downstream,
        )

        valid_metric_vals = {}
        if self.task_type == "continuous":
            valid_metric_vals[f"valid_mse"] = self.tabnet_finetuner_model.history[f"valid_mse"]
        else:
            valid_metric_vals["valid_f1_score"] = self.tabnet_finetuner_model.history["valid_f1_score"]
            valid_metric_vals["valid_roc_auc"] = self.tabnet_finetuner_model.history["valid_roc_auc"]
            valid_metric_vals["valid_accuracy"] = self.tabnet_finetuner_model.history["valid_accuracy"]

        return self.tabnet_pretrainer_model.history["loss"], self.tabnet_finetuner_model.history["loss"], self.tabnet_pretrainer_model.history["valid_unsup_loss_numpy"], valid_metric_vals


def run(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    label_task_type: str,
    predictors: List[Union[BaseEstimator, XGBModel]], 
    column_label_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    eval_out: str,
    tabnet_hyperparams: Dict,
    tabnet_pretrainer_fit_params: Dict,
    optimizer: str,
    optimizer_params: Dict,
    cols_info_dict: Dict,
    seed: int,
    epochs: int,
    batch_size: int,
    use_cuda: bool,
):
    central_tabnet = CentralTabnet(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        task_type=label_task_type,
        predictors=predictors,
        column_label_name=column_label_name,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        eval_out=eval_out,
        tabnet_hyperparams=tabnet_hyperparams,
        tabnet_pretrainer_fit_params=tabnet_pretrainer_fit_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        cols_info_dict=cols_info_dict,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        use_cuda=use_cuda,
    )
    X_train_latent, y_train, pretrain_epoch_times, finetuning_epoch_times = central_tabnet.fit()
    pretrainer_train_losses, finetuner_train_losses, pretrainer_valid_losses, finetuner_valid_metric_vals = central_tabnet.evaluate()

    # save epoch time values of pretraining
    save_epoch_times_and_plot(
        pretrain_epoch_times, 
        "pretraining",
    )

    # save epoch time values of finetuning
    save_epoch_times_and_plot(
        finetuning_epoch_times, 
        "finetuning"
    )

    save_train_and_valid_metrics(
        list_train_losses=pretrainer_train_losses,
        list_valid_f1_scores=pretrainer_valid_losses,
        list_valid_accuracy_scores=[],
        list_valid_roc_auc_scores=[],
        training_phase="pretraining",
        title="Central TabNet Pretraining Train Loss + Valid Loss (UnsupervisedLoss)"
    )

    if label_task_type == "continuous":
        save_train_and_valid_metrics(
            list_train_losses=finetuner_train_losses,
            list_valid_f1_scores=finetuner_valid_metric_vals["valid_mse"],
            list_valid_accuracy_scores=[],
            list_valid_roc_auc_scores=[],
            training_phase="finetuning",
            title="Central TabNet Finetuining Train Loss + Valid Metrics"
        )
    else:
        save_train_and_valid_metrics(
            list_train_losses=finetuner_train_losses,
            list_valid_f1_scores=finetuner_valid_metric_vals["valid_f1_score"],
            list_valid_accuracy_scores=finetuner_valid_metric_vals["valid_accuracy"],
            list_valid_roc_auc_scores=finetuner_valid_metric_vals["valid_roc_auc"],
            training_phase="finetuning",
            title="Central TabNet Finetuining Train Loss + Valid Metrics"
        )