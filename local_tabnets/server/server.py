import argparse
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import torch.distributed.rpc as rpc
import torch.distributed as dist
from torch.nn import functional as F
from typing import Any, List, Tuple, Dict, Union
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, roc_auc_score, accuracy_score
import copy
import numpy as np
import os
from xgboost import XGBModel
from sklearn.preprocessing import OneHotEncoder
import warnings
from pathlib import Path
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_network import initialize_non_glu
from torch.nn import Linear
from torch.distributed.optim import DistributedOptimizer
from shared.general_utils import infer_optimizer
warnings.filterwarnings("ignore")

# from preprocessing.transformer import DataTransformer
from shared.general_utils import evaluate_multiple_predictors, plot_tsne_of_latent
from local_tabnets.one_client.client import LocalTabNetClient
import time

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

def _call_method(method, rref, *args, **kwargs):
    """helper for _remote_method()"""
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    """
    executes method(*args, **kwargs) on the from the machine that owns rref
    very similar to rref.remote().method(*args, **kwargs), but method() doesn't have to be in the remote scope
    """
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    print(param_rrefs)
    return param_rrefs

def save_epoch_times_and_plot(values, experiment_name: str):
    with plt.style.context("ggplot"):
        # Save the values to a file
        np.savez_compressed(SCRIPT_DIR / f"localtabnets_{experiment_name}_epoch_times_values_server.npz", epoch_times=values)

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
        fig.savefig(str(SCRIPT_DIR / f"localtabnets_{experiment_name}_epoch_times_server_plot.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

def save_train_and_valid_metrics(
    list_train_losses: list,
    list_valid_f1_scores: list, 
    list_valid_accuracy_scores: list, 
    list_valid_roc_auc_scores: list, 
    experiment_name: str, 
    title: str
) -> None:
    assert len(list_train_losses) == len(list_valid_f1_scores) == len(list_valid_accuracy_scores) == len(list_valid_roc_auc_scores), "The validation lists should have the same length."

    with plt.style.context("ggplot"):
        # for testing whether the training loss is decreasing and converging to a certain value 
        fig, ax = plt.subplots(figsize=(9, 11))

        file_path_loss = str(SCRIPT_DIR / f"{experiment_name}_train_loss_valid_metrics_values_lt_server.npz")
        name_image_file = str(SCRIPT_DIR / f"{experiment_name}_train_loss_valid_metrics_plots_lt_server.pdf")

        np.savez_compressed(
            file_path_loss, 
            train_losses=list_train_losses,
            valid_f1_scores=list_valid_f1_scores, 
            valid_accuracy_scores=list_valid_accuracy_scores,
            valid_roc_auc_scores=list_valid_roc_auc_scores
        )

        ax.set_title(title)
        ax.plot(list_train_losses, label='Train Losses (cross-entropy)', color='green')
        ax.plot(list_valid_f1_scores, label='Valid F1-Score', color='blue')
        ax.plot(list_valid_accuracy_scores, label='Valid Accuracy', color='orange')
        ax.plot(list_valid_roc_auc_scores, label='Valid ROC-AUC', color='red')

        # Add labels and title
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        plt.legend()  # Show legend

        fig.savefig(name_image_file, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)

        plt.show()
        plt.close(fig)


class LocalTabNetServer():
    """
    This is the class that encapsulates the functions that need to be run on the server side
    This is the main driver of the training procedure.
    """

    def __init__(
        self,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        client_rrefs,
        epochs,
        batch_size,
        use_cuda,
        valid_ratio,
        test_ratio,
        eval_out,
        seed,
        task_type,
        predictors, # determines which classifiers to use on aggergated latent data
        column_label_name,
        tabnet_hyperparams,
        tabnet_pretrainer_fit_params,
        optimizer,
        optimizer_params,
    ):
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.batch_size = batch_size
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.predictors = predictors
        self.seed = seed
        self.valid_ratio = valid_ratio
        self.task_type = task_type
        self.test_ratio = test_ratio
        self.tabnet_hyperparams = tabnet_hyperparams
        self.tabnet_pretrainer_fit_params = tabnet_pretrainer_fit_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.eval_results_path = SCRIPT_DIR / eval_out
        self.lambda_sparse = self.tabnet_hyperparams["lambda_sparse"]
        self.patience_finetuning = self.tabnet_pretrainer_fit_params["patience"]

        # target_type = type_of_target(self.raw_df.to_numpy())
        print(f"{self.y_train=}")
        print(f"{self.y_test=}")
        print("number of epochs in initialization: ", epochs)
        print("Target type: ", self.task_type)

        # keep a reference to the client
        self.client_rrefs = []
        for client_rref in client_rrefs:
            self.client_rrefs.append(client_rref)

        self.input_latent_dims = [] # record the coded latent dim of each client
        for client_rref in self.client_rrefs:
            self.input_latent_dims.append(client_rref.remote().get_latent_dim().to_here())

        # config for server side TabNet model, input is determined by the output layer dims of each client model 
        self.input_dim = int(np.sum(self.input_latent_dims))
        print("self.input_dim latent dims: ", self.input_dim)
        
        # initialize tabnet pretraining models at clients
        for client_rref in self.client_rrefs:
            client_rref.rpc_sync().init_local_tabnet_pretrainer()
            client_rref.rpc_sync().init_local_tabnet_finetuner()

        # determine criterion and unique class labels count given task type
        if self.task_type == "continuous": # regression case
            self.unique_class_labels_count = 1
            self.criteron = torch.nn.functional.mse_loss # regression case
        else:
            self.unique_class_labels_count = len(np.unique(self.y_train))
            self.criteron = torch.nn.functional.cross_entropy # classification case

        self.final_mapping = Linear(self.input_dim, self.unique_class_labels_count, bias=False).to(self.device)
        initialize_non_glu(self.final_mapping, self.input_dim, self.unique_class_labels_count)

        self.param_rrefs_list_tabnet_classifier = param_rrefs(self.final_mapping)
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_tabnet_classifier += client_rref.rpc_sync().register_tabnet_classifier()

        self.tabnet_finetuning_opt = DistributedOptimizer(
           infer_optimizer(optimizer_name=optimizer), 
           self.param_rrefs_list_tabnet_classifier, 
           **optimizer_params
        )

        # construct local dataloader
        self.data_loader = DataLoader(self.y_train, self.batch_size, shuffle=False)
        self.iterloader = iter(self.data_loader)

    def save_model(self, model, model_name):
        torch.save(model.state_dict(), model_name)

    def start_fit_pretraining_sync(self):
        """
        Sends the fit command to all the clients synchronously
        """

        # Start sync fit process on all clients TabNet Pretraining
        for client_rref in self.client_rrefs:
            client_rref.rpc_sync().start_fit_pretraining()

    def start_pretraining_fit_async(self):
        """
        Sends the fit command to all the clients asynchronously
        """
        pending_client_pretrainer_fits = []

        # Start async pretraining fit process on all clients 
        for client_rref in self.client_rrefs:
            pending_client_pretrainer_fits.append(client_rref.rpc_async().start_fit_pretraining())
        
        # Wait for all clients to finish pretraining and the commence finetuning
        for pending_fit, client_rref in zip(pending_client_pretrainer_fits, self.client_rrefs):
            pending_fit.wait()
        
        return

    def evaluate(self):
        """
        This function evaluates the model on the test data
        """
        
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
            experiment_name="local_tabnets",
            logger=None,
            eval_results_path=self.eval_results_path, 
            task_type=self.task_type, 
            X_test=self.X_test_downstream, 
            y_test=self.y_test_downstream
        )

    def reset_iterloader(self) -> None:
        self.iterloader = iter(self.data_loader)

    def get_batch_from_dataloader(self) -> torch.Tensor:
        try:
            data = next(self.iterloader)
        except StopIteration:
            self.reset_iterloader()
            data = next(self.iterloader)
            print("StopIteration: iterloader is reset and new batch is loaded")
        return data

    def forward_pass_valid_data_finetuning(self):
        """
        This function performs a forward pass of the TabNet encoder
        """
        assert self.final_mapping.training == False, "final_fc_mapping is not in eval mode"

        intermediate_results_encoder_dict = {}
        intermediate_M_Loss_dict = {}
        future_object_encoder_dict = {}

        # get output from local models
        for client_rref in self.client_rrefs:
            future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder_valid()

        for client_rref in self.client_rrefs:
            # wait for each future object to be ready
            client_inter_output_logits = future_object_encoder_dict[client_rref].wait()

            # append the intermediate value to the list
            intermediate_results_encoder_dict[client_rref], intermediate_M_Loss_dict[client_rref] = client_inter_output_logits

        total_M_loss = 0
        for m_loss in intermediate_M_Loss_dict.values():
            total_M_loss += m_loss

        # concatenate the intermediate tensor values from all clients by column, 
        # e.g. cat([[1,2,3]], [[4,5,6]]) -> [[1,2,3,4,5,6]]
        concated_latent_data = torch.cat(list(intermediate_results_encoder_dict.values()), dim = 1).to(self.device)

        logits = self.final_mapping(concated_latent_data)

        return logits, total_M_loss

    def validate_model(
        self,
        best_val_loss: float,
        current_patience: int,
    ) -> Union[torch.Tensor, bool, float, int]:
        self.final_mapping.eval()
        
        stop_training = False
        is_maximize = False

        valid_loss_roc_auc = -1
        valid_loss_accuracy = -1

        with torch.no_grad():
            logits, _ = self.forward_pass_valid_data_finetuning()
            softmaxed_logits = torch.softmax(logits, axis=1).cpu().detach().numpy()

            if self.task_type == "continuous":
                valid_loss = mean_squared_error(self.y_valid, logits.cpu().detach().numpy())
                print(f"FINETUNING VALID MSE LOSS: {valid_loss}")
                is_maximize = False
            else:
                is_maximize = True
                preds = np.argmax(softmaxed_logits, axis=1)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true=self.y_valid, 
                    y_pred=preds, 
                    average='weighted'
                )
                valid_loss = f1

                if self.unique_class_labels_count == 2:
                    valid_loss_roc_auc = roc_auc_score(self.y_valid, softmaxed_logits[:, 1], average="weighted", multi_class='ovr')
                else:
                    valid_loss_roc_auc = roc_auc_score(self.y_valid, softmaxed_logits, average="weighted", multi_class='ovr')
                    
                valid_loss_accuracy = accuracy_score(self.y_valid, preds)

                print(f"FINETUNING VALID F1_SCORE: {valid_loss}")
                print(f"FINETUNING VALID ROC-AUC: {valid_loss_roc_auc}")
                print(f"FINETUNING VALID ACCURACY: {valid_loss_accuracy}")

        loss_change = valid_loss - best_val_loss
        max_improved = is_maximize and loss_change > 0.0
        min_improved = (not is_maximize) and (-loss_change > 0.0)

        # Check for early stopping
        if max_improved or min_improved:
            best_val_loss = valid_loss
            current_patience = 0
        else:
            if current_patience >= self.patience_finetuning:
                print(f"Validation loss hasn't improved for {self.patience_finetuning=} epochs. Stopping training.")
                stop_training = True
            current_patience += 1

        # reset to train mode
        self.final_mapping.train()

        return valid_loss, stop_training, best_val_loss, current_patience, valid_loss_accuracy, valid_loss_roc_auc

    def forward_pass_finetuning(
            self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function performs a forward pass of the TabNet encoder
        """
        self.final_mapping.train()

        intermediate_results_encoder_dict = {}
        intermediate_M_losses = {}
        future_object_encoder_dict = {}

        # get output from local models
        for client_rref in self.client_rrefs:
            future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder()

        for client_rref in self.client_rrefs:
            # wait for each future object to be ready
            client_inter_output_logits = future_object_encoder_dict[client_rref].wait()

            # append the intermediate value to the list
            intermediate_results_encoder_dict[client_rref], intermediate_M_losses[client_rref] = client_inter_output_logits
        
        total_M_loss = 0
        for m_loss in intermediate_M_losses.values():
            total_M_loss += m_loss

        # concatenate the intermediate tensor values from all clients by column, 
        # e.g. cat([[1,2,3]], [[4,5,6]]) -> [[1,2,3,4,5,6]]
        concated_latent_reps = torch.cat(list(intermediate_results_encoder_dict.values()), dim = 1).to(self.device)

        logits = self.final_mapping(concated_latent_reps)

        return logits, total_M_loss

    def fit_finetuning(self):
        # number of batches in one epoch, under uneven division the ceil value is taken
        # to complete the last batch
        num_batches = max(int(np.ceil(len(self.y_train) / self.batch_size)), 1)
        best_val_loss = float('inf')

        if self.task_type != "continuous":
            best_val_loss = -best_val_loss # maximize validation metric in case of multiclass/binary classification
        current_patience = 0

        # list to keep track of losses per epoch
        list_of_train_losses_per_epoch = []

        # list to keep track of training time per epoch
        list_of_training_times_per_epoch = []

        # list of valid loss per epoch
        list_of_valid_losses_per_epoch = []

        # list of valid roc-auc per epoch
        list_of_valid_losses_per_epoch_roc_auc = []

        # list of valid accuracy per peoch
        list_of_valid_losses_per_epoch_accuracy = []

        # reset iterloaders so that we start from the beginning of the dataset
        self.reset_iterloader()
        for client_rref in self.client_rrefs:
            client_rref.rpc_sync().reset_iterloader()

        training_start_time = time.time()
        for epoch in range(1, self.epochs+1):
            print(f"Epoch: {epoch}")
            losses_sum = 0

            epoch_start_time = time.perf_counter()
            for batch_amount in range(1, num_batches+1):
                with dist.autograd.context() as tabnet_finetuning_context:
                    y_target = self.get_batch_from_dataloader()
                    y_target = y_target.to(self.device)
                    y_target = y_target.view(-1)

                    if self.task_type == "continuous":
                        y_target = y_target.float()

                    # forward pass of clients and server for pretraining
                    # y_pred should be a tensor of shape (batch_size, 2)
                    # in binary case
                    logits, M_loss = self.forward_pass_finetuning()

                    # calculate the loss of the forawrd pass
                    # in case of regression, logits represents one value for each sample
                    loss = self.criteron(logits, y_target)

                    # Add the overall sparsity loss
                    loss = loss - self.lambda_sparse * M_loss

                    losses_sum += loss.item()

                    dist.autograd.backward(tabnet_finetuning_context, [loss])
                    
                    self.tabnet_finetuning_opt.step(tabnet_finetuning_context)

            epoch_end_time = time.perf_counter()
            print(f"Epoch {epoch} training time: {epoch_end_time - epoch_start_time}")
            list_of_training_times_per_epoch.append(epoch_end_time - epoch_start_time)

            if self.task_type == "continuous":
                print(f"MSE LOSS: {losses_sum / num_batches}")
            else:
                print(f"CROSS ENTROPY LOSS: {losses_sum / num_batches}")

            # append the average loss per epoch to the list
            list_of_train_losses_per_epoch.append(losses_sum / num_batches)

            valid_loss, stop_training, best_val_loss, current_patience, valid_acc, valid_roc_auc = self.validate_model(
                best_val_loss=best_val_loss,
                current_patience=current_patience,
            )

            list_of_valid_losses_per_epoch.append(valid_loss)
            list_of_valid_losses_per_epoch_roc_auc.append(valid_roc_auc)
            list_of_valid_losses_per_epoch_accuracy.append(valid_acc)

            print(f"{best_val_loss=}")
            print(f"{current_patience=}")

            assert self.final_mapping.training == True, "partial_decoder is not in training mode after validation"

            if stop_training:
                break

        training_end_time = time.time()
        print(f"Finetuning done in: {training_end_time - training_start_time}")

        # set all client finetuners to eval
        for client_rref in self.client_rrefs:
            client_rref.rpc_sync().set_eval_finetuner()
        self.final_mapping.eval()
        assert self.final_mapping.training == False, "final mapping layer should be in eval mode"

        return list_of_train_losses_per_epoch, list_of_valid_losses_per_epoch, list_of_training_times_per_epoch, list_of_valid_losses_per_epoch_accuracy, list_of_valid_losses_per_epoch_roc_auc

    def fit(self):
        # pretrain the models at all clients 
        self.start_pretraining_fit_async()

        list_of_train_losses_per_epoch, list_of_valid_losses_per_epoch, list_of_training_times_per_epoch, list_of_valid_losses_per_epoch_accuracy, list_of_valid_losses_per_epoch_roc_auc = \
            self.fit_finetuning()

        # save epoch time values of finetuning
        save_epoch_times_and_plot(
            list_of_training_times_per_epoch, 
            "finetuning",
        )

        save_train_and_valid_metrics(
            list_train_losses=list_of_train_losses_per_epoch,
            list_valid_f1_scores=list_of_valid_losses_per_epoch,
            list_valid_accuracy_scores=list_of_valid_losses_per_epoch_accuracy,
            list_valid_roc_auc_scores=list_of_valid_losses_per_epoch_roc_auc,
            experiment_name="finetuning",
            title="TabNet Finetuner (LT) Train Losses and Valid Metrics"
        )

        # fetch train latent data from all clients
        # and concatenate them into one tensor
        X_train_latent_data_clients = {}
        for client_ref in self.client_rrefs:
            X_train_latent_data_clients[client_ref] = client_ref.rpc_sync().forward_latent_data(
                fetch_test_data=False,
                fetch_valid_data=False
            )

        # this is input for classifier and not needed for backprop thus detach it
        # detaching it also prevents a pytorch error about this
        # convert the latent data to numpy arrays as the TabNet classifier expects numpy arrays
        X_train_latent = torch.cat(list(X_train_latent_data_clients.values()), dim=1).detach().cpu().numpy()
        print("X_train_latent_data_clients[0].shape: ", list(X_train_latent_data_clients.values())[0].shape)
        print("X_train_latent.shape: ", X_train_latent.shape)
        print(f"{X_train_latent=}")
        
        # fetch valid latent data from all clients
        # and concatenate them into one tensor
        X_valid_latent_clients = {}
        for client_ref in self.client_rrefs:
            X_valid_latent_clients[client_ref] = client_ref.rpc_sync().forward_latent_data(
                fetch_test_data=False,
                fetch_valid_data=True
            )
        X_valid_latent = torch.cat(list(X_valid_latent_clients.values()), dim=1).detach().cpu().numpy()
        print("X_valid_latent_clients[0].shape: ", list(X_valid_latent_clients.values())[0].shape)
        print("X_valid_latent.shape: ", X_valid_latent.shape)
        print(f"{X_valid_latent=}")

        # fetch test latent data from all clients
        # and concatenate them into one tensor
        X_test_latent_clients = {}
        for client_ref in self.client_rrefs:
            X_test_latent_clients[client_ref] = client_ref.rpc_sync().forward_latent_data(
                fetch_test_data=True,
                fetch_valid_data=False
            )
        X_test_latent = torch.cat(list(X_test_latent_clients.values()), dim=1).detach().cpu().numpy()
        print("X_test_latent_clients[0].shape: ", list(X_test_latent_clients.values())[0].shape)
        print("X_test_latent.shape: ", X_test_latent.shape)
        print(f"{X_test_latent=}")

        # concatenate the train, valid and test latent data
        X_latent_data = np.concatenate([X_train_latent, X_valid_latent, X_test_latent], axis=0)
        print("X_latent_data.shape: ", X_latent_data.shape)
        print(f"{X_latent_data=}")
        y_latent_data = np.concatenate([self.y_train, self.y_valid, self.y_test], axis=0)
        print("y_latent_data.shape: ", y_latent_data.shape)
        print(f"{y_latent_data=}")

        # split the dataset into train and test sets
        if self.task_type == "continuous":
            X_train_downstream, X_test_downstream, y_train_downstream, y_test_downstream = train_test_split(
                X_latent_data,
                y_latent_data,
                test_size=(self.test_ratio + self.valid_ratio),
                random_state=self.seed,
                shuffle=True,
            )
        else:
            X_train_downstream, X_test_downstream, y_train_downstream, y_test_downstream = train_test_split(
                X_latent_data,
                y_latent_data,
                test_size=(self.test_ratio + self.valid_ratio),
                random_state=self.seed,
                shuffle=True,
                stratify=y_latent_data,
            )
        print(f"{X_train_downstream=}")
        print(f"{X_train_downstream.shape=}")
        print(f"{X_test_downstream=}")
        print(f"{X_test_downstream.shape=}")
        print(f"{y_train_downstream=}")
        print(f"{y_train_downstream.shape=}")
        print(f"{y_test_downstream=}")
        print(f"{y_test_downstream.shape=}")

        self.X_test_downstream = X_test_downstream
        self.y_test_downstream = y_test_downstream

        # training completed at all clients. Now we can draw the latent data from the encoder.
        # we can use this latent data to train a classifier
        
        # train the predictors
        for predictor in self.predictors:
            print(f"Fitting predictor: {predictor}...")
            predictor.fit(X_train_downstream, y_train_downstream)
            print(f"Predictor {predictor} fitted!")
        
        return X_train_downstream, y_train_downstream

def run(
    rank,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    X_train_client_cols: List[Tuple[Dict[str, Union[List[str], Dict[str, float]]], pd.DataFrame]],
    splitted_X_test: List[pd.DataFrame],
    splitted_X_valid: List[pd.DataFrame],
    task_type: str,
    master_addr: str, 
    master_port: int, 
    num_clients: int, 
    use_cuda: bool,
    eval_out: str,
    column_label_name: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int,
    epochs: int,
    predictors: List[Union[BaseEstimator, XGBModel]],
    tabnet_hyperparams: dict,
    tabnet_pretrainer_fit_params: dict,
    optimizer: str,
    optimizer_params: dict,
):
    # set environment information
    world_size = num_clients + 1
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    print("world size: ", world_size, f"tcp://{master_addr}:{master_port}")
    
    if rank == 0:  # this is run only on the server side
        # https://github.com/pytorch/pytorch/issues/55615
        rpc.init_rpc(
            "server",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.PROCESS_GROUP,
            rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                num_send_recv_threads=8, rpc_timeout=999999, init_method=f"tcp://{master_addr}:{master_port}"
            ),
        )   

        # rememeber the keys that have list values
        keys_tabnet_hyperparams_with_list_values = []
        for key, value in tabnet_hyperparams.items():
            if isinstance(value, list):
                keys_tabnet_hyperparams_with_list_values.append(key)
                assert len(value) == num_clients, f"length of {key=},{len(value)=} must be equal to the number of clients {num_clients}"

        clients = []
        if keys_tabnet_hyperparams_with_list_values:
            for worker_idx, (cols_info_dict, X_train_df) in enumerate(X_train_client_cols):                
                curr_client_hyperparams_dict = copy.deepcopy(tabnet_hyperparams)
                for key in keys_tabnet_hyperparams_with_list_values:
                    curr_client_hyperparams_dict[key] = tabnet_hyperparams[key][worker_idx]
                print(f"worker {worker_idx + 1} has {curr_client_hyperparams_dict=}")

                clients.append(
                    rpc.remote(
                        "client"+str(worker_idx + 1), 
                        LocalTabNetClient, 
                        kwargs=dict(
                            X_train_df = X_train_df,
                            X_valid_df = splitted_X_valid[worker_idx],
                            X_test_df = splitted_X_test[worker_idx],
                            y_valid_np = y_valid,
                            y_train_np = y_train,
                            task_type = task_type,
                            client_id = worker_idx + 1,
                            epochs = epochs, 
                            cols_info_dict = cols_info_dict,
                            use_cuda = use_cuda, 
                            batch_size = batch_size,
                            tabnet_hyperparams = curr_client_hyperparams_dict,
                            tabnet_pretrainer_fit_params = tabnet_pretrainer_fit_params,
                            optimizer = optimizer,
                            optimizer_params = optimizer_params,
                            seed = seed,
                            train_ratio = train_ratio,
                            test_ratio = test_ratio,
                        )
                    )
                )
                print("register remote client"+str(worker_idx + 1), clients[0])
        else:
            for worker_idx, (cols_info_dict, X_train_df) in enumerate(X_train_client_cols):
                clients.append(
                    rpc.remote(
                        "client"+str(worker_idx + 1), 
                        LocalTabNetClient, 
                        kwargs=dict(
                            X_train_df = X_train_df,
                            X_valid_df = splitted_X_valid[worker_idx],
                            X_test_df = splitted_X_test[worker_idx],
                            y_valid_np = y_valid,
                            y_train_np = y_train,
                            task_type = task_type,
                            client_id = worker_idx + 1,
                            epochs = epochs, 
                            cols_info_dict = cols_info_dict,
                            use_cuda = use_cuda, 
                            batch_size = batch_size,
                            tabnet_hyperparams = tabnet_hyperparams,
                            tabnet_pretrainer_fit_params = tabnet_pretrainer_fit_params,
                            optimizer = optimizer,
                            optimizer_params = optimizer_params,
                            seed = seed,
                            train_ratio = train_ratio,
                            test_ratio = test_ratio,
                        )
                    )
                )
                print("register remote client"+str(worker_idx + 1), clients[0])

        local_tabnets_server = LocalTabNetServer(
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            task_type=task_type,
            client_rrefs=clients,
            epochs=epochs,
            use_cuda=use_cuda, 
            batch_size=batch_size,
            predictors=predictors,
            eval_out=eval_out,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            seed=seed,
            column_label_name=column_label_name,
            tabnet_hyperparams = tabnet_hyperparams,
            tabnet_pretrainer_fit_params = tabnet_pretrainer_fit_params,
            optimizer = optimizer,
            optimizer_params = optimizer_params,
        )
        X_train_latent, y_train = local_tabnets_server.fit()
        local_tabnets_server.evaluate()

    elif rank != 0:
        raise ValueError("Only rank 0 is allowed to run the server")

    rpc.shutdown()
