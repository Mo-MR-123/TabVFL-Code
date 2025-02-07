import argparse
from typing import Any, Dict, List, Tuple, Union
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch.distributed.rpc as rpc
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from xgboost import XGBModel
from torchinfo import summary
import warnings 
warnings.filterwarnings("ignore")

import time 
from tabnet_vfl.server.tabnet_model import TabNetEncoder, TabNetDecoder, initialize_non_glu
from shared.general_utils import infer_optimizer, evaluate_multiple_predictors, plot_tsne_of_latent
from shared.tabnet_logger import TabNetLogger
from tabnet_vfl.one_client.client import TabNetClient

# Global constants
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem

# init seeds for reproducibility and logging module
logger = TabNetLogger(SCRIPT_NAME).get_logger()


def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    # logger.info(param_rrefs)
    return param_rrefs


def save_losses_plot(
    list_train_losses: list, 
    list_valid_losses: list, 
    training_phase: str, 
    title: str
) -> None:
    assert len(list_train_losses) == len(list_valid_losses), "The training and validation losses should have the same length."

    with plt.style.context("ggplot"):
        # for testing whether the training loss is decreasing and converging to a certain value 
        fig, ax = plt.subplots(figsize=(9, 11))

        file_path_loss = str(SCRIPT_DIR / f"{training_phase}_loss_values_tabnetvfl.npz")
        name_image_file = str(SCRIPT_DIR / f"{training_phase}_loss_plot_tabnetvfl.pdf")

        np.savez_compressed(file_path_loss, train_losses=list_train_losses, valid_losses=list_valid_losses)

        ax.set_title(title)
        ax.plot(list_train_losses, label='Training Loss', color='blue')
        ax.plot(list_valid_losses, label='Validation Loss', color='orange')

        # Add labels and title
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        plt.legend()  # Show legend

        fig.savefig(name_image_file, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
        
        plt.show()
        plt.close(fig)

def histogram_client_offline_counts(dict_client_offline_counts: dict, training_phase: str):
    with plt.style.context("ggplot"):
        # Extract keys and values
        keys = list(dict_client_offline_counts.keys())
        values = list(dict_client_offline_counts.values())

        # Save the dictionary as an npz file
        np.savez_compressed(SCRIPT_DIR / f'histogram_offline_counts_clients_{training_phase}_tabnetvfl.npz', **dict_client_offline_counts)
        
        # Create subplots
        fig, ax = plt.subplots(figsize=(9, 11))

        # Plot histogram
        ax.bar(keys, values)

        # Add labels and title
        ax.set_xlabel('Client IDs')
        ax.set_ylabel('Maximum Loss Difference (Absolute Values)') # between precious and current iteration
        ax.set_title(f'Histogram of maximum loss difference reached when clients go offline during {training_phase}')

        fig.savefig(str(SCRIPT_DIR / f'histogram_offline_counts_clients_{training_phase}.pdf'), format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.5)

        # Show the plot
        plt.show()

def save_epoch_times_and_plot(values: list, training_phase: str):
    with plt.style.context("ggplot"):
        file_path = SCRIPT_DIR / f"epoch_time_values_{training_phase}_tabnetvfl.npz"
        plot_path = str(SCRIPT_DIR / f"epoch_time_plot_{training_phase}_tabnetvfl.pdf")

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
        fig.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)

        plt.close(fig)

class TabNetServer():
    """
    This is the class that encapsulates the functions that need to be run on the server side
    This is the main driver of the training procedure.
    """

    def __init__(
        self, 
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        predictors: List[Union[BaseEstimator, XGBModel]],
        client_rrefs: list, 
        seed: int,
        tabnet_hyperparams: dict,
        decoder_split_ratios: list,
        epoch_failure_probability: float,
        optimizer: str,
        optimizer_params: dict,
        eval_out: str,
        task_type: str,
        batch_size: int,
        epochs: int, 
        use_cuda: bool, 
        train_ratio: float,
        valid_ratio: float,
        test_ratio: float,
        **kwargs
    ):
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.batch_size = batch_size
        self.predictors = predictors
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.tabnet_hyperparams = tabnet_hyperparams
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.seed = seed
        self.task_type = task_type

        # Path used to save the ROC curve data generated during evaluation
        self.eval_results_path = SCRIPT_DIR / eval_out
        
        # settings client failure simulation
        self.handle_client_failures_pretraining = True
        self.handle_client_failures_finetuning = True
        self.rng_failure = np.random.default_rng(self.seed)  # Create a random number generator with the specified seed
        self.epoch_failure_probability = epoch_failure_probability # probability of a client being offline in a given epoch
        self.use_caching_method = True
        self.mini_batch_client_failure = False
        self.pretrain_epochs = self.epochs
        self.finetune_epochs = self.epochs

        self.patience_pretraining = 10 # early stopping patience pretrain
        self.patience_finetuning = 10 # early stopping patience finetune

        print(f"{self.y_train=}")
        print(f"{self.y_test=}")

        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Predictors used: {self.predictors}")
        logger.info(f"TabNet Pretraining Hyperparameters: {tabnet_hyperparams}")
        logger.info(f"Seed: {self.seed}")
        logger.info(f"Task type: {self.task_type}")
        logger.info(f"device: {self.device}")
        logger.info(f"{self.pretrain_epochs=}")
        logger.info(f"{self.finetune_epochs=}")
        logger.info(f"{self.epoch_failure_probability=}")
        logger.info(f"{self.handle_client_failures_finetuning=}")
        logger.info(f"{self.handle_client_failures_pretraining=}")
        logger.info(f"{self.use_caching_method=}")
        logger.info(f"{self.mini_batch_client_failure=}")

        # keep a reference to the client
        self.client_rrefs = []
        for client_rref in client_rrefs:
            self.client_rrefs.append(client_rref)

        self.y_train_len = self.y_train.shape[0]
        logger.info(f"train data len: {self.y_train_len}")

        # HYPERPARAMETERS TABNET
        self.n_d: int = tabnet_hyperparams['n_d']
        self.n_a: int = tabnet_hyperparams['n_a']
        self.n_steps: int = tabnet_hyperparams['n_steps']
        self.gamma: float = tabnet_hyperparams['gamma'] # 1.3
        self.epsilon: float = tabnet_hyperparams['epsilon']
        self.n_independent: int = tabnet_hyperparams['n_independent']
        self.n_shared: int = tabnet_hyperparams['n_shared']
        self.mask_type: str = tabnet_hyperparams['mask_type']
        self.pretraining_ratio: float = tabnet_hyperparams['pretraining_ratio']
        self.n_shared_decoder: int = tabnet_hyperparams['n_shared_decoder']
        self.n_indep_decoder: int = tabnet_hyperparams['n_indep_decoder']
        self.virtual_batch_size: int = tabnet_hyperparams['virtual_batch_size']
        self.momentum_virtual_batch_size: float = tabnet_hyperparams['momentum'] # 0.02
        self.lambda_sparse: float = tabnet_hyperparams['lambda_sparse'] # For sparsity loss, not important for encoder-decoder design but important for supervised training

        # check if the input parameters are valid and expected
        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")
        if self.n_d <= 0:
            raise ValueError("n_d should be a positive integer.")
        if len(self.client_rrefs) <= 0:
            raise ValueError("There should be at least one client.")
        if self.n_d < len(self.client_rrefs):
            raise ValueError("n_d should be greater than or equal to the number of clients.")

        logger.info(f"n_d: {self.n_d}")
        logger.info(f"n_a: {self.n_a}")
        logger.info(f"n_steps: {self.n_steps}")
        logger.info(f"gamma: {self.gamma}")
        logger.info(f"epsilon: {self.epsilon}")
        logger.info(f"n_independent: {self.n_independent}")
        logger.info(f"n_shared: {self.n_shared}")
        logger.info(f"mask_type: {self.mask_type}")
        logger.info(f"pretraining_ratio: {self.pretraining_ratio}")
        logger.info(f"n_shared_decoder: {self.n_shared_decoder}")
        logger.info(f"n_indep_decoder: {self.n_indep_decoder}")
        logger.info(f"virtual_batch_size: {self.virtual_batch_size}")
        logger.info(f"momentum_virtual_batch_size: {self.momentum_virtual_batch_size}")
        logger.info(f"lambda_sparse: {self.lambda_sparse}")

        # uniform splits of n_d of local decoders at clients
        self.local_decoder_split_ratios = self.uniform_distribution_decoder_splits(self.n_d, self.client_rrefs)
        logger.info(f"local_decoder_split_ratios: {self.local_decoder_split_ratios.values()}")

        if sum(self.local_decoder_split_ratios.values()) != self.n_d:
            raise ValueError(
                f"""The sum of the decoder split ratios should be equal to n_d. 
                Got split values {self.local_decoder_split_ratios.values()} 
                summing to {sum(self.local_decoder_split_ratios.values())} with n_d = {self.n_d}""")
             
        # initialize tabnet models in client side
        for client_rref in self.client_rrefs:
            client_rref.rpc_sync().init_local_encoder_decoder(self.local_decoder_split_ratios[client_rref])

        self.output_encoder_clients_dims_list = [] # record the coded data dim of each client
        for client_rref in self.client_rrefs:
            self.output_encoder_clients_dims_list.append(client_rref.remote().get_local_encoder_dim().to_here())
        
        # config for server side TabNet model, input is determined by the output layer dims of each client model 
        self.encoder_input_dim = np.sum(self.output_encoder_clients_dims_list)
        logger.info(f"output_encoder_clients_dims_list: {self.output_encoder_clients_dims_list}")  
        logger.info(f"encoder_input_dim: {self.encoder_input_dim}")

        torch.manual_seed(self.seed)
        # initing the partial encoder and decoder of tabnet
        self.partial_encoder = TabNetEncoder(
            input_dim=self.encoder_input_dim,
            output_dim=self.encoder_input_dim,
            n_a=self.n_a,
            n_d=self.n_d,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum_virtual_batch_size,
            mask_type=self.mask_type,
            group_attention_matrix=torch.eye(self.encoder_input_dim).to(self.device), # NOTE: this attention matrix needs to be the same size as the intermediate results from clients combined
        ).to(self.device)
        self.partial_decoder = TabNetDecoder(
            input_dim=self.encoder_input_dim,
            n_d=self.n_d,
            n_steps=self.n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum_virtual_batch_size,
        ).to(self.device)

        # determine criterion and unique class labels count given task type
        if self.task_type == "continuous": # regression case
            self.unique_class_labels_count = 1
            self.criteron = torch.nn.functional.mse_loss # regression case
        else:
            self.unique_class_labels_count = len(np.unique(self.y_train))
            self.criteron = torch.nn.functional.cross_entropy # classification case
        
        logger.info(f"criterion: {self.criteron}")
        logger.info(f"unique_class_labels_count: {self.unique_class_labels_count}") 
        self.final_fc_mapping = torch.nn.Linear(self.n_d, self.unique_class_labels_count, bias=False).to(self.device)
        initialize_non_glu(self.final_fc_mapping, self.n_d, self.unique_class_labels_count)
        logger.info(f"self.final_fc_mapping: {self.final_fc_mapping}") 

        if self.use_caching_method and self.handle_client_failures_finetuning:
            self.client_to_tensor_batches = {}
            # init dict
            for client_rref in self.client_rrefs:
                self.client_to_tensor_batches[client_rref] = []
            print(f"{self.client_to_tensor_batches=}")

        if self.use_caching_method and self.handle_client_failures_pretraining:
            self.client_to_tensor_batches_pretrain = {}
            self.client_to_tensor_masks_pretrain = {}
            # init dicts
            for client_rref in self.client_rrefs:
                self.client_to_tensor_batches_pretrain[client_rref] = []
                self.client_to_tensor_masks_pretrain[client_rref] = []
            print(f"{self.client_to_tensor_batches_pretrain=}")
            print(f"{self.client_to_tensor_masks_pretrain=}")

        # create list of param_rrefs for pretraining TabNet encoder-decoder
        # in server side and client side
        self.param_rrefs_list_tabnet = param_rrefs(self.partial_encoder)
        self.param_rrefs_list_tabnet += param_rrefs(self.partial_decoder)
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_tabnet += client_rref.rpc_sync().register_local_encoder()
            self.param_rrefs_list_tabnet += client_rref.rpc_sync().register_local_decoder()

        # create separate list of param_rrefs for classification task with TabNet encoder
        # and classification model for predictions in server side and client side
        self.param_rrefs_list_tabnet_classifier = param_rrefs(self.partial_encoder)
        self.param_rrefs_list_tabnet_classifier += param_rrefs(self.final_fc_mapping)
        for client_rref in self.client_rrefs:
            self.param_rrefs_list_tabnet_classifier += client_rref.rpc_sync().register_local_encoder()
        
        # setup optimizer for pretraining TabNet encoder-decoder
        self.tabnet_pretraining_opt = DistributedOptimizer(
           infer_optimizer(optimizer_name=optimizer), 
           self.param_rrefs_list_tabnet, 
           **optimizer_params
        )

        # setup optimizer for classification task with TabNet encoder
        self.tabnet_finetuning_opt = DistributedOptimizer(
           infer_optimizer(optimizer_name=optimizer), 
           self.param_rrefs_list_tabnet_classifier, 
           **optimizer_params
        )

        # construct local dataloader
        self.data_loader = DataLoader(self.y_train, self.batch_size, shuffle=False)
        self.iterloader = iter(self.data_loader)

    def uniform_distribution_decoder_splits(self, n_d, clients_rrefs):
        """Splits the latent dimension from encoder into uniform chunks based on the number of clients

        Args:
            n_d (int): Latent vector dimension
            clients_rrefs (int): Number of clients

        Returns:
            dict[str, int]: Dictionary of client_rref to number of columns it receives
        """
        num_clients = len(clients_rrefs)
        each_client_receives = n_d // num_clients
        remaining = n_d % num_clients
        client_distribution = {client_rref: each_client_receives for client_rref in clients_rrefs}
        for i in range(remaining):
            client_distribution[clients_rrefs[i]] += 1
        return client_distribution
    
    def split_tensor_cols(self, tensor, split_ratios: dict) -> List[torch.Tensor]:
        """This function splits the tensor into chunks based on the split ratios.
        The split ratios are a dictionary of client_rref to number of columns it receives
            e.g.: 
            {
                client_rref1: 3, 
                client_rref2: 3, 
                client_rref3: 4
            }
        """
        split_indices = list(split_ratios.values())
        split_tensors = torch.split(tensor, split_indices, dim=1)
        return split_tensors
    
    def save_model(self, model, model_name) -> None:
        torch.save(model.state_dict(), model_name)
    
    def reset_iterloader(self) -> None:
        self.iterloader = iter(self.data_loader)

    def get_batch_from_dataloader(self) -> torch.Tensor:
        try:
            data = next(self.iterloader)
        except StopIteration:
            self.reset_iterloader()
            data = next(self.iterloader)
            logger.info("StopIteration: iterloader is reset and new batch is loaded")
        return data

    def forward_pass_finetuning(
        self, 
        clients_offline: list,
        simulate_client_failure: bool, 
        last_batch: bool,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function performs a forward pass of the TabNet encoder
        """
        self.partial_encoder.train()
        self.final_fc_mapping.train()

        intermediate_results_encoder_dict = {}
        future_object_encoder_dict = {}

        # get output from local models
        for client_rref in self.client_rrefs:
            if self.handle_client_failures_finetuning:
                if client_rref in clients_offline and simulate_client_failure:
                    # skipping corresponding batch of the skipped client to keep batch size consistent
                    client_rref.rpc_sync().get_batch_data()
                    continue
            future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder(is_pretraining=False)

        for client_idx, client_rref in enumerate(self.client_rrefs):
            if self.handle_client_failures_finetuning:
                if client_rref in clients_offline and simulate_client_failure:
                    if self.use_caching_method:
                        tensor_offline_client = self.client_to_tensor_batches[client_rref][batch_idx].detach().clone()
                    else:
                        if last_batch:
                            last_batch_size = self.y_train_len % self.batch_size
                            tensor_offline_client = torch.zeros(
                                (last_batch_size, self.output_encoder_clients_dims_list[client_idx]),
                                requires_grad=True
                            )
                        else:
                            tensor_offline_client = torch.zeros(
                                (self.batch_size, self.output_encoder_clients_dims_list[client_idx]),
                                requires_grad=True
                            )
                    intermediate_results_encoder_dict[client_rref] = tensor_offline_client
                    continue

            # wait for each future object to be ready
            client_inter_output_logits = future_object_encoder_dict[client_rref].wait()

            # append the intermediate value to the list
            intermediate_results_encoder_dict[client_rref] = client_inter_output_logits

            # cache intermediate result of current online client 
            # to reuse when it goes offline
            if self.use_caching_method and self.handle_client_failures_finetuning:
                # replace existing value with new batch values
                if 0 <= batch_idx < len(self.client_to_tensor_batches[client_rref]):
                    self.client_to_tensor_batches[client_rref][batch_idx] = client_inter_output_logits.detach().clone()
                else: # insert new batch values since they don't exist at batch_idx yet
                    self.client_to_tensor_batches[client_rref].insert(batch_idx, client_inter_output_logits.detach().clone())

        # concatenate the intermediate tensor values from all clients by column, 
        # e.g. cat([[1,2,3]], [[4,5,6]]) -> [[1,2,3,4,5,6]]
        input_intermediate_logits = torch.cat(list(intermediate_results_encoder_dict.values()), dim = 1).to(self.device)

        # forward pass of TabNet partial models
        encoder_output_steps, M_loss = self.partial_encoder(input_intermediate_logits)
        
        # aggregate encoder output from all clients to one latent vector of
        # size (batch_size, n_d)
        res = torch.sum(torch.stack(encoder_output_steps, dim=0), dim=0)

        logits = self.final_fc_mapping(res)

        return logits, M_loss

    def forward_pass_encoded_data(
        self,
        fetch_encoded_test_data=False,
        fetch_encoded_valid_data=False
    ) -> np.ndarray:
        """
        This function performs a forward pass of the TabNet encoder
        """
        self.partial_encoder.eval()

        assert self.partial_encoder.training == False, "partial_encoder is not in training mode"
        
        intermediate_results_encoder_dict = {}
        future_object_encoder_dict = {}

        # get output from local encoders
        for client_rref in self.client_rrefs:
            if fetch_encoded_test_data:
                future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder_test_data()
            elif fetch_encoded_valid_data:
                future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder_valid_data()
            else:
                future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder_train_data()
            
        for client_rref in self.client_rrefs:
            # wait for each future object to be ready
            client_inter_output_logits = future_object_encoder_dict[client_rref].wait()
            # append the intermediate value to the list
            intermediate_results_encoder_dict[client_rref] = client_inter_output_logits

        # concatenate the intermediate tensor values from all clients by column, 
        # e.g. cat([[1,2,3]], [[4,5,6]]) -> [[1,2,3,4,5,6]]
        input_intermediate_logits = torch.cat(list(intermediate_results_encoder_dict.values()), dim = 1).to(self.device)

        # forward pass of TabNet partial models
        with torch.no_grad():
            encoder_output_steps, _ = self.partial_encoder(input_intermediate_logits)
        
        # aggregate encoder output from all clients to one latent vector of
        # size (batch_size, n_d)
        encoded_representation = torch.sum(torch.stack(encoder_output_steps, dim=0), dim=0)

        return encoded_representation.cpu().numpy()


    def forward_pass_pretraining(
        self, 
        clients_offline: list,
        simulate_client_failure: bool, 
        last_batch: bool,
        batch_idx: int
    ) -> torch.Tensor:
        """
        This function performs a forward pass of the TabNet encoder and decoder
        which are used during pretrianing
        """
        self.partial_encoder.train()
        self.partial_decoder.train()

        assert self.partial_encoder.training == True, "partial_encoder is not in training mode"
        assert self.partial_decoder.training == True, "partial_decoder is not in training mode"

        intermediate_results_encoder_dict = {}
        results_prior_encoder_dict = {}
        future_object_encoder_dict = {}

        # get output from local encoders
        for client_rref in self.client_rrefs:
            if self.handle_client_failures_pretraining:
                if client_rref in clients_offline and simulate_client_failure:
                    # skipping corresponding batch of the skipped client to keep batch size consistent
                    client_rref.rpc_sync().get_batch_data()
                    continue
            future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder(is_pretraining=True)

        for client_idx, client_rref in enumerate(self.client_rrefs):
            if self.handle_client_failures_pretraining:
                if client_rref in clients_offline and simulate_client_failure:
                    if self.use_caching_method:
                        intermediate_results_failed_client = self.client_to_tensor_batches_pretrain[client_rref][batch_idx].detach().clone()
                    else:
                        if last_batch:
                            last_batch_size = self.y_train_len % self.batch_size
                            intermediate_results_failed_client = torch.zeros(
                                (last_batch_size, self.output_encoder_clients_dims_list[client_idx]),
                                requires_grad=True
                            )
                        else:
                            intermediate_results_failed_client = torch.zeros(
                                (self.batch_size, self.output_encoder_clients_dims_list[client_idx]),
                                requires_grad=True
                            )

                    intermediate_results_encoder_dict[client_rref] = intermediate_results_failed_client

                    if self.use_caching_method:
                        prior_failed_client = self.client_to_tensor_masks_pretrain[client_rref][batch_idx].detach().clone()
                    else:
                        if last_batch:
                            last_batch_size = self.y_train_len % self.batch_size
                            prior_failed_client = torch.zeros(
                                (last_batch_size, self.output_encoder_clients_dims_list[client_idx]),
                                requires_grad=True
                            )
                        else:
                            prior_failed_client = torch.zeros(
                                (self.batch_size, self.output_encoder_clients_dims_list[client_idx]),
                                requires_grad=True
                            )
                    results_prior_encoder_dict[client_rref] = prior_failed_client

                    continue

            # wait for each future object to be ready
            client_inter_output_logits, prior = future_object_encoder_dict[client_rref].wait()

            # append the intermediate value to the list
            intermediate_results_encoder_dict[client_rref] = client_inter_output_logits
            results_prior_encoder_dict[client_rref] = prior

            # cache prior and intermediate results of current client
            if self.use_caching_method and self.handle_client_failures_pretraining:
                # replace existing value with new batch values
                if 0 <= batch_idx < len(self.client_to_tensor_batches_pretrain[client_rref]):
                    self.client_to_tensor_batches_pretrain[client_rref][batch_idx] = client_inter_output_logits.detach().clone() 
                else: # insert new batch values since they don't exist at batch_idx yet
                    self.client_to_tensor_batches_pretrain[client_rref].insert(batch_idx, client_inter_output_logits.detach().clone())
                
                if 0 <= batch_idx < len(self.client_to_tensor_masks_pretrain[client_rref]):
                    self.client_to_tensor_masks_pretrain[client_rref][batch_idx] = prior.detach().clone() 
                else: 
                    self.client_to_tensor_masks_pretrain[client_rref].insert(batch_idx, prior.detach().clone())

        # concatenate the intermediate tensor values from all clients by column, 
        # e.g. cat([[1,2,3]], [[4,5,6]]) -> [[1,2,3,4,5,6]]
        input_intermediate_logits = torch.cat(list(intermediate_results_encoder_dict.values()), dim = 1).to(self.device)
        prior_concated = torch.cat(list(results_prior_encoder_dict.values()), dim = 1).to(self.device)

        # forward pass of TabNet partial models 
        encoder_output_steps, _ = self.partial_encoder(input_intermediate_logits, prior=prior_concated)
        decoder_output = self.partial_decoder(encoder_output_steps)

        # split decoder output for each client using the split information during init
        splitted_tensors = self.split_tensor_cols(decoder_output, self.local_decoder_split_ratios)

        total_loss_clients = 0
        decoder_res_client_rref_to_future_obj = {}

        # asynchronously forward pass decoder of all clients
        for client_rref, client_tensor in zip(self.client_rrefs, splitted_tensors):
            if self.handle_client_failures_pretraining:
                if client_rref in clients_offline and simulate_client_failure:
                    continue
            if self.use_cuda:
                client_tensor = client_tensor.cpu()
            decoder_loss_future = client_rref.rpc_async().forward_pass_decoder(client_tensor)
            decoder_res_client_rref_to_future_obj[client_rref] = decoder_loss_future

        # aggregate loss values from clients 
        # wait for each future object to be ready
        for client_rref in self.client_rrefs:
            if self.handle_client_failures_pretraining:
                if client_rref in clients_offline and simulate_client_failure:
                    continue
            decoder_loss_future = decoder_res_client_rref_to_future_obj[client_rref]
            total_loss_clients += decoder_loss_future.wait()
        
        return total_loss_clients

    def evaluate(self):
        """
        This function evaluates the model on the test data
        """
        hyperparams_log: Dict[str, Any] = {
            "pretrain_epochs": self.pretrain_epochs,
            "finetune_epochs": self.finetune_epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            **self.tabnet_hyperparams,
            "optimizer": self.optimizer,
            **self.optimizer_params,
            "patience_pretraining": self.patience_pretraining,
            "patience_finetuning": self.patience_finetuning,
            "epoch_failure_probability": self.epoch_failure_probability,
            "use_caching_client_failure_method": self.use_caching_method,
            "mini_batch_client_failure": self.mini_batch_client_failure
        }

        evaluate_multiple_predictors(
            predictors=self.predictors, 
            hyperparams=hyperparams_log,
            experiment_name="tabnet_vfl",
            logger=None, 
            eval_results_path=self.eval_results_path, 
            task_type=self.task_type, 
            X_test=self.X_test_downstream, 
            y_test=self.y_test_downstream,
        )

    def fit_predictor(self):
        logger.info(f"{self.y_train=}")

        curr_clients_offline = []

        X_train_latent = self.forward_pass_encoded_data(
            fetch_encoded_test_data=False,
            fetch_encoded_valid_data=False
        )
        logger.info(f"{X_train_latent=}")
        logger.info(f"{X_train_latent.shape=}")
        X_valid_latent = self.forward_pass_encoded_data(
            fetch_encoded_test_data=False,
            fetch_encoded_valid_data=True
        )
        logger.info(f"{X_valid_latent=}")
        logger.info(f"{X_valid_latent.shape=}")
        X_test_latent = self.forward_pass_encoded_data(
            fetch_encoded_test_data=True,
            fetch_encoded_valid_data=False
        )
        logger.info(f"{X_test_latent=}")
        logger.info(f"{X_test_latent.shape=}")

        # concatenate all X datasets to one array
        X_total_dataset = np.concatenate((X_train_latent, X_valid_latent, X_test_latent), axis=0)
        logger.info(f"{X_total_dataset.shape=}")
        # concatenate all labels to one array
        y_total_dataset = np.concatenate((self.y_train, self.y_valid, self.y_test), axis=0)
        logger.info(f"{y_total_dataset.shape=}")

        # split the dataset into train and test sets
        if self.task_type == "continuous":
            X_train_downstream, X_test_downstream, y_train_downstream, y_test_downstream = train_test_split(
                X_total_dataset,
                y_total_dataset,
                test_size=(self.test_ratio + self.valid_ratio),
                random_state=42,
                shuffle=True,
            )
        else:
            X_train_downstream, X_test_downstream, y_train_downstream, y_test_downstream = train_test_split(
                X_total_dataset,
                y_total_dataset,
                test_size=(self.test_ratio + self.valid_ratio),
                random_state=42,
                shuffle=True,
                stratify=y_total_dataset,
            )
        logger.info(f"{X_train_downstream=}")
        logger.info(f"{X_train_downstream.shape=}")
        logger.info(f"{X_test_downstream=}")
        logger.info(f"{X_test_downstream.shape=}")
        logger.info(f"{y_train_downstream=}")
        logger.info(f"{y_train_downstream.shape=}")
        logger.info(f"{y_test_downstream=}")
        logger.info(f"{y_test_downstream.shape=}")

        self.X_test_downstream = X_test_downstream
        self.y_test_downstream = y_test_downstream

        # train the predictors
        for predictor in self.predictors:
            logger.info(f"Fitting predictor: {predictor}...")
            predictor.fit(X_train_downstream, y_train_downstream) # type: ignore
            logger.info(f"Predictor {predictor} fitted!")
        
        return X_train_downstream, y_train_downstream
    
    def fit_finetuning(self):
        """
        This function fits the model in a supervised manner
        """
        # number of batches in one epoch, under uneven division the ceil value is taken
        # to complete the last batch
        num_batches = max(int(np.ceil(self.y_train_len / self.batch_size)), 1)

        best_val_loss = float('inf')
        if self.task_type != "continuous":
            best_val_loss = -best_val_loss # maximize validation metric in case of multiclass/binary classification
        current_patience = 0

        # list to keep track of losses per epoch
        list_of_train_losses_per_epoch = []
        # list to keep track of training time per epoch
        list_of_training_times_per_epoch = []
        # list of how many times clients were offline during training
        dict_count_offline_clients = {}
        # list of valid loss per epoch
        list_of_valid_losses_per_epoch = []

        # reset iterloaders so that we start from the beginning of the dataset
        self.reset_iterloader()
        for client_rref in self.client_rrefs:
            client_rref.rpc_sync().reset_iterloader()

        training_start_time = time.time()
        for epoch in range(1, self.finetune_epochs+1):
            logger.info(f"Epoch: {epoch}")

            if not self.mini_batch_client_failure:
                curr_clients_offline = []
                if self.handle_client_failures_finetuning:
                    # loop through each client and decide whether each of them should go
                    # offline or not
                    # DO NOT SIMULATE CLIENT FAILURES FROM THE FIRST EPOCH
                    # WE NEED TO POPULATE THE CACHED STORAGE OF BATCHES SO WE
                    # CAN USE IT. ASSUME NONE OF CLIENT WILL GO OFFLINE AT THE START
                    if epoch != 1:
                        for curr_client in self.client_rrefs:
                            random_sampled_num = self.rng_failure.random()
                            if random_sampled_num < self.epoch_failure_probability:
                                curr_clients_offline.append(curr_client)
                        str_of_clients_offline = "(" + ", ".join(
                            [str(client_rref.owner().id) for client_rref in curr_clients_offline]
                        ) + ")"
                        if dict_count_offline_clients.get(str_of_clients_offline, None) is None: 
                            dict_count_offline_clients[str_of_clients_offline] = 0
                        else:
                            dict_count_offline_clients[str_of_clients_offline] += 1
                        print(f"{str_of_clients_offline=}")

            losses_sum = 0

            epoch_start_time = time.perf_counter()
            for batch_amount in range(1, num_batches+1):

                # simulating client failure here (mini-batch level)
                if self.mini_batch_client_failure:
                    curr_clients_offline = []
                    if self.handle_client_failures_finetuning:
                        # loop through each client and decide whether each of them should go
                        # offline or not
                        # DO NOT SIMULATE CLIENT FAILURES FROM THE FIRST EPOCH
                        # WE NEED TO POPULATE THE CACHED STORAGE OF BATCHES SO WE
                        # CAN REUSE IT. ASSUME NONE OF CLIENT WILL GO OFFLINE AT THE START
                        if epoch != 1:
                            for curr_client in self.client_rrefs:
                                random_sampled_num = self.rng_failure.random()
                                if random_sampled_num < self.epoch_failure_probability:
                                    curr_clients_offline.append(curr_client)
                            str_of_clients_offline = "(" + ", ".join(
                                [str(client_rref.owner().id) for client_rref in curr_clients_offline]
                            ) + ")"
                            if dict_count_offline_clients.get(str_of_clients_offline, None) is None: 
                                dict_count_offline_clients[str_of_clients_offline] = 0
                            else:
                                dict_count_offline_clients[str_of_clients_offline] += 1
                            print(f"{str_of_clients_offline=}")

                with dist.autograd.context() as tabnet_finetuning_context:
                    y_target = self.get_batch_from_dataloader()
                    y_target = y_target.to(self.device)
                    y_target = y_target.view(-1)

                    if self.task_type == "continuous":
                        y_target = y_target.float()

                    # indicate whether this is the last batch of the epoch
                    last_batch = batch_amount == num_batches

                    # forward pass of clients and server for pretraining
                    # y_pred should be a tensor of shape (batch_size, 2)
                    # in binary case
                    if self.handle_client_failures_finetuning:
                        logits, M_loss = self.forward_pass_finetuning(
                            curr_clients_offline,
                            True, 
                            last_batch,
                            batch_idx=batch_amount-1
                        )
                    else:
                        logits, M_loss = self.forward_pass_finetuning(
                            [],
                            False, 
                            last_batch,
                            batch_idx=batch_amount-1
                        )

                    # calculate the loss of the forawrd pass
                    # in case of regression, logits represents one value for each sample
                    # calculate the loss of the forawrd pass
                    if self.task_type == "continuous":
                        # Convert logits to probabilities using softmax
                        probs = torch.softmax(logits, dim=1)
                        loss = self.criteron(probs, y_target)
                    else:
                        loss = self.criteron(logits, y_target)

                    # Add the overall sparsity loss
                    loss = loss - self.lambda_sparse * M_loss

                    losses_sum += loss.item()

                    dist.autograd.backward(tabnet_finetuning_context, [loss])
                    
                    self.tabnet_finetuning_opt.step(tabnet_finetuning_context)

            epoch_end_time = time.perf_counter()
            logger.info(f"Epoch {epoch} training time: {epoch_end_time - epoch_start_time}")
            list_of_training_times_per_epoch.append(epoch_end_time - epoch_start_time)

            if self.task_type == "continuous":
                logger.info(f"MSE LOSS: {losses_sum / num_batches}")
            else:
                logger.info(f"CROSS ENTROPY LOSS: {losses_sum / num_batches}")

            # append the average loss per epoch to the list
            list_of_train_losses_per_epoch.append(losses_sum / num_batches)

            # evaluate the model on the validation data 
            if self.handle_client_failures_finetuning:
                valid_loss, stop_training, best_val_loss, current_patience = self.validate_model(
                    is_pretraining=False,
                    best_val_loss=best_val_loss,
                    current_patience=current_patience,
                )
            else:
                valid_loss, stop_training, best_val_loss, current_patience = self.validate_model(
                    is_pretraining=False,
                    best_val_loss=best_val_loss,
                    current_patience=current_patience,
                )

            list_of_valid_losses_per_epoch.append(valid_loss)

            logger.info(f"{best_val_loss=}")
            logger.info(f"{current_patience=}")

            assert self.partial_encoder.training == True, "partial_encoder is not in training mode after validation"
            assert self.final_fc_mapping.training == True, "partial_decoder is not in training mode after validation"

        training_end_time = time.time()
        logger.info(f"Finetuning done in: {training_end_time - training_start_time}")

        return list_of_train_losses_per_epoch, list_of_valid_losses_per_epoch, list_of_training_times_per_epoch, dict_count_offline_clients
    
    def forward_pass_valid_data_finetuning(self):
        """
        This function performs a forward pass of the TabNet encoder
        """
        assert self.partial_encoder.training == False, "partial_encoder is not in eval mode"
        assert self.final_fc_mapping.training == False, "final_fc_mapping is not in eval mode"

        intermediate_results_encoder_dict = {}
        future_object_encoder_dict = {}

        # get output from local models
        for client_rref in self.client_rrefs:
            future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder_valid(is_pretraining=False)

        for client_idx, client_rref in enumerate(self.client_rrefs):
            # wait for each future object to be ready
            client_inter_output_logits = future_object_encoder_dict[client_rref].wait()

            # append the intermediate value to the list
            intermediate_results_encoder_dict[client_rref] = client_inter_output_logits

        # concatenate the intermediate tensor values from all clients by column, 
        # e.g. cat([[1,2,3]], [[4,5,6]]) -> [[1,2,3,4,5,6]]
        input_intermediate_logits = torch.cat(list(intermediate_results_encoder_dict.values()), dim = 1).to(self.device)

        # forward pass of TabNet partial models 
        encoder_output_steps, M_loss = self.partial_encoder(input_intermediate_logits)
        
        # aggregate encoder output from all clients to one latent vector of
        # size (batch_size, n_d)
        res = torch.sum(torch.stack(encoder_output_steps, dim=0), dim=0)

        logits = self.final_fc_mapping(res)

        return logits, M_loss

    def forward_pass_valid_data_pretraining(self) -> torch.Tensor:
        assert self.partial_encoder.training == False, "partial_encoder is not in training mode"
        assert self.partial_decoder.training == False, "partial_decoder is not in training mode"

        intermediate_results_encoder_dict = {}
        results_prior_encoder_dict = {}
        future_object_encoder_dict = {}

        # get output from local encoders
        for client_rref in self.client_rrefs:
            future_object_encoder_dict[client_rref] = client_rref.rpc_async().forward_pass_encoder_valid(is_pretraining=True)

        for client_idx, client_rref in enumerate(self.client_rrefs):
            # wait for each future object to be ready
            client_inter_output_logits, prior = future_object_encoder_dict[client_rref].wait()

            # append the intermediate value to the list
            intermediate_results_encoder_dict[client_rref] = client_inter_output_logits
            results_prior_encoder_dict[client_rref] = prior

        # concatenate the intermediate tensor values from all clients by column, 
        # e.g. cat([[1,2,3]], [[4,5,6]]) -> [[1,2,3,4,5,6]]
        input_intermediate_logits = torch.cat(list(intermediate_results_encoder_dict.values()), dim = 1).to(self.device)
        prior_concated = torch.cat(list(results_prior_encoder_dict.values()), dim = 1).to(self.device)

        # forward pass of TabNet partial models 
        encoder_output_steps, _ = self.partial_encoder(input_intermediate_logits, prior=prior_concated)
        decoder_output = self.partial_decoder(encoder_output_steps)

        # split decoder output for each client using the split information during init
        splitted_tensors = self.split_tensor_cols(decoder_output, self.local_decoder_split_ratios)

        total_loss_clients = 0
        decoder_res_client_rref_to_future_obj = {}

        # asynchronously forward pass decoder of all clients
        for client_rref, client_tensor in zip(self.client_rrefs, splitted_tensors):
            if self.use_cuda:
                client_tensor = client_tensor.cpu()
            decoder_loss_future = client_rref.rpc_async().forward_pass_decoder_valid(client_tensor)
            decoder_res_client_rref_to_future_obj[client_rref] = decoder_loss_future

        # aggregate loss values from clients 
        # wait for each future object to be ready
        for client_rref in self.client_rrefs:
            decoder_loss_future = decoder_res_client_rref_to_future_obj[client_rref]
            total_loss_clients += decoder_loss_future.wait()
        
        return total_loss_clients


    def validate_model(
        self, 
        is_pretraining: bool,
        best_val_loss: float,
        current_patience: int,
    ) -> Union[torch.Tensor, bool, float, int]:
        if is_pretraining:
            self.partial_encoder.eval()
            self.partial_decoder.eval()
        else:
            self.partial_encoder.eval()
            self.final_fc_mapping.eval()
        
        # lst_valid_losses = []
        stop_training = False
        is_maximize = False

        with torch.no_grad():
            if is_pretraining:
                valid_loss = self.forward_pass_valid_data_pretraining()
                valid_loss = valid_loss.item()
            else:
                logits, _ = self.forward_pass_valid_data_finetuning()
                softmaxed_logits = torch.softmax(logits, axis=1).cpu().detach().numpy()

                # logits is a tensor of shape (batch_size, 2) representing the classification prediction
                # in case of regression, logits represents one output value for each sample
                if self.task_type == "continuous":
                    valid_loss = mean_squared_error(self.y_valid, logits.cpu().detach().numpy())
                    logger.info(f"FINETUNING VALID MSE LOSS: {valid_loss}")
                    is_maximize = False
                else:
                    is_maximize = True
                    preds = np.argmax(softmaxed_logits, axis=1)
                    if self.unique_class_labels_count == 2:
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true=self.y_valid, 
                            y_pred=preds, 
                            average='weighted'
                        )
                        valid_loss = f1
                    else:
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true=self.y_valid, 
                            y_pred=preds, 
                            average='weighted'
                        )
                        valid_loss = f1
                    logger.info(f"FINETUNING VALID F1_SCORE: {valid_loss}")

        loss_change = valid_loss - best_val_loss
        max_improved = is_maximize and loss_change > 0.0
        min_improved = (not is_maximize) and (-loss_change > 0.0)

        # Check for early stopping
        if max_improved or min_improved:
            best_val_loss = valid_loss
            current_patience = 0
        else:
            if is_pretraining:
                if current_patience >= self.patience_pretraining:
                    print(f"Validation loss hasn't improved for {self.patience_pretraining=} epochs. Stopping training.")
                    stop_training = True
            else:
                if current_patience >= self.patience_finetuning:
                    print(f"Validation loss hasn't improved for {self.patience_finetuning=} epochs. Stopping training.")
                    stop_training = True
            current_patience += 1

        # reset to train mode
        if is_pretraining:
            self.partial_encoder.train()
            self.partial_decoder.train()
        else:
            self.partial_encoder.train()
            self.final_fc_mapping.train()

        return valid_loss, stop_training, best_val_loss, current_patience

    def fit_pretraining(self):
        """
        E: the interval epochs to swap models
        """
        # number of batches in one epoch,
        # e.g. 1001 training samples, batch size 100 -> 11 batches
        # server does not have data, so this is used only for training loop.
        # stopping criteria reached when the final batch is done
        num_batches = max(int(np.ceil(self.y_train_len / self.batch_size)), 1)

        best_val_loss = float('inf')
        current_patience = 0

        logger.info(f"Initial best val loss: {best_val_loss}")
        logger.info(f"Initial current patience: {current_patience}")

        # list to keep track of losses per epoch
        list_of_train_losses_per_epoch = []

        # list to keep track of training time per epoch
        list_of_training_times_per_epoch = []

        # list of valid loses per epoch
        list_of_valid_losses_per_epoch = []

        # dict of how many times clients were offline during training
        dict_count_offline_clients = {}

        # reset iterloaders so that we start from the beginning of the dataset
        # as we will start a new epoch so start from the first batch
        self.reset_iterloader()
        for client_rref in self.client_rrefs:
            client_rref.rpc_sync().reset_iterloader()

        training_start_time = time.time()
        for epoch in range(1, self.pretrain_epochs+1):
            logger.info(f"Epoch: {epoch}")

            losses_sum = 0

            if not self.mini_batch_client_failure:
                curr_clients_offline = []
                if self.handle_client_failures_pretraining:
                    # loop through each client and decide whether each of them should go
                    # offline or not
                    # DO NOT SIMULATE CLIENT FAILURES FROM THE FIRST EPOCH
                    # WE NEED TO POPULATE THE CACHED STORAGE OF BATCHES SO WE
                    # CAN USE IT. ASSUME NONE OF CLIENT WILL GO OFFLINE AT THE START
                    if epoch != 1:
                        for curr_client in self.client_rrefs:
                            random_sampled_num = self.rng_failure.random()
                            if random_sampled_num < self.epoch_failure_probability:
                                curr_clients_offline.append(curr_client)
                        str_of_clients_offline = "(" + ", ".join(
                            [str(client_rref.owner().id) for client_rref in curr_clients_offline]
                        ) + ")"
                        if dict_count_offline_clients.get(str_of_clients_offline, None) is None: 
                            dict_count_offline_clients[str_of_clients_offline] = 0
                        else:
                            dict_count_offline_clients[str_of_clients_offline] += 1
                        print(f"{str_of_clients_offline=}")
                
            epoch_start_time = time.perf_counter()
            for batch_amount in range(1, num_batches+1):
                # simulating client failure here (mini-batch level)
                if self.mini_batch_client_failure:
                    curr_clients_offline = []
                    if self.handle_client_failures_pretraining:
                        # loop through each client and decide whether each of them should go
                        # offline or not
                        # DO NOT SIMULATE CLIENT FAILURES FROM THE FIRST EPOCH
                        # WE NEED TO POPULATE THE CACHED STORAGE OF BATCHES SO WE
                        # CAN USE IT. ASSUME NONE OF CLIENT WILL GO OFFLINE AT THE START
                        if epoch != 1:
                            for curr_client in self.client_rrefs:
                                random_sampled_num = self.rng_failure.random()
                                if random_sampled_num < self.epoch_failure_probability:
                                    curr_clients_offline.append(curr_client)
                            str_of_clients_offline = "(" + ", ".join(
                                [str(client_rref.owner().id) for client_rref in curr_clients_offline]
                            ) + ")"
                            if dict_count_offline_clients.get(str_of_clients_offline, None) is None: 
                                dict_count_offline_clients[str_of_clients_offline] = 0
                            else:
                                dict_count_offline_clients[str_of_clients_offline] += 1
                            print(f"{str_of_clients_offline=}")

                with dist.autograd.context() as pretrain_tabnet_context:
                    # indicate whether this is the last batch of the epoch
                    last_batch = batch_amount == num_batches

                    # forward pass of clients and server for pretraining
                    if self.handle_client_failures_pretraining:
                        loss = self.forward_pass_pretraining(
                            curr_clients_offline, 
                            True,
                            last_batch,
                            batch_idx=batch_amount
                        )
                    else:
                        loss = self.forward_pass_pretraining(
                            [], 
                            False,
                            last_batch,
                            batch_idx=batch_amount
                        )

                    # if all clients are chosen to be offline, then skip and continue to the next batch
                    if len(curr_clients_offline) == len(self.client_rrefs):
                        continue
                    
                    dist.autograd.backward(pretrain_tabnet_context, [loss])

                    self.tabnet_pretraining_opt.step(pretrain_tabnet_context)
                    
                    losses_sum += loss.item()
                    
            epoch_end_time = time.perf_counter()
            logger.info(f"Epoch {epoch} training time: {epoch_end_time - epoch_start_time}")
            list_of_training_times_per_epoch.append(epoch_end_time - epoch_start_time)

            logger.info(f"PRETRAINER LOSS: {losses_sum / num_batches}")

            # append the average loss per epoch to the list
            list_of_train_losses_per_epoch.append(losses_sum / num_batches)  

            # evaluate the model on the validation data 
            if self.handle_client_failures_pretraining:
                valid_loss, stop_training, best_val_loss, current_patience = self.validate_model(
                    is_pretraining=True,
                    best_val_loss=best_val_loss,
                    current_patience=current_patience,
                )
            else:
                valid_loss, stop_training, best_val_loss, current_patience = self.validate_model(
                    is_pretraining=True,
                    best_val_loss=best_val_loss,
                    current_patience=current_patience,
                )

            list_of_valid_losses_per_epoch.append(valid_loss)

            logger.info(f"{best_val_loss=}")
            logger.info(f"{current_patience=}")

            assert self.partial_encoder.training == True, "partial_encoder is not in training mode after validation"
            assert self.partial_decoder.training == True, "partial_decoder is not in training mode after validation"

            logger.info(f"PRETRAINING VALIDATION UNSUP_LOSS: {valid_loss}")

        training_end_time = time.time()
        logger.info(f"Pretraining done in: {training_end_time - training_start_time}")

        return list_of_train_losses_per_epoch, list_of_valid_losses_per_epoch, list_of_training_times_per_epoch, dict_count_offline_clients


def run(
    rank,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    X_train_per_client: List[Tuple[Dict[str, Union[List[str], Dict[str, float]]], pd.DataFrame]],
    splitted_X_test: List[pd.DataFrame],
    splitted_X_valid: pd.DataFrame,
    task_type: str,
    master_addr: str, 
    master_port: int, 
    num_clients: int, 
    use_cuda: bool,
    epoch_failure_probability: float,
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
    decoder_split_ratios: list,
    optimizer: str,
    optimizer_params: dict,
):
    try:
        # set environment information
        world_size = num_clients + 1
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        logger.info(f"world size: {world_size}")
        logger.info(f"IP: tcp://{master_addr}:{master_port}")
        logger.info(f"Rank: {rank}")

        if rank == 0:  # this is run only on the server side
            rpc.init_rpc(
                "server",
                rank=rank,
                world_size=world_size,
                backend=rpc.BackendType.PROCESS_GROUP,
                rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                    num_send_recv_threads=8, rpc_timeout=99999, init_method=f"tcp://{master_addr}:{master_port}"
                ),
            )

            clients = []
            for worker_idx, (cols_info_dict, X_train_df) in enumerate(X_train_per_client):
                clients.append(
                    rpc.remote(
                        "client" + str(worker_idx + 1), 
                        TabNetClient, 
                        kwargs=dict(
                            X_train_df=X_train_df,
                            X_test_df=splitted_X_test[worker_idx],
                            X_valid_df=splitted_X_valid[worker_idx],
                            client_id=worker_idx + 1,
                            epochs=epochs, 
                            cols_info_dict=cols_info_dict,
                            use_cuda=use_cuda, 
                            batch_size=batch_size,
                            test_ratio=test_ratio,
                            seed=seed,
                            pretraining_ratio=tabnet_hyperparams["pretraining_ratio"],
                        )
                    )
                )
                logger.info(f"register remote client-{str(worker_idx+1)} - {clients[worker_idx]=}")

            
            server = TabNetServer(
                y_train=y_train,
                y_valid=y_valid,
                y_test=y_test,
                predictors=predictors,
                client_rrefs=clients,
                task_type=task_type,
                seed=seed,
                epoch_failure_probability=epoch_failure_probability,
                column_label_name=column_label_name,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
                eval_out=eval_out,
                tabnet_hyperparams=tabnet_hyperparams,
                decoder_split_ratios=decoder_split_ratios,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                batch_size=batch_size,
                epochs=epochs,
                use_cuda=use_cuda,
            )
            pretraining_train_losses, pretraining_valid_losses, epoch_time_values_pretraining, dict_counts_offline_clients_pretraining = \
                server.fit_pretraining()
            finetuning_train_losses, finetuning_valid_losses, epoch_time_values_finetuning, dict_counts_offline_clients_finetuning = \
                server.fit_finetuning()
            X_train_latent, y_train = server.fit_predictor()
            server.evaluate()

            # plot histogram offline client counts
            histogram_client_offline_counts(dict_counts_offline_clients_pretraining, "pretraining")

            # plot histogram offline client counts
            histogram_client_offline_counts(dict_counts_offline_clients_finetuning, "finetuning")

            # save epoch time values of pretraining
            save_epoch_times_and_plot(
                epoch_time_values_pretraining, 
                "pretraining",
            )

            # save epoch time values of finetuning
            save_epoch_times_and_plot(
                epoch_time_values_finetuning, 
                "finetuning",
            )

            # for testing whether the training loss is decreasing and converging to a certain value 
            save_losses_plot(
                finetuning_train_losses, 
                finetuning_valid_losses, 
                "finetuning", 
                "Train and Valid losses during finetuning TabNet VFL"
            )

            # for testing whether the training loss is decreasing and converging to a certain value 
            save_losses_plot(
                pretraining_train_losses, 
                pretraining_valid_losses, 
                "pretraining", 
                "Train and Valid Losses during pretraining TabNet VFL"
            )
            
            # plot tsne of the train latent data
            # plot_tsne_of_latent(X_train_latent, y_train, "TabNet VFL", SCRIPT_DIR)
        
        elif rank != 0:
            raise ValueError("Only rank 0 is allowed to run the server")
    except Exception as e:
        logger.exception(e)
    finally:
        rpc.shutdown()

def run_from_cmd(
    rank,
    world_size, 
    ip, 
    port, 
    dataset, 
    epochs, 
    use_cuda, 
    batch_size,
):
    try:
        # set environment information
        os.environ["MASTER_ADDR"] = ip
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        logger.info(f"number of epochs before initialization: {epochs}")
        logger.info(f"world size: {world_size}")
        logger.info(f"IP: tcp://{ip}:{port}")
        logger.info(f"Rank: {rank}")

        if rank == 0:  # this is run only on the server side
            rpc.init_rpc(
                "server",
                rank=rank,
                world_size=world_size,
                backend=rpc.BackendType.PROCESS_GROUP,
                rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                    num_send_recv_threads=8, rpc_timeout=120, init_method=f"tcp://{ip}:{port}"
                ),
            )

            clients = []
            for worker in range(world_size-1):
                clients.append(
                    rpc.remote(
                        "client"+str(worker+1), 
                        TabNetClient, 
                        kwargs=dict(
                            dataset=dataset, 
                            epochs=epochs, 
                            use_cuda=use_cuda, 
                            batch_size=batch_size,
                            client_id=worker + 1
                        )
                    )
                )
                logger.info(f"register remote client-{str(worker+1)} - {clients[worker]=}")

            
            server = TabNetServer(
                client_rrefs=clients,
                dataset=dataset,
                epochs=epochs,
                use_cuda=use_cuda, 
                batch_size=batch_size,
            )
            server.fit_pretraining()
            server.fit_finetuning()
            server.evaluate()

        elif rank != 0:
            raise ValueError("Only rank 0 is allowed to run the server")
    except Exception as e:
        logger.exception(e)
    finally:
        rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument(
        "--dataset", type=str, default="Adult"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument('--use_cuda',  type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    if args.rank is not None:
        # run with a specified rank (need to start up another process with the opposite rank elsewhere)
        run_from_cmd(
            rank=args.rank,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            dataset=args.dataset,
            epochs=args.epochs,
            use_cuda=args.use_cuda,
            batch_size=args.batch_size,
        )
