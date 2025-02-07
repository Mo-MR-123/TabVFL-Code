import argparse
from matplotlib import pyplot as plt
import torch
import pandas as pd
import torch.distributed.rpc as rpc
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import os
from pytorch_tabnet.pretraining import TabNetPretrainer
from local_tabnets.one_client.modified_tabnet import TabNetClassifier, TabNetRegressor
import warnings
from pathlib import Path
from torchinfo import summary


from shared.general_utils import infer_optimizer

random_seed = 42

# Set the random seed for NumPy
np.random.seed(random_seed)

# Set the random seed for PyTorch CPU operations
torch.manual_seed(random_seed)

# Set the random seed for PyTorch GPU operations (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

SCRIPT_DIR = Path(__file__).resolve().parent


def save_losses_pretraining_plot(
    train_losses: list, 
    valid_losses: list,
    experiment_name: str, 
    client_id:int, 
    title: str
):
    with plt.style.context("ggplot"):
        # for testing whether the training loss is decreasing and converging to a certain value 
        fig, ax = plt.subplots(figsize=(9, 11))

        np.savez_compressed(SCRIPT_DIR / f"localtabnets_{experiment_name}_losses_client{client_id}.npz", train_losses=train_losses, valid_losses=valid_losses)

        ax.set_title(title)
        ax.plot(train_losses, label='Training Loss', color='blue')
        ax.plot(valid_losses, label='Validation Loss', color='orange')

        # Add labels and title
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        plt.legend()  # Show legend

        fig.savefig(str(SCRIPT_DIR / f"localtabnets_{experiment_name}_losses_client{client_id}_plot.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

def save_epoch_times_and_plot(values, experiment_name: str, client_id: int):
    with plt.style.context("ggplot"):
        # Save the values to a file
        np.savez_compressed(SCRIPT_DIR / f"localtabnets_{experiment_name}_epoch_times_values_client{client_id}.npz", epoch_times=values)

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
        fig.savefig(str(SCRIPT_DIR / f"localtabnets_{experiment_name}_epoch_times_client{client_id}_plot.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)


def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    # print(param_rrefs)
    return param_rrefs

class LocalTabNetClient():
    """
    This is the class that encapsulates the functions that need to be run on the client side
    Despite the name, this source code only needs to reside on the server, the real MDGANClient
    will be initialized with the code in client side via RPC() call.
    """

    def __init__(
        self, 
        X_train_df: pd.DataFrame,
        X_valid_df: pd.DataFrame,
        X_test_df: pd.DataFrame, 
        y_train_np: np.ndarray,
        y_valid_np: np.ndarray,
        task_type: str,
        client_id,
        cols_info_dict,
        tabnet_hyperparams: dict,
        tabnet_pretrainer_fit_params: dict,
        optimizer: str,
        optimizer_params: dict,
        seed,
        epochs, 
        use_cuda, 
        batch_size, 
        train_ratio,
        test_ratio,
        **kwargs
    ):
        self.client_id = client_id
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.seed = seed
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.X_train = X_train_df
        self.X_valid = X_valid_df
        self.X_test = X_test_df
        self.y_train = y_train_np
        self.y_valid = y_valid_np
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.task_type = task_type
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.tabnet_hyperparams = tabnet_hyperparams
        self.pretraining_done = False
        self.finetuning_done = False

        print(f"{client_id=} Epochs: {epochs}")
        print(f"{client_id=}, {self.seed=}")
        print(f"{client_id=}, {self.X_train=}")
        print(f"{client_id=}, {self.X_test=}")

        # convert df to numpy to prevent compatibility issues
        self.X_train = self.X_train.to_numpy()
        self.X_valid = self.X_valid.to_numpy()
        self.X_test = self.X_test.to_numpy()

        if self.task_type != "continuous":
            self.y_train = self.y_train.ravel()
            self.y_valid = self.y_valid.ravel()
        else:
            self.y_train = self.y_train
            self.y_valid = self.y_valid
            
        print(f"{client_id=}, {self.y_train=}")
        print(f"{client_id=}, {self.y_train.shape=}")
        print(f"{client_id=}, {self.y_valid=}")
        print(f"{client_id=}, {self.y_valid.shape=}")

        self.input_dim = self.X_train.shape[1]

        if self.task_type == "continuous":
            self.valid_metric_postfix = "_mse"
        else:
            if self.task_type == "multiclass":
                self.valid_metric_postfix = "_accuracy"
            else:
                self.valid_metric_postfix = "_auc"

        # Pretrainer TabNet Hyperparameters
        self.tabnet_pretrain_params = dict(
            **tabnet_hyperparams,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            seed=self.seed,
            device_name=self.device,
        )
        # print(self.tabnet_pretrain_params)

        self.pretraining_fit_params = dict(
            **tabnet_pretrainer_fit_params,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
        )
        print(self.pretraining_fit_params)

        # construct local dataloader
        self.data_loader = DataLoader(self.X_train, self.batch_size, shuffle=False)
        self.iterloader = iter(self.data_loader)

    def register_tabnet_classifier(self):
        return param_rrefs(self.local_tabnet_finetuner.network)

    def start_fit_pretraining(self):
        self.local_tabnet_pretrainer.fit(
            X_train=self.X_train,
            eval_set=[self.X_valid],
            eval_name=["valid_losses"],
            **self.pretraining_fit_params
        )

        # save epoch time values of pretraining
        save_epoch_times_and_plot(
            self.local_tabnet_pretrainer.epoch_training_times, 
            "pretraining",
            self.client_id,
        )

        # save pretraining plots and train/valid values
        save_losses_pretraining_plot(
            train_losses=self.local_tabnet_pretrainer.history["loss"], 
            valid_losses=self.local_tabnet_pretrainer.history["valid_unsup_loss_numpy"],
            experiment_name="pretraining",
            client_id=self.client_id,
            title="Local TabNet Pretraining Train + Valid Loss"
        )

        self.pretraining_done = True

        # use fit function of modified finetuner to set model of finetuner to the pretrainer model
        self.local_tabnet_finetuner.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            eval_set=[(self.X_valid, self.y_valid)],
            eval_name=["valid_losses"],
            from_unsupervised=self.local_tabnet_pretrainer,
            drop_last=self.pretraining_fit_params["drop_last"],
            virtual_batch_size=self.pretraining_fit_params["virtual_batch_size"],
            patience=self.pretraining_fit_params["patience"],
            num_workers=self.pretraining_fit_params["num_workers"],
            max_epochs=self.epochs,
            batch_size=self.batch_size,
        )

    def get_latent_dim(self):
        return self.tabnet_pretrain_params["n_d"]

    def reset_iterloader(self) -> None:
        self.iterloader = iter(self.data_loader)
    
    def get_batch_data(self) -> torch.Tensor:
        try:
            data = next(self.iterloader)
        except StopIteration:
            self.reset_iterloader()
            data = next(self.iterloader)
        return data

    def forward_pass_encoder(self):
        """This function should only be used during training for forward pass of encoder
        """
        self.local_tabnet_finetuner.network.train()
        
        assert self.local_tabnet_finetuner.network.training == True

        curr_batch_of_data = self.get_batch_data()

        curr_batch_of_data = curr_batch_of_data.to(self.device).float()
        self.curr_batch_of_data = curr_batch_of_data

        latent_rep, M_loss = self.local_tabnet_finetuner.network(curr_batch_of_data)
        if self.use_cuda:
            return latent_rep.cpu(), M_loss.cpu()
        else:
            return latent_rep, M_loss

    def set_eval_finetuner(self):
        self.local_tabnet_finetuner.network.eval()
        assert self.local_tabnet_finetuner.network.training == False, "finetuner should be in eval mode after training"
        return

    def forward_pass_encoder_valid(self):
        self.local_tabnet_finetuner.network.eval()
        
        assert self.local_tabnet_finetuner.network.training == False, "local encoder is not in eval mode"

        with torch.no_grad():
            # convert numbers to float of the current batch of data
            validation_samples = torch.from_numpy(self.X_valid).to(self.device).float()

            latent_rep, M_loss = self.local_tabnet_finetuner.network(validation_samples)

            self.local_tabnet_finetuner.network.train()

            # send intermediate output to server by converting the output to cpu first
            # since PyTorch RPC does not support CUDA tensors 
            if self.use_cuda:
                return latent_rep.cpu(), M_loss.cpu()
            else:
                return latent_rep, M_loss

    def forward_latent_data(
        self, 
        fetch_valid_data=False,
        fetch_test_data=False
    ):
        self.local_tabnet_finetuner.network.eval()

        assert self.pretraining_done is True
        assert self.local_tabnet_finetuner.network.training is False
        assert self.local_tabnet_finetuner.network.tabnet.training is False
        assert self.local_tabnet_finetuner.network.tabnet.encoder.training is False
        
        if fetch_test_data:
            print(f"{self.client_id=}, Fetch test data is True.")

            # DEBUGGING TEST
            with torch.no_grad():
                print(f"{self.X_test[0:5,:]=}")
                debug_steps_output_test, _ = self.local_tabnet_finetuner.network.tabnet.encoder(torch.from_numpy(self.X_test).to(self.device).float())
                debug_steps_summed_test = torch.sum(torch.stack(debug_steps_output_test, dim=0), dim=0)
                print(f"First test sample latent: {debug_steps_summed_test=}")
            
            with torch.no_grad():
                steps_output, _ = self.local_tabnet_finetuner.network.tabnet.encoder(torch.from_numpy(self.X_test).to(self.device).float())
                steps_summed = torch.sum(torch.stack(steps_output, dim=0), dim=0)
            
            if self.use_cuda:
                 steps_summed = steps_summed.cpu()

            assert steps_summed.requires_grad is False, "steps_summed requires grad is True during fetch test data." 
            assert steps_summed.device.type == "cpu", "steps_summed device is not cpu during fetch test data."

            return steps_summed

        elif fetch_valid_data:
            print(f"{self.client_id=}, Fetch valid data is True.")

            # DEBUGGING TEST
            with torch.no_grad():
                print(f"{self.X_valid[0:5,:]=}")
                debug_steps_output_test, _ = self.local_tabnet_finetuner.network.tabnet.encoder(torch.from_numpy(self.X_valid).to(self.device).float())
                debug_steps_summed_test = torch.sum(torch.stack(debug_steps_output_test, dim=0), dim=0)
                print(f"First test sample latent: {debug_steps_summed_test=}")
            
            with torch.no_grad():
                steps_output, _ = self.local_tabnet_finetuner.network.tabnet.encoder(torch.from_numpy(self.X_valid).to(self.device).float())
                steps_summed = torch.sum(torch.stack(steps_output, dim=0), dim=0)
            
            if self.use_cuda:
                 steps_summed = steps_summed.cpu()

            assert steps_summed.requires_grad is False, "steps_summed requires grad is True during fetch test data." 
            assert steps_summed.device.type == "cpu", "steps_summed device is not cpu during fetch test data."

            return steps_summed
        else:
            print(f"{self.client_id}, Fetch test data is False.")
            print(f"{self.client_id}, Fetch valid data is False.")

            # DEBUGGING TRAIN
            with torch.no_grad():
                print(f"{self.X_train[0:5,:]=}")
                debug_steps_output, _ = self.local_tabnet_finetuner.network.tabnet.encoder(torch.from_numpy(self.X_train).to(self.device).float())
                debug_steps_summed_train = torch.sum(torch.stack(debug_steps_output, dim=0), dim=0)
                print(f"First train sample latent: {debug_steps_summed_train=}")

            # loop through the whole data using the dataloader in batches and get the latent data
            with torch.no_grad():
                steps_output, _ = self.local_tabnet_finetuner.network.tabnet.encoder(torch.from_numpy(self.X_train).to(self.device).float())
                steps_summed = torch.sum(torch.stack(steps_output, dim=0), dim=0)
            
            if self.use_cuda:
                return steps_summed.cpu()
            
            print(f"{self.client_id}, {steps_summed[0]=}")

            assert steps_summed.requires_grad is False, "steps_summed requires grad is True during fetch train data." 
            assert steps_summed.device.type == "cpu", "steps_summed device is not cpu during fetch test data."

            return steps_summed

    def init_local_tabnet_finetuner(self):
        if self.task_type == "continuous":
            self.local_tabnet_finetuner = TabNetRegressor(
                optimizer_fn=infer_optimizer(self.optimizer),
                optimizer_params=self.optimizer_params,
                seed=self.seed,
                device_name=self.device,
                epsilon=self.tabnet_hyperparams["epsilon"],
                lambda_sparse=self.tabnet_hyperparams["lambda_sparse"],
                gamma=self.tabnet_hyperparams["gamma"],
                momentum=self.tabnet_hyperparams["momentum"],
                clip_value=self.tabnet_hyperparams["clip_value"],
                n_shared_decoder=self.tabnet_hyperparams["n_shared_decoder"], 
                n_indep_decoder=self.tabnet_hyperparams["n_indep_decoder"],
            )   
        else:
            self.local_tabnet_finetuner = TabNetClassifier(
                optimizer_fn=infer_optimizer(self.optimizer),
                optimizer_params=self.optimizer_params,
                seed=self.seed,
                device_name=self.device,
                epsilon=self.tabnet_hyperparams["epsilon"],
                lambda_sparse=self.tabnet_hyperparams["lambda_sparse"],
                gamma=self.tabnet_hyperparams["gamma"],
                momentum=self.tabnet_hyperparams["momentum"],
                clip_value=self.tabnet_hyperparams["clip_value"],
                n_shared_decoder=self.tabnet_hyperparams["n_shared_decoder"], 
                n_indep_decoder=self.tabnet_hyperparams["n_indep_decoder"],            
            )
        
        # added input for set network to work
        self.local_tabnet_finetuner.input_dim = self.X_train.shape[1]

        self.local_tabnet_finetuner._set_network()
        
        print(f"local finetuner at client {self.client_id}: initialized")
        print(f"{self.local_tabnet_finetuner=}")
        
        assert self.local_tabnet_finetuner is not None 

    def init_local_tabnet_pretrainer(self):
        # init local encoder 
        self.local_tabnet_pretrainer = TabNetPretrainer(
            input_dim=self.input_dim,
            seed=self.seed,
            optimizer_fn=infer_optimizer(self.optimizer),
            optimizer_params=self.optimizer_params,
            device_name=self.device,
            **self.tabnet_hyperparams,
        )

        print(f"local pretrainer {self.local_tabnet_pretrainer} at client {self.client_id}: initialized")
        
        assert self.local_tabnet_pretrainer is not None 


def run(
    rank, 
    master_addr, 
    master_port, 
    num_clients,  
):
    if rank == 0:
        raise ValueError("rank 0 is reserved for the server")
    
    world_size = num_clients + 1

    # set environment information
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    print("world size: ", world_size, f"tcp://{master_addr}:{master_port}")

    rpc.init_rpc(
        "client"+str(rank),
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.PROCESS_GROUP,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
            num_send_recv_threads=4, rpc_timeout=999999, init_method=f"tcp://{master_addr}:{master_port}"
        ),
    )
    print("client"+str(rank)+" is joining")

    rpc.shutdown()