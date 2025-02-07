import argparse
import traceback
from typing import Any, List, Tuple, Union
import torch
import pandas as pd
from pathlib import Path
from torch.nn import Linear
import torch.distributed.rpc as rpc
from torch.utils.data import DataLoader
# from torch.nn import functional as F
# from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch.nn import Linear, Module
import warnings
from torchinfo import summary
warnings.filterwarnings("ignore")

from pytorch_tabnet.tab_network import TabNetEncoder
from tabnet_vfl_local_encoder.one_client.tabnet_client_model import RandomObfuscator, UnsupervisedLoss
from tabnet_vfl_local_encoder.one_client.tabnet_utils import create_group_matrix, initialize_non_glu

random_seed = 42

# Set the random seed for NumPy
np.random.seed(random_seed)

# Set the random seed for PyTorch CPU operations
torch.manual_seed(random_seed)

# Set the random seed for PyTorch GPU operations (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Global constants
# SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent

# np.random.seed(SEED) # IMPORTANT FOR THE SHUFFLE OF THE DATA
# Set a seed for PyTorch
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

# def _call_method(method, rref, *args, **kwargs):
#     """helper for _remote_method()"""
#     return method(rref.local_value(), *args, **kwargs)
    
# def _remote_method(method, rref, *args, **kwargs):
#     """
#     executes method(*args, **kwargs) on the from the machine that owns rref
#     very similar to rref.remote().method(*args, **kwargs), but method() doesn't have to be in the remote scope
#     """
#     args = [method, rref] + list(args)
#     return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    # print(param_rrefs)
    return param_rrefs


class LocalEncoder(Module):
    def __init__(
        self,
        input_dim: int,
        encoder_params: dict,
        group_matrix: torch.Tensor, 
        pretraining_ratio: float,
    ):
        super(LocalEncoder, self).__init__()
        # self.is_training = True

        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=input_dim,
            group_attention_matrix=group_matrix,
            **encoder_params
        )
        
        self.masker = RandomObfuscator(pretraining_ratio=pretraining_ratio, group_matrix=group_matrix)

    def forward(self, x, is_pretraining) -> Union[Tuple[List[torch.Tensor], Any, Any, Any], Tuple[torch.Tensor, torch.Tensor]]:
        # if self.is_training:
        if is_pretraining:
            masked_x, obfuscated_groups, obfuscated_vars = self.masker(x)
            # set prior of encoder with obfuscated groups
            prior = 1 - obfuscated_groups
            steps_out, _ = self.encoder(masked_x, prior=prior)
            return steps_out, x, obfuscated_vars
        else:
            steps_out, M_loss = self.encoder(x)
            res = torch.sum(torch.stack(steps_out, dim=0), dim=0)
            return res, M_loss

class TabNetClientEncoder():
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
        client_id: int,
        epochs: int, 
        use_cuda: bool,
        batch_size: int,
        cols_info_dict: dict,
        encoder_params: dict,
        # valid_ratio: float,
        test_ratio: float,
        seed: int,
        pretraining_ratio: float,
        **kwargs
    ):
        self.client_id = client_id
        self.epochs = epochs
        self.seed = seed
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.pretraining_ratio = pretraining_ratio
        self.encoder_params = encoder_params
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if self.device.type != 'cpu':
            self.use_cuda = True
        self.X_train = X_train_df
        self.X_valid = X_valid_df
        self.X_test = X_test_df
        # self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.column_number = self.X_train.shape[1]
        self.column_names = list(self.X_train.columns)
        self.categorical_columns = cols_info_dict['categorical_columns']
        self.log_columns = []
        self.mixed_columns= cols_info_dict['mixed_columns']
        self.integer_columns = cols_info_dict['integer_columns']
        self.reconstruction_loss = UnsupervisedLoss
        self.curr_x_masked = None
        self.curr_obfuscated_vars = None
        self.curr_embedded_x = None
        self.steps_out = None
        self.decoder_input_nd_dim = None

        # for validation
        self.steps_out_valid = None
        self.curr_embedded_x_valid = None
        self.curr_obfuscated_vars_valid = None

        print(f"{client_id=}, ------------------- INIT -------------------")
        print(f"{client_id=}, Epochs received: {epochs}")
        print(f"{client_id=}, use_cuda received: {use_cuda}")
        print(f"{client_id=}, batch_size received: {batch_size}")
        print(f"{client_id=}, X_train DataFrame: {self.X_train.head()}")
        print(f"{client_id=}, DataFrame column num: {self.column_number}")
        print(f"{client_id=}, DataFrame column names: {self.column_names}")
        print(f"{client_id=}, seed: {self.seed}")
        print(f"{client_id=}, pretraining_ratio: {self.pretraining_ratio}")
        print(f"{client_id=}, DataFrame.shape: {self.X_train.shape=}")
        print(f"{client_id=}, {cols_info_dict=}")
        print(f"{client_id=}, {self.categorical_columns=}")
        print(f"{client_id=}, {self.mixed_columns=}")
        print(f"{client_id=}, {self.integer_columns=}")
        print(f"{client_id=}, {self.reconstruction_loss=}")
        print(f"{client_id=}, {self.encoder_params=}")

        # input dimension of the preprocessed data before embedding
        self.input_dim = self.X_train.shape[1]
        print(f"{self.client_id=} input_dim: ", self.input_dim)
        
        self.group_attention_matrix = create_group_matrix([], self.input_dim).to(self.device) # for grouping preprocessed features into groups to be treated/processed as one feature by the attention mechanisms
        print(f"{self.client_id=} group_attention_matrix.shape: ", self.group_attention_matrix.shape)

        # construct local dataloader
        self.data_loader = DataLoader(self.X_train.to_numpy(), self.batch_size, shuffle=False)
        self.iterloader = iter(self.data_loader)

        print(f"{client_id=}, ------------------- INIT END -------------------")

    def get_local_encoder_dim(self):
        return self.encoder_params["n_d"]

    def init_decoder(self, new_nd_split):
        """Replaces the current decoder and assigns a new n_d split input dim. 
        This is supposed to be used during dynamic change of the splits during experiments
        and during first initialization of the decoder.

        Args:
            new_nd_split (int): n_d input dimension of the decoder
        """
        self.decoder_input_nd_dim = new_nd_split
        print(f"{self.client_id=} self.decoder_input_nd_dim = {self.decoder_input_nd_dim}")
        print(f"{self.client_id=} self.input_dim = {self.input_dim}")
        torch.manual_seed(self.seed)
        self.local_decoder = Linear(self.decoder_input_nd_dim, self.input_dim, bias=False).to(self.device)
        initialize_non_glu(self.local_decoder, self.decoder_input_nd_dim, self.input_dim)
        
    def init_encoder(self):
        torch.manual_seed(self.seed)
        self.local_encoder = LocalEncoder(
            input_dim=self.input_dim,
            encoder_params=self.encoder_params,
            group_matrix=self.group_attention_matrix,
            pretraining_ratio=self.pretraining_ratio,
        ).to(self.device)

    def reset_iterloader(self) -> None:
        self.iterloader = iter(self.data_loader)

    def get_batch_data(self) -> torch.Tensor:
        try:
            data = next(self.iterloader)
        except StopIteration:
            self.reset_iterloader()
            data = next(self.iterloader)
        return data
    
    def forward_pass_decoder_valid(self, server_partial_decoder_logits):
        self.local_decoder.eval()

        assert self.local_encoder.training == True
        assert self.local_decoder.training == False
        assert self.curr_embedded_x_valid is not None
        assert self.curr_obfuscated_vars_valid is not None

        with torch.no_grad():
            reconstructed_output = self.local_decoder(server_partial_decoder_logits.to(self.device).float())

            loss = self.reconstruction_loss(reconstructed_output, self.curr_embedded_x_valid, self.curr_obfuscated_vars_valid)

            # send loss to server by converting the output to cpu first
            # since PyTorch RPC does not support CUDA tensors 
            self.local_decoder.train()

            if self.use_cuda:
                return loss.cpu()
            else:
                return loss
    
    def forward_pass_encoder_valid(self, is_pretraining: bool):
        self.local_encoder.eval()
        
        assert self.local_encoder.training == False, "local encoder is not in eval mode"

        with torch.no_grad():
            # convert numbers to float of the current batch of data
            validation_samples = torch.from_numpy(self.X_valid.to_numpy()).to(self.device).float()

            if is_pretraining:
                steps_out, embedded_x, obfuscated_vars = self.local_encoder(validation_samples, is_pretraining)

                self.steps_out_valid = steps_out
                self.curr_embedded_x_valid = embedded_x
                self.curr_obfuscated_vars_valid = obfuscated_vars

                self.local_encoder.train()

                # send intermediate output to server by converting the output to cpu first
                # since PyTorch RPC does not support CUDA tensors 
                if self.use_cuda:
                    # converting tensors in steps_out list to cpu for transmission
                    for tensor_idx in range(len(steps_out)):
                        steps_out[tensor_idx] = steps_out[tensor_idx].cpu()
                    
                    for tensor in steps_out:
                        assert tensor.device.type == "cpu"
                    return steps_out
                else:
                    return steps_out
            else: 
                intermediate_output, M_loss = self.local_encoder(validation_samples, is_pretraining)

                self.local_encoder.train()

                if self.use_cuda:
                    return intermediate_output.cpu(), M_loss.cpu()
                else:
                    return intermediate_output, M_loss

    def forward_pass_encoder_test_data(self) -> torch.Tensor:
        self.local_encoder.eval()

        assert self.local_encoder.training == False

        with torch.no_grad():
            intermediate_output, _ = self.local_encoder(torch.from_numpy(self.X_test.to_numpy()).to(self.device).float(), False)

        # send intermediate output to server by converting the output to cpu first
        # since PyTorch RPC does not support CUDA tensors 
        if self.use_cuda:
            return intermediate_output.cpu()
        else:
            return intermediate_output

    def forward_pass_encoder_valid_data(self) -> torch.Tensor:
        self.local_encoder.eval()

        assert self.local_encoder.training == False

        with torch.no_grad():
            intermediate_output, _ = self.local_encoder(torch.from_numpy(self.X_valid.to_numpy()).to(self.device).float(), False)

        # send intermediate output to server by converting the output to cpu first
        # since PyTorch RPC does not support CUDA tensors 
        if self.use_cuda:
            return intermediate_output.cpu()
        else:
            return intermediate_output

    def forward_pass_encoder_train_data(self) -> torch.Tensor:
        self.local_encoder.eval()

        assert self.local_encoder.training == False

        with torch.no_grad():
            intermediate_output, _ = self.local_encoder(torch.from_numpy(self.X_train.to_numpy()).to(self.device).float(), False)

        # send intermediate output to server by converting the output to cpu first
        # since PyTorch RPC does not support CUDA tensors 
        if self.use_cuda:
            return intermediate_output.cpu()
        else:
            return intermediate_output

    def forward_pass_encoder(self, is_pretraining: bool):
        """
        This function should only be used during training for forward pass of encoder
        """
        self.local_encoder.train()
        
        assert self.local_encoder.training == True

        curr_batch_of_data = self.get_batch_data()

        curr_batch_of_data = curr_batch_of_data.to(self.device).float()
        self.curr_batch_of_data = curr_batch_of_data

        if is_pretraining:
            steps_out, embedded_x, obfuscated_vars = self.local_encoder(curr_batch_of_data, is_pretraining)

            self.steps_out = steps_out
            self.curr_embedded_x = embedded_x
            self.curr_obfuscated_vars = obfuscated_vars

            # send intermediate output to server by converting the output to cpu first
            # since PyTorch RPC does not support CUDA tensors 
            if self.use_cuda:
                # converting tensors in steps_out list to cpu for transmission
                for tensor_idx in range(len(steps_out)):
                    steps_out[tensor_idx] = steps_out[tensor_idx].cpu()
                
                for tensor in steps_out:
                    assert tensor.device.type == "cpu"
                
                return steps_out
            else:
                return steps_out
        else: 
            latent_output, M_loss = self.local_encoder(curr_batch_of_data, is_pretraining)
            if self.use_cuda:
                return latent_output.cpu(), M_loss.cpu()
            else:
                return latent_output, M_loss
    
    def forward_pass_decoder(self, server_partial_decoder_logits):
        """calculate reconstruction loss and update global model params with gradients
        autograd needs to be used here so we need context from server and optimizer
        but server has to keep track of both not clients. So applied solution:
            send only loss back to server after calculating reconstruction loss
            the server then updates the params of all models in client and the server itself
            since the server has the rrefs of all the params acquired during initialization step with clients
        
        server_partial_decoder_logits (torch.Tensor): logits of the partial decoder part in the server with shape (batch_size, decoder_input_nd_dim)
        """
        assert self.local_encoder.training == True
        assert self.local_decoder.training == True
        assert self.curr_embedded_x is not None
        assert self.curr_batch_of_data is not None
        assert self.steps_out != []

        reconstructed_output = self.local_decoder(server_partial_decoder_logits.to(self.device).float())

        loss = self.reconstruction_loss(reconstructed_output, self.curr_embedded_x, self.curr_obfuscated_vars)

        # send loss to server by converting the output to cpu first
        # since PyTorch RPC does not support CUDA tensors 
        if self.use_cuda:
            return loss.cpu()
        else:
            return loss
    

    def init_local_encoder_decoder(
        self, 
        decoder_nd_dim: int,
    ):
        self.init_encoder()

        # init local decoder
        self.init_decoder(decoder_nd_dim)

        print(f"local encoder at client {self.client_id}: initialized")
        print(f"local decoder at client {self.client_id}: initialized")
        
        if self.use_cuda:
            self.local_encoder.cuda()
            self.local_decoder.cuda()
        
        assert self.local_encoder is not None and self.local_decoder is not None 

    def get_data_length(self):
        return len(self.X_train)

    def get_client_id(self):
        return self.client_id
    
    def register_local_encoder(self):
        return param_rrefs(self.local_encoder)
    
    def register_local_decoder(self):
        return param_rrefs(self.local_decoder)

    def send_client_local_encoder_rrefs(self):
        """Send a reference to the local_encoder (for future RPC calls) and the conditioner, transformer and steps/epoch"""
        return rpc.RRef(self.local_encoder)

    def send_client_local_decoder_rrefs(self):
        """Send a reference to the local_decoder (for future RPC calls) and the conditioner, transformer and steps/epoch"""
        return rpc.RRef(self.local_decoder)

def run(
    rank, 
    master_addr, 
    master_port, 
    num_clients,
):
    try:
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
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        rpc.shutdown()
