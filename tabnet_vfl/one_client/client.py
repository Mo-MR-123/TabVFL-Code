import argparse
import traceback
import torch
import pandas as pd
from pathlib import Path
from torch.nn import Linear
import torch.distributed.rpc as rpc
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import os
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, Sequential, MSELoss, Sigmoid
import warnings
from torchinfo import summary
warnings.filterwarnings("ignore")

from tabnet_vfl.one_client.tabnet_client_model import RandomObfuscator, UnsupervisedLoss
from tabnet_vfl.one_client.tabnet_utils import create_group_matrix, initialize_non_glu

random_seed = 42

# Set the random seed for NumPy
np.random.seed(random_seed)

# Set the random seed for PyTorch CPU operations
torch.manual_seed(random_seed)

# Set the random seed for PyTorch GPU operations (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Global constants
SCRIPT_DIR = Path(__file__).resolve().parent

def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    return param_rrefs

class LocalEncoder(Module):
    def __init__(
        self, 
        input_dim,
        group_matrix, 
        pretraining_ratio, 
    ):
        super(LocalEncoder, self).__init__()

        self.bn = BatchNorm1d(input_dim, momentum=0.01)
        
        # FC layer to prevent data leakage through batch norm layer. In special cases,
        # BN layer could just output the same input ad its output which is not allowed
        # BUT if the results are way worse, then we should either puzzle around with output dim
        # or completely remove the fc layer
        self.fc_layer = Linear(in_features=input_dim, out_features=input_dim, bias=False)
        initialize_non_glu(self.fc_layer, input_dim, input_dim)

        self.masker = RandomObfuscator(pretraining_ratio=pretraining_ratio, group_matrix=group_matrix)

    def forward(self, x, is_training):
        if is_training:
            masked_x, obfuscated_groups, obfuscated_vars = self.masker(x)
            # set prior of encoder with obfuscated groups
            prior = 1 - obfuscated_groups
            x_bn = self.bn(masked_x)
            x_fc = self.fc_layer(x_bn)
            return x_fc, x, prior, obfuscated_vars
        else:
            x_bn = self.bn(x)
            x_fc = self.fc_layer(x_bn)
            return x_fc

class TabNetClient():
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
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if self.device.type != 'cpu':
            self.use_cuda = True
        self.X_train = X_train_df
        self.X_valid = X_valid_df
        self.X_test = X_test_df
        self.column_number = self.X_train.shape[1]
        self.column_names = list(self.X_train.columns)
        self.categorical_columns = cols_info_dict['categorical_columns']
        self.log_columns = []
        self.mixed_columns= cols_info_dict['mixed_columns']
        self.integer_columns = cols_info_dict['integer_columns']
        self.reconstruction_loss = UnsupervisedLoss
        self.curr_obfuscated_vars = None
        self.curr_embedded_x = None
        self.intermediate_output_bn = None
        self.decoder_input_nd_dim = None

        # for validation
        self.curr_obfuscated_vars_valid = None
        self.curr_embedded_x_valid = None
        self.intermediate_output_bn_valid = None

        print(f"{client_id=}, ------------------- INIT -------------------")
        print(f"{client_id=}, Epochs received: {epochs}")
        print(f"{client_id=}, use_cuda received: {use_cuda}")
        print(f"{client_id=}, batch_size received: {batch_size}")
        print(f"{client_id=}, DataFrame: {self.X_train.head()}")
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
        print(f"{client_id=}, {self.X_train=}")
        print(f"{client_id=}, {self.X_test=}")
        print(f"{client_id=}, {self.X_valid=}")

        # input dimension of the preprocessed data before embedding
        self.input_dim = self.X_train.shape[1]
        print(f"{self.client_id=} input_dim: ", self.input_dim)
        
        self.group_attention_matrix = create_group_matrix([], self.input_dim).to(self.device) # for grouping preprocessed features into groups to be treated/processed as one feature by the attention mechanisms

        # construct local dataloader
        self.data_loader = DataLoader(self.X_train.to_numpy(), self.batch_size, shuffle=False)
        self.iterloader = iter(self.data_loader)

        print(f"{client_id=}, ------------------- INIT END -------------------")

    def get_local_encoder_dim(self):
        return self.input_dim

    def init_decoder(self, new_nd_split):
        """Replaces the current decoder and assigns a new n_d split input dim. 
        This is supposed to be used during dynamic change of the splits during experiments
        and during first initialization of the decoder.

        Args:
            new_nd_split (int): n_d input dimension of the decoder
        """
        print(f"{self.client_id=} ------------------- INIT DECODER -------------------")
        self.decoder_input_nd_dim = new_nd_split
        print(f"{self.client_id=} self.decoder_input_nd_dim = {self.decoder_input_nd_dim}")
        print(f"{self.client_id=} self.input_dim = {self.input_dim}")
        torch.manual_seed(self.seed)
        self.local_decoder = Linear(self.decoder_input_nd_dim, self.input_dim, bias=False).to(self.device)
        initialize_non_glu(self.local_decoder, self.decoder_input_nd_dim, self.input_dim)

        print(f"{self.client_id=} ------------------- INIT DECODER END -------------------")

    def init_encoder(self):
        print(f"{self.client_id=} ------------------- INIT ENCODER -------------------")

        torch.manual_seed(self.seed)
        self.local_encoder = LocalEncoder(
            input_dim=self.input_dim,
            group_matrix=self.group_attention_matrix,
            pretraining_ratio=self.pretraining_ratio,
        ).to(self.device)
        print(f"{self.client_id=} ------------------- INIT ENCODER END -------------------")
    
    def reset_iterloader(self) -> None:
        self.iterloader = iter(self.data_loader)

    def get_batch_data(self) -> torch.Tensor:
        try:
            data = next(self.iterloader)
        except StopIteration:
            self.reset_iterloader()
            data = next(self.iterloader)
        return data
    
    def forward_pass_encoder_valid(self, is_pretraining: bool):
        self.local_encoder.eval()
        
        assert self.local_encoder.training == False, "local encoder is not in eval mode"

        with torch.no_grad():
            validation_samples = torch.from_numpy(self.X_valid.to_numpy()).to(self.device).float()

            if is_pretraining:
                intermediate_masked_x_bn, embedded_x, prior, obfuscated_vars = self.local_encoder(validation_samples, is_pretraining)

                self.intermediate_output_bn_valid = intermediate_masked_x_bn
                self.curr_embedded_x_valid = embedded_x
                self.curr_obfuscated_vars_valid = obfuscated_vars

                self.local_encoder.train()

                # send intermediate output to server by converting the output to cpu first
                # since PyTorch RPC does not support CUDA tensors 
                if self.use_cuda:
                    return intermediate_masked_x_bn.cpu(), prior.cpu()
                else:
                    return intermediate_masked_x_bn, prior
            else: 
                intermediate_output = self.local_encoder(validation_samples, is_pretraining)

                self.local_encoder.train()

                if self.use_cuda:
                    return intermediate_output.cpu()
                else:
                    return intermediate_output
            
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

    def forward_pass_encoder_test_data(self) -> torch.Tensor:
        self.local_encoder.eval()

        assert self.local_encoder.training == False

        with torch.no_grad():
            intermediate_output = self.local_encoder(torch.from_numpy(self.X_test.to_numpy()).to(self.device).float(), False)

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
            intermediate_output = self.local_encoder(torch.from_numpy(self.X_valid.to_numpy()).to(self.device).float(), False)

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
            intermediate_output = self.local_encoder(torch.from_numpy(self.X_train.to_numpy()).to(self.device).float(), False)

        # send intermediate output to server by converting the output to cpu first
        # since PyTorch RPC does not support CUDA tensors 
        if self.use_cuda:
            return intermediate_output.cpu()
        else:
            return intermediate_output

    def forward_pass_encoder(self, is_pretraining: bool):
        """This function should only be used during training for forward pass of encoder
        """
        self.local_encoder.train()
        
        assert self.local_encoder.training == True

        curr_batch_of_data = self.get_batch_data()

        # convert numbers to float of the current batch of data
        curr_batch_of_data = curr_batch_of_data.to(self.device).float()
        self.curr_batch_of_data = curr_batch_of_data

        if is_pretraining:
            assert is_pretraining == True

            intermediate_masked_x_bn, embedded_x, prior, obfuscated_vars = self.local_encoder(curr_batch_of_data, is_pretraining)

            self.intermediate_output_bn = intermediate_masked_x_bn
            self.curr_embedded_x = embedded_x
            self.curr_obfuscated_vars = obfuscated_vars

            # send intermediate output to server by converting the output to cpu first
            # since PyTorch RPC does not support CUDA tensors 
            if self.use_cuda:
                return intermediate_masked_x_bn.cpu(), prior.cpu()
            else:
                return intermediate_masked_x_bn, prior
        else: 
            assert is_pretraining == False, "pretraining is done"

            intermediate_output = self.local_encoder(curr_batch_of_data, is_pretraining)
            if self.use_cuda:
                return intermediate_output.cpu()
            else:
                return intermediate_output
    
    def forward_pass_decoder(self, server_partial_decoder_logits):
        """calculate reconstruction loss and update global model params with gradients
        autograd needs to be used here so we need context from server and optimizer
        but server has to keep track of both not clients. So applied solution:
            send only loss back to server after calculating reconstruction loss
            the server then updates the params of all models in client and the server itself
            since the server has the rrefs of all the params acquired during initialization step with clients
        
        server_partial_decoder_logits (torch.Tensor): logits of the partial decoder part in the server with shape (batch_size, decoder_input_nd_dim)
        """
        self.local_decoder.train()

        assert self.local_encoder.training == True
        assert self.local_decoder.training == True
        assert self.curr_embedded_x is not None
        assert self.curr_batch_of_data is not None
        assert self.intermediate_output_bn is not None

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
        # init local encoder 
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
        """Send a reference to the local_encoder (for future RPC calls) steps/epoch"""
        return rpc.RRef(self.local_encoder)

    def send_client_local_decoder_rrefs(self):
        """Send a reference to the local_decoder (for future RPC calls) steps/epoch"""
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
                num_send_recv_threads=4, rpc_timeout=999, init_method=f"tcp://{master_addr}:{master_port}"
            ),
        )
        print("client"+str(rank)+" is joining")
    except Exception as e:
        print(e)
        traceback.print_exc()
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
        if rank == 0:
            raise ValueError("rank 0 is reserved for the server")
            
        # set environment information
        os.environ["MASTER_ADDR"] = ip
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        print("number of epochs before initialization: ", epochs)
        print("world size: ", world_size, f"tcp://{ip}:{port}")
        
        rpc.init_rpc(
            "client"+str(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.PROCESS_GROUP,
            rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                num_send_recv_threads=4, rpc_timeout=120, init_method=f"tcp://{ip}:{port}"
            ),
        )
        print("client"+str(rank)+" is joining")
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument(
        "--dataset", type=str, default="mnist"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument('--use_cuda',  type=str, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
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
