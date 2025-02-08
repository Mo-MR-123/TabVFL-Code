# TabVFL: A Distributed Framework for Latent Representation Learning in Vertical Federated Learning

This code is part of my Master Thesis at TU Delft. The research is conducted under the supervision and in collaboration with Zilong Zhao, Lydia Y. Chen and Jérémie Decouchant. The [related paper](https://arxiv.org/abs/2404.17990) is submitted to arXiv and can be cited as follows: 
```
@misc{rashad2024tabvflimprovinglatentrepresentation,
      title={TabVFL: Improving Latent Representation in Vertical Federated Learning}, 
      author={Mohamed Rashad and Zilong Zhao and Jeremie Decouchant and Lydia Y. Chen},
      year={2024},
      eprint={2404.17990},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.17990}, 
}
```

Autoencoders are widely used for compressing high-dimensional data which could enhance downstream model performance on downstream tasks. However, existing designs in VFL train separate autoencoders for each participant, potentially breaking feature correlations. To address this challenge and given the prevalence of tabular data in VFL, we adapt TabNet—a state-of-the-art model for tabular data—to the VFL setting.

We propose TabVFL, a novel distributed VFL framework that:

- Preserves privacy by mitigating data leakage with a fully connected layer.
- Maintains feature correlations by learning a unified latent representation.
- Enhances robustness against client failures during training.

The project uses Torch Distributed RPC framework to enable inter-process communication for sending and receiving values/matrices. Although the experiments are run locally, the code can be extended to enable inter-process network communication. A distributed optimizer is also used to keep track of the distributed values being sent to allow backpropagation between clients. The use of this framework is inspired by the work of [Zilong Zhao](https://github.com/zhao-zilong).

Refer to [TabNet](https://github.com/dreamquark-ai/tabnet) library for the full implementation details of the model.

## Prerequisites

- Python 3.8
- Linux (this project does not work on windows)


## Datasets 
The used classification datasets:
- Forest Cover Type (multiclass) (source: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)
- Intrusion (multiclass) (source: https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset)
- Bank Marketing (binary) (source: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?datasetId=4471&sortBy=voteCount)
- Air Passenger Satisfaction (binary, train.csv is used only) (source: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- Rice MSC (multiclass) (source: https://www.kaggle.com/datasets/muratkokludataset/rice-msc-dataset)

NOTE: `Forest Cover Type` and `Intrusion` are already present and preprocessed in the `datasets` folder. Since github doesn't allow big dataset to be stored, you have to download the remaining ones and move them to the `datasets` folder to be used for the experiments. 

### Manual Preprocessing Required On Datasets
- Make sure that you rename columns that have subscripts __ to - since during the splitting of the dataset into vertical partitions, the unique columns are inferred by splitting on __. So, if there are __ symbols in the preprocessed dataset, they should correspond to a one-hot encoded feature. NOTE: Some notebooks already do that for you, but not all of them.

### Preprocessing

Before using the datasets, they must first be preprocessed using the corresponding jupyter notebooks in the `preprocessing` folder. After the notebooks are finished running, a new preprocessed dataset is created in the `datasets` folder with a `_preprocessed` suffix in the file name. This newly created dataset is the one that must be used in the corresponding config yaml file. Some of the notebooks contain some visualizations showing some insights about the data as well. NOTE: you may have to update the name of the dataset to preprocess in the correponding notebook. 

## Installing Dependencies
Make sure to match pytorch cuda version with the cuda version of your nvidia gpu. If those do not match, then that may result in unexpected CUDA errors. Change the CUDA version in `install_dependencies.sh` file if there is a mismatch.

1- Install `python3-venv`:
```
sudo apt-get install python3-venv
```

2- Create a virtual environment called `venv` in the root project folder:
```
python -m venv tabnet_venv
```

3- Activate virtual environment:
```
source tabnet_venv/bin/activate
```

4 - Install/Setup Pytorch and other dependencies from `requirements.txt` file using shell script:
```
chmod +x install_dependencies.sh 
source install_dependencies.sh
```

## Relevant folders and files 

- `tabnet_vfl` folder:  
Contains everything related to the TabNet VFL system that is proposed in the thesis. This folder is used to run experiments for TabVFL design. It contains a `server` folder which contains relevant code related to the adjusted TabNet model in VFL. `server.py` is used to run the server that is responsible for coordinating communication and training of TabNet with clients.

- `local_tabnets` folder:
This folder contains the prior work design. Clients possess a local version of the TabNet model. The model is pre-trained locally on the client's feature split data. The encoder-decoder of the Tabnet model is adjusted a bit to be used as such. The server contains only an FC-layer that is used during finetuning.

- `pytorch_tabnet` folder: This is the original tabnet model without any modifications. Some parts are used by central tabnet and clients of local tabnets. 

- `tabnet_vfl_local_encoder` folder:
TabVFL design but with encoder being local at each client instead of being split.

- `central_tabnet` folder:
The non-federated tabnet run on the full dataset.  

- `bootstrapper.py` file:
This is the main entry point for running the experiments. Which experiment to run and which config yaml file to use can be indicated using the `--experiment_type` and `--config_path` flags respectively. Supported flags:
```sh
'--datasets_folder_path', required=True, 'Relative/Absolute path to the folder containing the datasets.'

'--config_path', required=True, 'Relative/Absolute path to the YAML file containing config info, e.g., use_cuda, num_clients, precentages_columns_each_client etc... .'

'--hyperparams_path', required=True, 'Relative/Absolute path to the YAML file containing tabnet hyperparameters info, e.g., batch_size, n_d, n_a etc... .'

'--experiment_type', required=True, default='tabnet_vfl', 'What experiment to run.'

'--task_type', 'Overwrites the inferred task if needed (either binary or multiclass). NOTE: Although continuous is supported, it is not extensively tested as for this research we did not consider any datasets with continuous label types.'

'--predictors', required=True, 'What predictors to use for evaluation. Multiple predictors can be specified by separating them with a comma.'

'--eval_out', default='eval_data.npz', 'The file name where evaluation results are stored.'
```

## Usage

To start a experiment, create a shell script that runs the `bootstrapper.py` with the required flags. Make sure that you have created config yaml file for the dataset that you want to use and that you have preprocessed the dataset. For each experiment, there is already a shell script created in the root folder of the project that are named using format `run_<design_name>_experiment.sh`. The values can be tweaked to your heart's desire. In case you want to create your own, you can take a look at the example file `run_bootstrapper_example.sh`.

---

The config yaml files are already created for each dataset and are indicated using `config_experiments_<dataset_name>.yaml` format for naming of the files. In case you want to create your own config yaml file, take a look at one of the already created ones (e.g. `config_experiments_air_passenger.yaml`). The following data need to be provided:
```yaml
Dataset specific requirements:
- seed: The seed to use during the experiment
- dataset: The dataset to be processed during experiment
- column_label_name: The column containing the label information for classification
- integer_columns: The columns in the dataset that are of integer type
- categorical_columns: The columns in the dataset that are of categorical type (e.g. one-hot encoded columns)
- mixed_columns: This property is not used for now, but leave this as [] to make sure the code works for now.
- train_ratio, valid_ratio and test_ratio: The ratio of the dataset to use for train, valid and test respectively

NOTE: The features that will be split is combination of "mixed_columns" + "integer_columns" + "categorical_columns" + "column_label_name" in this order

tabnet_vfl:
- master_addr: IP address used to run the experiment, "127.0.0.1" for simulation on localhost
- master_port: The port used to run the experiment, make sure the chosen port is unused
- num_clients: Number of guest clients that participate
- use_cuda: Whether or not to use GPU CUDA cores 
- epoch_failure_probability: The failure probability for client failure experiment (NOTE: this is just a placeholder. Current code does not support client failure setup, please navigate to client-failure-experiment branch)
- data_column_split: The features to assign for each participating guest client. For example, if 5 guest clients participate, then each client should have a specific ratio of features assigned to it uniformally. The total sum of the ratios should add up to 1. So assigning more features to one client means that other clients get less features assigned to them due to this contraint

local_tabnets:
Same as tabnet_vfl without the epoch_failure_probability

central_tabnet:
use_cuda: Whether to use cuda or not
```

---

Also make sure to have a yaml file for the hyperparameters to use (see `hyperparams_experiments.yaml`). The hyperparams file contains hyperparameters for each design. 

NOTE: Both `tabnet_vfl` and `tabvfl_local_encoder` design have an extra hyperparameter that is not part of tabnet called `decoder_split_ratios`. This parameter indicates the ratio of the partial decoder output to each client after splitting at the server. The total ratio should add up to one, otherwise an error occurs. 

Moreover, the `patience` value cannot be passed throguh the `hyperparams_experiments.yaml` file to `tabnet_vfl` and `tabvfl_local_encoder`. This value is set to 10, which is also the case for other designs. If you want to change the `patience` value of the aforementioned designs then you need to navigate to the corresponding `server.py` file and change the `self.patience_pretraining` and/or `self.patience_finetuning`. Since the patience functionality had to be reimplemented for `tabnet_vfl` and `tabvfl_local_encoder`, we customized in a way that you can have different values for patience depending on training phase. However, for all experiments, `patience_pretraining == patience_finetuning`.

## Output of the experiments

After each experiment run, couple of plots and files are generated. These include:

 - Epoch runtime values (in .npz format) and plots of the pretraining and finetuning 
 - Pretraining and finetuning training loss and validation metric values per epoch (including plots corresponding to those values)
 - CSV file containing the performance values of each predictor on the latent test values. This sheet is stored in the temp folder with name format "{experiment_name}_<random_generated_characters>.csv". Example: `central_tabnet_gln324kr.csv`. The file opens up automatically using `xdg-open` command. You can change this in `shared/general_utils::open_df_temp` function. 

## Client Failure Experiment

See branch `client-failure-experiment` branch for code related to the client failure experiment.

## Latent Quality Experiment

For this experiment, the latent value `n_d` in the `hyperparams_experiments.sh` should be modified to the required latent dimension. Note that only for the `local_tabnets` design, the latent dimension should be split evenly among the guest clients.In case an uneven value is chosen, the remainder values should be distributed to guest 1 until guest N in this order (left to right). Example in case of 5 guest clients: n_d = 7, then remainder is 7 mod 5 = 2. Hence guest 1 and guest 2 should be assigned n_d=2 and other guests n_d=1. So n_d=7 becomes [2,2,1,1,1] in the hyperparams file.

We tested 3 runs with different seeds and manually noted the evaluation values in an excel sheet. For each datasets, we created a separate excel sheet noting the average performance of each design. These excel sheets are stored in `experiments_results/latent_data_evaluation_experiment` for each dataset. The results of all of these datasets are noted in `experiments_results/latent_eval_means_per_design.xlsx`. The `experiments_results/experiment_latent_eval_differences.xlsx` file contains the improvement of TabVFL compared to the baseline design local tabnets. The improvements are denoted for each evaluation metrics (ROC-AUC, F1-score and Accuracy). 

To generate an image of the bar plots for gauging the difference in performance of the designs, the python script `experiments_results/latent_data_evaluation_plots.py` can be run. The script reads the values in `experiments_results/latent_eval_means_per_design.xlsx` for each design to create the image.