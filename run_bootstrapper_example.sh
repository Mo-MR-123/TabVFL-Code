#!/bin/bash

python bootstrapper.py \
    # Path to the dataset to use for the experiment
    --datasets_folder_path "./datasets" \
    # Path to the config file corresponding to the dataset
    --config_path "./config_experiments_air_passenger.yaml" \
    # Path to the hyperparameters yaml file to use for the model
    --hyperparams_path "./hyperparams_experiments.yaml" \
    # Experiment type: tabnet_vfl, local_tabnets, central_tabnet or tabnet_vfl_local_encoder
    --experiment_type "central_tabnet" \
    # You can explicitely indicate what the task type is of the dataset
    --task_type "binary" \
    # You can choose what predictors to use for evaluation, make sure they are comma separated
    --predictor "svm,logisticreg,mlp,randomforest,decisiontree,xgboost"