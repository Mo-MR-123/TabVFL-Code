#!/bin/bash

python3 bootstrapper.py \
    --datasets_folder_path "./datasets" \
    --config_path "./config_experiments_loan.yaml" \
    --hyperparams_path "./hyperparams_experiments.yaml" \
    --experiment_type "local_tabnets" \
    --task_type "binary" \
    --predictor "svm,logisticreg,mlp,randomforest,decisiontree,xgboost"
