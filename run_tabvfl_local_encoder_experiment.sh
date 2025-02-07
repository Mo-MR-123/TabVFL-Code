#!/bin/bash

python3 bootstrapper.py \
    --datasets_folder_path "./datasets" \
    --config_path "./config_experiments_covertype.yaml" \
    --hyperparams_path "./hyperparams_experiments.yaml" \
    --experiment_type "tabnet_vfl_local_encoder" \
    --task_type "multiclass" \
    --predictor "svm,logisticreg,mlp,randomforest,decisiontree,xgboost"
