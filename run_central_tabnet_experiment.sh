#!/bin/bash

python bootstrapper.py \
    --datasets_folder_path "./datasets" \
    --config_path "./config_experiments_covertype.yaml" \
    --hyperparams_path "./hyperparams_experiments.yaml" \
    --experiment_type "central_tabnet" \
    --task_type "multiclass" \
    --predictor "svm,logisticreg,mlp,randomforest,decisiontree,xgboost"
