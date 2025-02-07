# TabVFL: Improving Latent Representation in Vertical Federated Learning

## Client Failure Experiment

This is a separate branch implementing the client failure simulation in the TabVFL design. The results are stored in the `experiments_results` folder. 

**NOTE:** The client failure simulation is either completely or partially implemented in other designs (except central tabnet). Since client failure simulation is used only in TabVFL, the simulation on other designs is not tested. Use at your own risk.  

The `client_failure_f1_score_tabvfl_cache_plus_pretrain` excel file contains the results for the cache method simulation taking into account client failures in both training phases (finetuning + pretraining).

The `client_failure_f1_score_tabvfl_zeros_plus_pretrain` excel file contains the results for the zeros method simulation also taking into account client failures in both training phases.

The python script `client_failure_designs_improvement_calc` calculates the metrics shown in `TABLE IV` in the paper using both excel files mentioned above.

The script `client_failure_tabvfl_methods_plus_pretrain_plots` generates a plot showing the effect of client failure probability on the F1-score (Fig. 6 in the paper).