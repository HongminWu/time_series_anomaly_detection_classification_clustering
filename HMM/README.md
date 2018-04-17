# birl HMM Repository
---

# Purpose Of This Repository
This repository deals with data published by [topic_multimodal node](https://github.com/birlrobotics/birl_baxter_tasks/blob/master/scripts/real_task_common/real_topic_multimodal.py) which resides in [birl_baxter_tasks repository](https://github.com/birlrobotics/birl_baxter_tasks).

The data we're dealing with contain robot endpoint pose and endpoint wrench. We want to analyse those data that are from successful trials, and in the test phase, to detect anomalies in arriving data. This repository serves this purpose.

For the details of our method, have a look at this [paper(link to be updated)]().


# How To Run These Codes
## To Train HMM Model

1. install dependencies
   - [hmmlearn](https://github.com/hmmlearn/hmmlearn) 
   - [hongminhmmpkg(link to be updated)]()
   
1. download our [dataset repository](https://github.com/sklaw/baxter_pick_and_place_data)

1. tune configuration in ./hmm_for_baxter_using_only_success_trials/training_config.py. 
    
    usually, we only need to modify the __config_by_user__ variable. The fields of this variable we need to modify are:
    - bath_path: set this path to be the folder path of the dataset which you want to use. E.g.:
    ```
    path_to_dataset_repository/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170714
    ```

1. run the following commands:

    ```
    cd hmm_for_baxter_using_only_success_trials
    python birl_hmm_user_interface.py --train-model --train-threshold
    ```
    
    these two commands will train the HMM model and train the threshold which is used in anomaly detection.
    
1. check the dataset folder, you will find 2 folders named __model__ and __figure__. __model__ stores the models trained with the dataset. __figure__ stores some plots that help us assess the models.
     
  
## To Run HMM Online Service

Remember that our goal is to provide anomaly detection service for robot task execution. run the following command to bring up this service:

```
python birl_hmm_user_interface.py --online-service
```
