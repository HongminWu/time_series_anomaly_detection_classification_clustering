#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from math import (
    log,
    exp
)
from matplotlib import pyplot as plt
import util
import json
import model_generation



def plot_trials_loglik_curves_of_one_state(
    np_matrix_traj_by_time, 
    curve_owner, 
    state_no, 
    figure_save_path, 
    title = None,
    ):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    trial_amount = np_matrix_traj_by_time.shape[0]

    from matplotlib.pyplot import cm 
    color=iter(cm.rainbow(np.linspace(0, 1, trial_amount)))

    for row_no in range(np_matrix_traj_by_time.shape[0]):
        c=next(color)
        trial_name = curve_owner[row_no]
        ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], color=c)


    if title is None:
        title = 'state %s trial likelihood plot'%(state_no,)
    ax.set_title(title)

    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, title+".png"), format="png")
        
    plt.close(fig)
        
    
def run(model_save_path, 
    model_type,
    figure_save_path,
    threshold_c_value,
    trials_group_by_folder_name):


        
    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)
    list_of_trials = trials_group_by_folder_name.values() 

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    training_report_by_state = {}
    for state_no in range(1, state_amount+1):
        try:
            training_report_by_state[state_no] = json.load(open(model_save_path+"/model_s%s_training_report.json"%(state_no,), 'r'))
        except IOError:
            print 'training report of state %s not found'%(state_no,)
            continue

    model_config_by_state = {}
    for state_no in training_report_by_state:
        best_model_record = training_report_by_state[state_no][0]
        best_model_id = best_model_record.keys()[0]
        model_config_by_state[state_no] = joblib.load(model_save_path+"/model_s%s_config_%s.pkl"%(state_no, best_model_id))


    training_data_group_by_state = {}
    training_length_array_group_by_state = {}

    for state_no in training_report_by_state:

        length_array = []
        for trial_no in range(len(list_of_trials)):
            length_array.append(list_of_trials[trial_no][state_no].shape[0])
            if trial_no == 0:
                data_tempt = list_of_trials[trial_no][state_no]
            else:
                data_tempt = np.concatenate((data_tempt,list_of_trials[trial_no][state_no]),axis = 0)

        X = data_tempt 
        lengths = length_array

        list_of_scored_models = training_report_by_state[state_no]
        model_config_template = model_config_by_state[state_no]

        for idx in range(len(list_of_scored_models)):
            model_id = list_of_scored_models[idx].keys()[0]
            model_score = list_of_scored_models[idx].values()[0]
            model_config = util.bring_model_id_back_to_model_config(model_id, model_config_template)
            model_generator = model_generation.get_model_generator(model_type, model_config)
            model, trash = next(model_generator)

            model = model.fit(X, lengths=lengths)

            all_log_curves_of_this_state = []
            curve_owner = []

            for trial_name in trials_group_by_folder_name:
                curve_owner.append(trial_name)
                one_log_curve_of_this_state = [] 
               
                one_log_curve_of_this_state = util.fast_log_curve_calculation(
                    trials_group_by_folder_name[trial_name][state_no],
                    model,
                )

                all_log_curves_of_this_state.append(one_log_curve_of_this_state)


            np_matrix_traj_by_time = np.matrix(all_log_curves_of_this_state)

            plot_trials_loglik_curves_of_one_state(
                np_matrix_traj_by_time, 
                curve_owner, 
                state_no, 
                os.path.join(figure_save_path, 'check_if_score_metric_converge_loglik_curves', 'state_%s'%(state_no,)), 
                title='state_%s_training_rank_%s_id_%s_score_%s'%(state_no, idx, model_id, model_score)
            )
