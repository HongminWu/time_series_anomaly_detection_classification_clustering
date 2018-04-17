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
import time
import util

import ipdb



def plot_log_prob_of_all_trials(
    list_of_log_prob_mat,
    log_prob_owner, 
    state_no, 
    figure_save_path):


    trial_amount = len(list_of_log_prob_mat)
    hidden_state_amount = list_of_log_prob_mat[0].shape[1]
    fig, ax_list = plt.subplots(nrows=hidden_state_amount)
    if hidden_state_amount == 1:
        ax_list = [ax_list]

    from matplotlib.pyplot import cm 
    import numpy as np

    color=iter(cm.rainbow(np.linspace(0, 1, trial_amount)))
    for i in range(trial_amount):
        c=next(color)
        log_prob_mat = list_of_log_prob_mat[i][:, :].transpose()

        hidden_state_amount = log_prob_mat.shape[0]

        for row_no in range(hidden_state_amount):
            if i == 0:
                ax_list[row_no].plot(log_prob_mat[row_no].tolist(), linestyle="solid", color=c)
                title = 'state %s trial hidden state %s log_prob plot'%(state_no, row_no)
                ax_list[row_no].set_title(title)
            else:
                ax_list[row_no].plot(log_prob_mat[row_no].tolist(), linestyle="solid", color=c)




    if not os.path.isdir(figure_save_path+'/hidden_state_log_prob_plot'):
        os.makedirs(figure_save_path+'/hidden_state_log_prob_plot')
    title = 'state %s trial hidden state log_prob plot'%(state_no,)
    fig.savefig(os.path.join(figure_save_path, 'hidden_state_log_prob_plot', title+".eps"), format="eps")
    plt.close(1)
    
def run(model_save_path, 
    figure_save_path,
    threshold_c_value,
    trials_group_by_folder_name):


        
    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    threshold_constant = 10
    threshold_offset = 10

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        try:
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        except IOError:
            print 'model of state %s not found'%(state_no,)
            continue

    expected_log = []
    std_of_log = []
    deri_threshold = []




    for state_no in model_group_by_state:

        list_of_log_prob_mat = []
        log_prob_owner = []
        for trial_name in trials_group_by_folder_name:
            log_prob_owner.append(trial_name)

            
            hidden_state_log_prob = util.get_hidden_state_log_prob_matrix(
                trials_group_by_folder_name[trial_name][state_no],
                model_group_by_state[state_no]
            )

            list_of_log_prob_mat.append(hidden_state_log_prob)

        # use np matrix to facilitate the computation of mean curve and std 
        plot_log_prob_of_all_trials(
            list_of_log_prob_mat, 
            log_prob_owner, 
            state_no, 
            figure_save_path)
