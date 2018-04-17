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



def assess_threshold_and_decide(
    np_matrix_traj_by_time, 
    curve_owner, 
    state_no, 
    output_dir, 
    data_class,
):

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    trial_amount = np_matrix_traj_by_time.shape[0]

    from matplotlib.pyplot import cm 
    color=iter(cm.rainbow(np.linspace(0, 1, trial_amount)))

    for row_no in range(np_matrix_traj_by_time.shape[0]):
        c=next(color)
        trial_name = curve_owner[row_no]
        gradient = np_matrix_traj_by_time[row_no][0, 1:]-np_matrix_traj_by_time[row_no][0, :-1]
        ax.plot(gradient.tolist()[0], color=c, label='trial \"%s\"'%curve_owner[row_no])
        



    title = 'trial class \"%s\": graident of log-likelihood output by skill model %s'%(data_class, state_no, )
    ax.set_title(title)
    ax.set_xlabel('time step')
    ax.set_ylabel('log probability')
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, title+".eps"), format="eps")
    fig.savefig(os.path.join(output_dir, title+".png"), format="png")


    plt.close(1)

        

            
        
        
    
def run(model_save_path, 
    figure_save_path,
    threshold_c_value,
    trials_group_by_folder_name,
    data_class,
):

    output_dir = os.path.join(
        figure_save_path,
        "gradient_of_log_likelihood_plot",
        data_class,
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
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

    for state_no in model_group_by_state:


        all_log_curves_of_this_state = []
        curve_owner = []
        for trial_name in trials_group_by_folder_name:
            curve_owner.append(trial_name)
            one_log_curve_of_this_state = [] 

            one_log_curve_of_this_state = util.fast_log_curve_calculation(
                trials_group_by_folder_name[trial_name][state_no],
                model_group_by_state[state_no]
            )


            all_log_curves_of_this_state.append(one_log_curve_of_this_state)

        # use np matrix to facilitate the computation of mean curve and std 
        np_matrix_traj_by_time = np.matrix(all_log_curves_of_this_state)


        assess_threshold_and_decide(
            np_matrix_traj_by_time, 
            curve_owner, 
            state_no, 
            output_dir, 
            data_class,
        )
