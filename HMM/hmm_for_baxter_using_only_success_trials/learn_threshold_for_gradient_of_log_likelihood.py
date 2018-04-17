#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import time
import util



def assess_threshold_and_decide(
    gradient_traj_by_time, 
    curve_owner, 
    state_no, 
    figure_save_path, 
):

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # plot log curves of all trials
    for row_no in range(gradient_traj_by_time.shape[0]):
        trial_name = curve_owner[row_no]
        if row_no == 0:
            ax.plot(gradient_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray', label='trials')
        else:
            ax.plot(gradient_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray')

    min_gradient = gradient_traj_by_time.min()
    max_gradient = gradient_traj_by_time.max()
    gradient_range = max_gradient-min_gradient

    threshold = min_gradient-gradient_range/2

    title = 'state %s use threshold %s'%(state_no, threshold, )
    ax.set_title(title)

    output_dir = os.path.join(figure_save_path, 'threshold_for_gradient_of_log_likelihood')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    ax.axhline(y=threshold, linestyle='solid', color='red')
    fig.savefig(os.path.join(output_dir, "state %s.eps"%(state_no, )), format="eps")
    fig.savefig(os.path.join(output_dir, "state %s.png"%(state_no, )), format="png")

    plt.close(1)
        
    return threshold
        
    
def run(model_save_path, 
    figure_save_path,
    threshold_c_value,
    trials_group_by_folder_name):


        
    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        try:
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        except IOError:
            print 'model of state %s not found'%(state_no,)
            continue

    threshold_group_by_state = {}

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

        gradient_traj_by_time = np_matrix_traj_by_time[:, 1:]-np_matrix_traj_by_time[:, :-1]

        threshold_group_by_state[state_no] = assess_threshold_and_decide(
            gradient_traj_by_time, 
            curve_owner, 
            state_no, 
            figure_save_path, 
        )

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
   
    if len(threshold_group_by_state) != 0:
        joblib.dump(threshold_group_by_state, model_save_path+"/threshold_for_gradient_of_log_likelihood.pkl")
