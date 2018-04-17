#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import time
import util
import ipdb

def assess_threshold_and_decide(
    threshold_c_value,
    mean_of_log_curve,
    std_of_log_curve,
    np_matrix_traj_by_time,
    curve_owner,
    state_no,
    figure_save_path,
    score_time_cost_per_point):

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # plot log curves of all trials
    for row_no in range(np_matrix_traj_by_time.shape[0]):
        trial_name = curve_owner[row_no]
        if row_no == 0:
            ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray', label='trials')
        else:
            ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray')

    # plot mean-c*std log curve
    for c in np.arange(0, 20, 2):
        ax.plot((mean_of_log_curve-c*std_of_log_curve).tolist()[0], label="mean-%s*std"%(c,), linestyle='dotted')

    c = threshold_c_value
    title = 'state %s use threshold with c=%s'%(state_no, c, )
    ax.set_title(title)

    output_dir = os.path.join(figure_save_path, 'threshold_for_log_likelihood')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    ax.plot((mean_of_log_curve-c*std_of_log_curve).tolist()[0], label="mean-%s*std"%(c,), linestyle='solid')
    fig.savefig(os.path.join(output_dir, "state %s threshold_c %s.eps"%(state_no, c)), format="eps")
    fig.savefig(os.path.join(output_dir, "state %s threshold_c %s.png"%(state_no, c)), format="png")

    plt.close(1)

    if threshold_c_value == 0:
        return mean_of_log_curve
    else:
        return mean_of_log_curve-c*std_of_log_curve


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

    expected_log = {}
    std_of_log = {}
    threshold = {}
    average_predict_proba = {}

    for state_no in model_group_by_state:
        compute_score_time_cost = 0
        total_step_times = 0


        all_log_curves_of_this_state = []
        all_predict_proba_of_this_state = []
        curve_owner = []
        for trial_name in trials_group_by_folder_name:
            curve_owner.append(trial_name)
            one_log_curve_of_this_state = []
            one_predice_proba_of_this_state = []

            start_time = time.time()

            one_log_curve_of_this_state = util.fast_log_curve_calculation(
                trials_group_by_folder_name[trial_name][state_no],
                model_group_by_state[state_no]
            )
            all_log_curves_of_this_state.append(one_log_curve_of_this_state)

            one_predice_proba_of_this_state = model_group_by_state[state_no].predict_proba(trials_group_by_folder_name[trial_name][state_no]) # HDPHSMM haven't implemented this
            all_predict_proba_of_this_state.append(one_predice_proba_of_this_state)

            compute_score_time_cost += time.time()-start_time
            total_step_times += len(trials_group_by_folder_name[trial_name][state_no])

        # use np matrix to facilitate the computation of mean curve and std
        np_matrix_traj_by_time = np.matrix(all_log_curves_of_this_state)
        mean_of_log_curve = np_matrix_traj_by_time.mean(0)
        std_of_log_curve = np_matrix_traj_by_time.std(0)

        # use np matrix to facilitate the computation of average probability of each hidden state
        sum_predict_proba_of_this_state = np.sum(all_predict_proba_of_this_state, axis=0)
        average_sum_predict_proba_of_this_state = sum_predict_proba_of_this_state/len(all_predict_proba_of_this_state)

        score_time_cost_per_point = float(compute_score_time_cost)/total_step_times
        decided_threshold_log_curve = assess_threshold_and_decide(
            threshold_c_value,
            mean_of_log_curve,
            std_of_log_curve,
            np_matrix_traj_by_time,
            curve_owner,
            state_no,
            figure_save_path,
            score_time_cost_per_point)
        expected_log[state_no] = mean_of_log_curve.tolist()[0]
        threshold[state_no] = decided_threshold_log_curve.tolist()[0]
        std_of_log[state_no] = std_of_log_curve.tolist()[0]
        average_predict_proba[state_no] = average_sum_predict_proba_of_this_state
        joblib.dump(all_log_curves_of_this_state, model_save_path + "/all_log_curves_of_this_state.pkl")
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    joblib.dump(expected_log, model_save_path + "/mean_of_log_likelihood.pkl")
    joblib.dump(threshold, model_save_path    + "/threshold_for_log_likelihood.pkl")
    joblib.dump(std_of_log, model_save_path   + "/std_of_log_likelihood.pkl")
    joblib.dump(average_predict_proba, model_save_path + "/average_predict_proba.pkl")
