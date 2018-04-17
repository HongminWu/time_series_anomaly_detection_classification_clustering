from sklearn.externals import joblib
import util
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
import os
import ipdb
import math

def color_bg_by_state(state_order, state_color, state_start_idx, ax, ymin=0.0, ymax=1.0):
    for idx in range(len(state_start_idx)-1):
        color = util.rgba_to_rgb_using_white_bg(state_color[state_order[idx]][:3], 0.25)
        start_at = state_start_idx[idx]
        end_at = state_start_idx[idx+1]
        ax.axvspan(start_at, end_at, facecolor=color, ymax=ymax, ymin=ymin)

def run(model_save_path, 
    figure_save_path,
    trials_group_by_folder_name,
    state_order_group_by_folder_name,
    parsed_options):

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

    state_color = {}
    color=iter(cm.rainbow(np.linspace(0, 1, state_amount)))
    for state_no in model_group_by_state:
        state_color[state_no] = color.next()


    output_dir = os.path.join(figure_save_path, 'test_if_gradient_can_detect_state_switch')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)



    trial_amount = len(trials_group_by_folder_name)
    subpolt_amount_for_each_trial = 2
    subplot_per_row = 1
    subplot_amount = trial_amount*subpolt_amount_for_each_trial
    row_amount = int(math.ceil(float(subplot_amount)/subplot_per_row))
    fig, ax_mat = plt.subplots(nrows=row_amount, ncols=subplot_per_row)
    if row_amount == 1:
        ax_mat = ax_mat.reshape(1, -1)
    if subplot_per_row == 1:
        ax_mat = ax_mat.reshape(-1, 1)

    ax_list = []
    for i in range(trial_amount):
        for k in range(subpolt_amount_for_each_trial):
            j = subpolt_amount_for_each_trial*i+k
            
            row_no = j/subplot_per_row
            col_no = j%subplot_per_row
            ax_list.append(ax_mat[row_no, col_no])


    trial_count = -1
    for trial_name in trials_group_by_folder_name:
        trial_count += 1

        X = None

        state_start_idx = [0]

        state_order = state_order_group_by_folder_name[trial_name]
        for state_no in state_order:
            if X is None:
                X = trials_group_by_folder_name[trial_name][state_no]
            else:
                X = np.concatenate((X, trials_group_by_folder_name[trial_name][state_no]),axis = 0)
            state_start_idx.append(len(X))

        plot_idx = trial_count*2
        ax_loglik = ax_list[plot_idx]
        ax_loglik_gradient = ax_list[plot_idx+1]


        color_bg_by_state(state_order, state_color, state_start_idx, ax_loglik)
        
        color_bg_by_state(state_order, state_color, state_start_idx, ax_loglik_gradient)


        log_lik_mat = []
        log_lik_gradient_mat = []
        mat_row_color = []
        mat_row_name = []
        for state_no in model_group_by_state:
            log_lik_curve = np.array(util.fast_log_curve_calculation(
                X,
                model_group_by_state[state_no]
            ))
            log_lik_gradient_curve = log_lik_curve[1:]-log_lik_curve[:-1]
            
            log_lik_mat.append(log_lik_curve)
            log_lik_gradient_mat.append(log_lik_gradient_curve)
            mat_row_color.append(state_color[state_no])
            mat_row_name.append('state %s'%(state_no,))

        log_lik_mat = np.matrix(log_lik_mat)
        log_lik_gradient_mat = np.matrix(log_lik_gradient_mat)


        log_lik_gradient_mat[log_lik_gradient_mat<0] = 0
        for row_no in range(log_lik_mat.shape[0]):
            ax_loglik.plot(log_lik_mat[row_no].tolist()[0], label=mat_row_name[row_no], color=mat_row_color[row_no])
            ax_loglik_gradient.plot(log_lik_gradient_mat[row_no].tolist()[0], label=mat_row_name[row_no], color=mat_row_color[row_no])




        title = "log-likelihood of %s HMM models"%state_amount
        ax_loglik.set_title(title)
        ax_loglik.set_ylabel('log probability')
        ax_loglik.set_xlabel('time step')
        title = "gradient of log-likelihood of %s HMM models"%state_amount
        ax_loglik_gradient.set_title(title)
        ax_loglik_gradient.set_ylabel('log probability')
        ax_loglik_gradient.set_xlabel('time step')

        title = "trial %s"%(trial_name,)


    fig.set_size_inches(8*subplot_per_row,2*row_amount)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "test_if_gradient_can_detect_state_switch.png"), format="png")
    fig.savefig(os.path.join(output_dir, "test_if_gradient_can_detect_state_switch.eps"), format="eps")
