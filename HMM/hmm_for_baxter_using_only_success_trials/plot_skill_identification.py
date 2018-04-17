#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import util
import ipdb


def run(model_save_path,
        figure_save_path,
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

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        from matplotlib.pyplot import cm
    for trial_name in trials_group_by_folder_name:
        color = iter(cm.rainbow(np.linspace(0, 1, state_amount)))
        all_log_curves_of_this_model = [[]]
        for model_no in model_group_by_state:
            all_log_curves_of_this_model.append([])
            for state_no in range(1, state_amount + 1):
               one_log_curve_of_this_model = util.fast_log_curve_calculation(
                    trials_group_by_folder_name[trial_name][state_no],
                    model_group_by_state[model_no])
               all_log_curves_of_this_model[model_no] = np.hstack([all_log_curves_of_this_model[model_no], one_log_curve_of_this_model])
            ax.plot(all_log_curves_of_this_model[model_no], linestyle="solid", label='state_'+ str(model_no), color=next(color))
        title = ('skill_identification' + trial_name)
        ax.set_title(title)
        if not os.path.isdir(figure_save_path + '/skill_identification_plot'):
            os.makedirs(figure_save_path + '/skill_identification_plot')
        fig.savefig(os.path.join(figure_save_path, 'skill_identification_plot', title + ".jpg"), format="jpg")
    fig.show()
