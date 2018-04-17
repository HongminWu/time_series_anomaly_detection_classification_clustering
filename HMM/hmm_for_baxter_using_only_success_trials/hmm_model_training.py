#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import (
    scale,
    normalize
)
import copy
import model_generation
import model_score
import util
import training_config
import matplotlib.pylab as plt
import pandas as pd
from matplotlib import cm
from matplotlib import colors
import ipdb

def run(model_save_path,
    model_type,
    model_config,
    score_metric,
    trials_group_by_folder_name):

    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)
    list_of_trials = trials_group_by_folder_name.values()

    trials_amount = len(trials_group_by_folder_name)

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    one_trial_data_group_by_state = list_of_trials[0]
    state_amount = len(one_trial_data_group_by_state)

    training_data_group_by_state = {}
    training_length_array_group_by_state = {}

    for state_no in range(1, state_amount+1):
        length_array = []
        for trial_no in range(len(list_of_trials)):
            length_array.append(list_of_trials[trial_no][state_no].shape[0])
            if trial_no == 0:
                data_tempt = list_of_trials[trial_no][state_no]
            else:
                data_tempt = np.concatenate((data_tempt,list_of_trials[trial_no][state_no]),axis = 0)
        training_data_group_by_state[state_no] = data_tempt
        training_length_array_group_by_state[state_no] = length_array

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    for state_no in range(1, state_amount+1):
        model_list = []
        train_data = training_data_group_by_state[state_no]
        lengths    = training_length_array_group_by_state[state_no]
        best_model, model_id = train_hmm_model(train_data, lengths)
        joblib.dump(
            best_model['model'],
            os.path.join(model_save_path, "model_s%s.pkl"%(state_no,))
        )

        joblib.dump(
            best_model['now_model_config'],
            os.path.join(
                model_save_path,
                "model_s%s_config_%s.pkl"%(state_no, model_id)
            )
        )

        joblib.dump(
            None,
            os.path.join(
                model_save_path,
                "model_s%s_score_%s.pkl"%(state_no, best_model['score'])
            )
        )

        train_report = [{util.get_model_config_id(i['now_model_config']): i['score']} for i in sorted_model_list]
        import json
        json.dump(
            train_report,
            open(
                os.path.join(
                    model_save_path,
                    "model_s%s_training_report.json"%(state_no)
                ), 'w'
            ),
            separators = (',\n', ': ')
        )

        # plot the hidden state sequence for each state
        print
        print
        print 'Finish fitting the posterior model -> Generating the hidden state sequence...'
        print
        print
        model = best_model['model']
        if model_type == 'hmmlearn\'s HMM':
            _, model.z = model.decode(X, algorithm="viterbi")

        elif model_type == 'BNPY\'s HMM':
            model.z = model.decode(X, lengths)

        elif model_type == 'PYHSMM\'s HMM':
            model.z = model.model.stateseqs[0]

        elif model_type == 'hmmlearn\'s GMMHMM':
            _, model.z = model.decode(X, algorithm="viterbi")

        else:
            print 'Sorry, this model cannot obtain the hidden state sequence'
            return

        # plt.close("all")
        # Xdf = pd.DataFrame(X) # plot the original multimodal signals
        # Xdf.plot()

        # im_data  = np.tile(model.z, 2)
        # cmap =cm.get_cmap('jet',np.max(model.z))
        # print np.unique(model.z)
        # ax.imshow(im_data[None], aspect='auto', interpolation='nearest', vmin = 0, vmax = np.max(model.z), cmap = cmap, alpha = 0.5)

        fig, ax = plt.subplots(nrows=1,ncols=1)
        trial_len = len(model.z) / trials_amount
        color=iter(cm.rainbow(np.linspace(0, 1, trials_amount)))
        zhat = []
        for iTrial in range(trials_amount):
           zSeq = model.z[iTrial*trial_len:(iTrial+1)*trial_len] 
           ax.plot(zSeq, color=next(color)) #, linewidth=2.0
           zhat.append(zSeq.tolist() + [zSeq[-1]])
        plt.show()
        zdf = pd.DataFrame(zhat)
        plt.title('The hidden state_sequence of state_%d' % (state_no))
        zdf.to_csv(model_save_path + '/zhat.csv', index = False)

def train_hmm_model(train_data, lengths):
    model_list = []
    lengths[-1] -=1
    model_generator = model_generation.get_model_generator(training_config.model_type_chosen, training_config.model_config)
    for model, now_model_config in model_generator:
        model = model.fit(train_data, lengths=lengths)  # n_samples, n_features
        score = model_score.score(training_config.score_metric, model, train_data, lengths)
        if score == None:
            print "scorer says to skip this model, will do"
            continue
        model_list.append({
            "model": model,
            "now_model_config": now_model_config,
            "score": score
        })
        print 'score:', score
        model_generation.update_now_score(score)        
    sorted_model_list = sorted(model_list, key=lambda x:x['score'])
    best_model = sorted_model_list[0]
    model_id = util.get_model_config_id(best_model['now_model_config'])
    return best_model, model_id
        

        
