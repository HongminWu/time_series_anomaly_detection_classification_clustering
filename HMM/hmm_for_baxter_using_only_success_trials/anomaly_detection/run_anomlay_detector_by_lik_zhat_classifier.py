'''
@HongminWu April-21, 2018
This script for testing a new anomaly detector.
step-1: train the normial models for each skill

step-2: save the (lik_of_all_states, zhat) and (log_lik, zhat) pair data, for each time step, 
    lik_of_all_states should be a 1*K vector, 
    log_lik should be computed with fw-bw algorithm 
    zhat is a value from (1,2,..., K), can be drived from vertebi algorithm 


step-3: I draw the zhat as the traget label, and concatenate all the lik_of_all_states with 
the same zhat value as training features to train a multiple label classifier.

step-4: modeling the log_lik with the same zhat value with a 1D Gaussian model

step-5: online: for each time step, we can get the lik_of_all_state and log_lik, and inference the correspounding zhat, and then calcualte the probability of p(log_lik | 1d-gaussian model). If the probability less than 50%, deem as anomaly.
'''

import os, ipdb
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier        
from scipy.special import logsumexp
import HMM.hmm_for_baxter_using_only_success_trials.training_config as training_config
import HMM.hmm_for_baxter_using_only_success_trials.util as util
import HMM.hmm_for_baxter_using_only_success_trials.hmm_model_training as hmm_model_training

DO_TRAINING = False
def train_norminal_model():
    normial_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tag_3')
    normial_data_group_by_tag = util.get_anomaly_data_for_labelled_case(training_config, normial_data_path)
    temp = normial_data_group_by_tag.values()
    lengths = []
    for i in range(len(temp)):
        lengths.append(temp[i][1].shape[0])
        if i == 0:
            x_train = temp[i][1]
        else:
            x_train = np.concatenate((x_train, temp[i][1]), axis = 0)
    best_model, model_id = hmm_model_training.train_hmm_model(x_train, lengths)
    if not os.path.isdir(training_config.model_save_path):
        os.makedirs(training_config.model_save_path)    
    joblib.dump(
        best_model['model'],
        os.path.join(training_config.model_save_path, "model_s%s.pkl"%(1,)))

    model = best_model['model']
    lik_of_all_states, zhat = model.decode(x_train, lengths=lengths)
    lik_zhat_df = pd.DataFrame(lik_of_all_states, columns=['k_{0}'.format(col) for col in range(lik_of_all_states.shape[1])])
    log_curve = [logsumexp(lik_of_all_states[i]) for i in range(len(lik_of_all_states))]
    lik_zhat_df['log'] = log_curve
    lik_zhat_df['zhat'] = zhat
    lik_zhat_df.to_csv(training_config.model_save_path + '/lik_zhat_df.csv', index = False)
    return model, lik_zhat_df

if __name__=="__main__":
    if DO_TRAINING:
        model, df = train_norminal_model()
    else:
        df    = pd.read_csv(training_config.model_save_path + '/lik_zhat_df.csv', sep=',')
        model = joblib.load(training_config.model_save_path + "/model_s%s.pkl"%(1,))
    K = model.K
    k_id_list = ['k_{0}'.format(i) for i in range(K)]
    all_liklihood_data = df[k_id_list].values
    all_log_data       = df['log'].values
    zhat = df['zhat']
    MLPclf = MLPClassifier(solver='sgd', alpha=1e-5, tol=1e-9,  max_iter = 100000, random_state = 1)
    MLPclf.fit(all_liklihood_data, zhat)
    z_predict = MLPclf.predict(all_liklihood_data)
    from evaluate_metrics import plot_confusion_matrix
    plot_confusion_matrix.run(zhat, z_predict, str(np.unique(zhat).tolist()))
    import  matplotlib.pyplot as plt
    plt.show()
