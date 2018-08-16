'''
@HongminWu April-21, 2018
This script for testing a new anomaly detector.
step-1: train the introspection models for each skill

step-2: save the (lik_of_all_states, zhat) and (log_lik, zhat) pair data, for each time step, 
    lik_of_all_states should be a 1*K vector, 
    log_lik should be computed with fw-bw algorithm 
    zhat is a value from (1,2,..., K), can be drived from vertebi algorithm 


step-3: I draw the zhat as the traget label, and concatenate all the lik_of_all_states with 
the same zhat value as training features to train a multiple label classifier.

step-4: modeling the log_lik with the same zhat value with a 1D Gaussian model

step-5: online: for each time step, we can get the lik_of_all_state and log_lik, and inference the correspounding zhat, and then calcualte the probability of p(log_lik | 1d-gaussian model). If the probability less than 50%, deem as anomaly.
'''

import os, sys, ipdb
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import HMM.hmm_for_baxter_using_only_success_trials.training_config as training_config
import HMM.hmm_for_baxter_using_only_success_trials.util as util
import HMM.hmm_for_baxter_using_only_success_trials.hmm_model_training as hmm_model_training
import bnpy
import hmmlearn
from HMM.hmm_for_baxter_using_only_success_trials.log_likelihood_incremental_calculator import interface 

DO_TRAINING = False

colors  = ['r', 'g', 'b', 'g', 'c', 'm', 'y', 'k']
markers = ['o', '+', '*', 's', 'x', '>', '<', '.']

def gpr_for_zid_and_zlog(zid, zlog):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(zid.reshape(-1,1), zlog)
    x_ = np.linspace(0, 1000, 100)
    y_mean, y_cov = gp.predict(x_[:, np.newaxis], return_cov = True)
    plt.plot(x_, y_mean, 'k', lw = 3, zorder = 9)
    plt.fill_between(x_, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha = 0.5, color = 'k')

def gmm_for_zid_and_zlog(zid, zlog, nplot):
    import itertools
    from scipy import linalg
    import matplotlib as mpl
    from sklearn import mixture
    X = np.vstack((np.array(zid).T, np.array(zlog).T)).T
    lowest_bic = np.infty
    bic = []
    n_components_range = range(3, 8)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange', 'r', 'g', 'b', 'gold'])
    clf = best_gmm
    bars = []
    
    # Plot the BIC scores
    spl = plt.subplot(nplot[0], nplot[1], nplot[2])
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range): (i + 1) * len(n_components_range)], width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(nplot[0], nplot[1], nplot[2]-1)
    Y_ = clf.predict(X)
    for i, (mean, color) in enumerate(zip(clf.means_, color_iter)):
        if clf.covariance_type == 'full':
            cov = clf.covariances_[i][:2,:2]
            
        elif clf.covariance_type == 'tied':
            cov = clf.covariances_[:2,:2]
            
        elif clf.covariance_type == 'diag':
            cov = np.diag(clf.covariances_[i][:2])
            
        elif clf.covariance_type == 'spherical':
            cov = np.eye(clf.means_.shape[1]) * clf.covariances_[i]
            
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: {0} model, {1} components'.format(clf.covariance_type, clf.n_components))
    plt.subplots_adjust(hspace=.35, bottom=.02)
    return Y_
    
def calc_threshold_from_logsumexp_of_specific_zhat(zUnique, zHatBySeq, logBySeq):
    for n, z in enumerate(zUnique):
        plt.figure(n)
        all_zid_by_z  = []
        all_zlog_by_z = []
        plt.subplot(2, 1, 1)
        for iSeq in range(len(zHatBySeq)):
            zHat = zHatBySeq[iSeq]
            log  = logBySeq[iSeq]
            zid  = np.where(zHat == z)[0]
            zlog = [log[i] for i in zid]
            #plt.scatter(zid, zlog, marker = markers[n], zorder = 10)
            all_zid_by_z  += zid.tolist()
            all_zlog_by_z += zlog
        plt.xlabel('time')
        plt.ylabel('log-likelihood')
        plt.title('All the log-likelihood values of hidden state: {0}'.format(z))
        #gpr_for_zid_and_zlog(np.array(all_zid_by_z), all_zlog_by_z)
        #gmm_for_zid_and_zlog(all_zid_by_z, all_zlog_by_z, nplot = [2, 1, 2])
    plt.show()
    
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
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        calculator = interface.get_calculator(model)
        log = []
        for i in range(x_train.shape[0]):
            sample = x_train[i,:].reshape(1,-1)
            logsumlik = calculator.add_one_sample_and_get_loglik(sample)
            log.append(logsumlik)
            if i == 0:
                first_val = logsumlik
        log = [first_val] + np.diff(log).tolist()
        _, state_sequence = model.decode(x_train)
        _zhat_prob_log = pd.DataFrame()        
        _zhat_prob_log['zhat'] =  state_sequence        
        _zhat_prob_log['log']  = log
        _zhat_prob_log.to_csv(training_config.model_save_path + '/zhat_log.csv', index = False)        
    else:
        zHatBySeq, probBySeq, logBySeq = model.decode(x_train, lengths=lengths)

        for nSeq in range(len(zHatBySeq)):
            model.show_single_sequence(nSeq, zhat_T = zHatBySeq[nSeq])
        _zhat = np.concatenate(zHatBySeq)
        _prob = np.concatenate(probBySeq)
        _log  = np.concatenate(logBySeq)
        _zhat_prob_log = pd.DataFrame()
        _zhat_prob_log['zhat'] =  _zhat
        #calc_threshold_from_logsumexp_of_specific_zhat(_zhat_prob_log['zhat'].unique(), zHatBySeq, logBySeq)
        _zhat_prob_log['log']  = _log
        for k in range(_prob.shape[1]):
            _zhat_prob_log['k_{0}'.format(k)] = _prob[:,k]
        _zhat_prob_log.to_csv(training_config.model_save_path + '/zhat_log.csv', index = False)
    return model, _zhat_prob_log

def test_anomaly_detection_through_hidden_state(model, threshold_dict):
    normial_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tag_7')
    normial_data_group_by_tag = util.get_anomaly_data_for_labelled_case(training_config, normial_data_path)
    temp = normial_data_group_by_tag.values()
    lengths = []
    for i in range(len(temp)):
        lengths.append(temp[i][1].shape[0])
        if i == 0:
            x = temp[i][1]
        else:
            x = np.concatenate((x, temp[i][1]), axis = 0)
            
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        zhats = model.predict(x)
        logliks = model.score(x)
        for i in range(len(x)):
            zhat = model.predict(x[i].reshape(1,-1))
            curr_loglik = model.score(x[i].reshape(1,-1))
            ipdb.set_trace()
            if curr_loglik < threshold_dict[zhat[0]]:
                print 'anomaly'
            else:
                print 'success'
                
    else:
        prev_sample = None
        for isampe in range(len(x)):
            if prev_sample is None:
                prev_sample = x[isampe]
                continue
            else:
                test_sample = np.vstack((prev_sample, x[isampe]))
                prev_sample = x[isampe]
            logSoftEv = model.get_emission_log_prob_matrix(test_sample)
            log_startprob = np.log(model.model.allocModel.get_init_prob_vector())
            log_transmat  = np.log(model.model.allocModel.get_trans_prob_matrix())        
            zHat =  model.runViterbiAlg(logSoftEv, log_startprob, log_transmat)
            try:
                fmsg, margPrObs = bnpy.allocmodel.hmm.HMMUtil.FwdAlg(np.exp(log_startprob), np.exp(log_transmat), np.exp(logSoftEv))
            except FloatingPointError:
                pass
            cur_loglik = np.log(margPrObs)
            if cur_loglik[0] > threshold_dict[zHat[0]]:
                print 'normal'
            else:
                print 'anomaly'
            
if __name__=="__main__":
    
    if DO_TRAINING:
        model, df = train_norminal_model()
    else:
        model = joblib.load(training_config.model_save_path + "/model_s%s.pkl"%(1,))
        df    = pd.read_csv(training_config.model_save_path + '/zhat_log.csv', sep=',')
    print training_config.model_save_path
    '''
    K = model.K
    k_id_list = ['k_{0}'.format(i) for i in range(K)]
    # for LSTM_FCN
    df[ ['zhat'] + k_id_list].to_csv('tag_3_x_train', header = False, index=False)
    df[ ['zhat'] + k_id_list].to_csv('tag_3_x_test', header = False, index=False)    
    '''

    # plot
    threshold_dict = {}
    for i, iz in enumerate(sorted(df['zhat'].unique().tolist())):
        plt.subplot(len(df['zhat'].unique().tolist()), 1, i+1)
        plt.plot(df['log'].loc[df['zhat'] == iz].values, marker = markers[i], color = colors[i], linestyle = 'None', )
        zlog = df['log'].loc[df['zhat'] == iz].values

        '''
        zlog_min = zlog.min()
        zlog_max = zlog.max()
        threshold = zlog_min - (zlog_max - zlog_min)/2
        '''
        
        zlog_mean = np.mean(zlog)
        zlog_var  = np.var(zlog)
        threshold = zlog_mean - 2.0 * zlog_var
        threshold_dict[iz] = threshold
        
        plt.axhline(threshold, color = 'r', linewidth=4, label='Threshold') 
        plt.axhline(zlog_mean, linestyle='--', color = 'black', linewidth=2, label='Mean')       
        plt.title('Concatenate all the log-likelihood values of hidden state {0}'.format(iz))
        plt.legend()
    plt.show()
    np.save(os.path.join(training_config.model_save_path, 'threshold.npy'), threshold_dict)
    test_anomaly_detection_through_hidden_state(model, threshold_dict)

    '''  
    plt.figure()
    plt.title('All logsumexp values in variable color')
    for i, iz in enumerate(df['zhat'].unique().tolist()):
        df['log'].loc[df['zhat'] == iz].plot(marker = markers[i], color = colors[i], linestyle = 'None', label = iz)
        plt.legend()
    '''
    plt.show()
    

    '''
    # for MLPClassifier 
    all_liklihood_data = df[k_id_list].values
    zhat               = df['zhat'].values
    MLPclf = MLPClassifier(solver='sgd', alpha=1e-5, tol=1e-9,  max_iter = 100000, random_state = 1)

    MLPclf.fit(all_liklihood_data, zhat)
    z_predict = MLPclf.predict(all_liklihood_data)
    from evaluate_metrics import plot_confusion_matrix
    plot_confusion_matrix.run(zhat, z_predict, str(df['zhat'].unique()))
    '''
