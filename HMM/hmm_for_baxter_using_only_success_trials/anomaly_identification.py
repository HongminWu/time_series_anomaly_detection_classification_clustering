#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import util
import training_config
import pandas as pd
import random
import ipdb


def run(anomaly_data_path_for_testing,
        model_save_path,
        figure_save_path,):
    '''
        1. load all the anomalous trained models
        2. load testing anomaly data
        3. plot the log-likelihood wrt each model and plot in a same figure
    '''

    # load trained anomaly models
    anomaly_model_group_by_label = {}
    folders = os.listdir(training_config.anomaly_model_save_path)
    for fo in folders:
        path = os.path.join(training_config.anomaly_data_path, fo)
        if not os.path.isdir(path):
            continue
        anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                               fo,
                                               training_config.config_by_user['data_type_chosen'],
                                               training_config.config_by_user['model_type_chosen'],
                                               training_config.model_id)
        try:
            anomaly_model_group_by_label[fo] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))
        except IOError:
            print 'anomaly model of  %s not found'%(fo,)
            continue

    # one-folder
	confuse_matrix = {}
    folders = os.listdir(anomaly_data_path_for_testing)
    for fo in folders:
        predict_class = []
        data_path = os.path.join(anomaly_data_path_for_testing, fo)
        if not os.path.isdir(path):
            continue
        anomaly_testing_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, data_path)
		# one-file
        for trial_name in anomaly_testing_group_by_folder_name:
        
            '''
            #plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            from matplotlib.pyplot import cm
            color = iter(cm.rainbow(np.linspace(0, 1, len(anomaly_model_group_by_label))))
            '''

            calc_cofidence_resourse = []
            for model_label in anomaly_model_group_by_label:
                one_log_curve_of_this_model = util.fast_log_curve_calculation(
                        anomaly_testing_group_by_folder_name[trial_name][1],
                        anomaly_model_group_by_label[model_label])
#                one_predict_proba_of_this_state = anomaly_model_group_by_label[model_label].predict_proba(anomaly_testing_group_by_folder_name[trial_name][1]) # HDPHSMM haven't implemented this
#                one_hidden_stateSeq_of_this_state =  anomaly_model_group_by_label[model_label].decode(anomaly_testing_group_by_folder_name[trial_name][1],len(anomaly_testing_group_by_folder_name[trial_name][1])-1)
                calc_cofidence_resourse.append({
                    'model_label'       : model_label,
                    'culmulative_loglik': one_log_curve_of_this_model[-1],
                    'loglik_curve'      : one_log_curve_of_this_model,
#                    'predict_proba'     : one_predict_proba_of_this_state,
#                    'hidden_stateSeq'   : one_hidden_stateSeq_of_this_state
                })

            '''
                #--plot
                c = next(color)
                plot_line, = ax.plot(one_log_curve_of_this_model, linestyle="solid", color = c)
                plot_line.set_label(model_label)
            title = ('Anomaly_identification for ' + fo)
            ax.set_title(title)
            '''

            sorted_list = sorted(calc_cofidence_resourse, key=lambda x:x['culmulative_loglik']) # from small to large
            optimal_result = sorted_list[-1]
            classified_model = optimal_result['model_label']
            predict_class.append(classified_model)

            all_log_curves_of_this_state, threshold, confidence = get_confidence_of_identification(optimal_result)
            
            '''
            if confidence < 0.0:
                df = pd.DataFrame(anomaly_testing_group_by_folder_name[trial_name][1], columns=training_config.interested_data_fields)
                id = random.randint(1000,10000)
                _name = 'unknown_anomaly_' + str(id)
                unknown_anomaly_path = os.path.join(training_config.anomaly_data_path, _name)
                os.makedirs(unknown_anomaly_path)
                print 'generated a new anomaly:' + _name
                print '*\n'*5
                print 'synthetic data generation'
                import generate_synthetic_data
                generate_synthetic_data.run_finite_differece_matrix(df=df,   num_data = 5, csv_save_path=unknown_anomaly_path, trial_name)
            '''
                
            #--plot
            '''
            for no_trial in range(len(all_log_curves_of_this_state)):
                ax.plot(all_log_curves_of_this_state[no_trial], linestyle= '--', color = 'gray', label = 'trials')
            ax.plot(threshold.tolist()[0], linestyle='--', color='gold', label='threshold')
            ax.legend(loc='upper left')
            ax.text(20,optimal_result['culmulative_loglik']/2, optimal_result['model_label'] + ': ' + str(confidence),
                    ha = 'center', va = 'baseline',
                    bbox=dict(boxstyle="round",
                              ec=(1., 0.6, 0.6),
                              fc=(1., 0.9, 0.9),)
                    )
            if not os.path.isdir(figure_save_path + '/anomaly_identification_plot'):
                    os.makedirs(figure_save_path + '/anomaly_identification_plot')
            fig.savefig(os.path.join(figure_save_path, 'anomaly_identification_plot', fo + ":" + trial_name + ".jpg"), format="jpg")
#            fig.show(1)
#           raw_input('testing another trial?? Please any key to continue')
            '''
        print 'Finish testing: '+ fo + '\n'
        confuse_matrix[fo] = predict_class

    _items  = confuse_matrix.keys()
    _matrix = np.identity(len(_items))
    for row in _items:
        for col in _items:
            r = _items.index(row)
            c = _items.index(col)
            _matrix[r, c] = confuse_matrix[row].count(col)
    print 'print the confusion matrix...'
    print _items
    print _matrix

def get_confidence_of_identification(optimal_result):
    confidence_metric = ['culmulative_loglik_divide_by_the_culmulative_mean_loglik',
                        'posterior_of_gmm_model',
                        'calc_kullback_leibler_divergence_of_predict_proba',
                        'hamming_distance_of_hidden_state_sequence',
                        ]
    CONF_TYPE = confidence_metric[0]

    anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                           optimal_result['model_label'],
                                           training_config.config_by_user['data_type_chosen'],
                                           training_config.config_by_user['model_type_chosen'],
                                           training_config.model_id)

    if CONF_TYPE == 'culmulative_loglik_divide_by_the_culmulative_mean_loglik':
        c_value = 5
        all_log_curves_of_this_state = joblib.load(os.path.join(anomaly_model_path, 'all_log_curves_of_this_state.pkl'))
        std_of_log_likelihood        = joblib.load(os.path.join(anomaly_model_path, 'std_of_log_likelihood.pkl'))
        np_matrix_traj_by_time = np.matrix(all_log_curves_of_this_state)
        mean_of_log_curve = np_matrix_traj_by_time.mean(0)
        threshold = mean_of_log_curve - std_of_log_likelihood[1]

        confidence = optimal_result['culmulative_loglik'] - threshold.tolist()[0][-1]
        return all_log_curves_of_this_state, threshold, confidence

    elif CONF_TYPE == 'posterior_of_gmm_model':
        #load -> build a hmm model -> calculate the probability of testing sequence
        all_log_curves_of_this_state = joblib.load(os.path.join(anomaly_model_path, 'all_log_curves_of_this_state.pkl'))
        data = np.ndarray([])
        for icurve in range(len(all_log_curves_of_this_state)):
            tVal        = range(len(all_log_curves_of_this_state[icurve]))
            feature     = all_log_curves_of_this_state[icurve]
            data_points = np.vstack([np.array(tVal), feature]).T
            if icurve == 0:
                data = data_points
            else:
                data = np.vstack([data, data_points])

        # fit a gmm model
        from sklearn import mixture
        gmm = mixture.GaussianMixture(n_components = 5, covariance_type = 'diag').fit(data)

        tVal = range(optimal_result['loglik_curve'].shape[0])
        testing_data = np.vstack([np.array(tVal), optimal_result['loglik_curve']]).T
        confidence = gmm.score(testing_data)
        return all_log_curves_of_this_state, None,  confidence


    elif CONF_TYPE == 'calc_kullback_leibler_divergence_of_predict_proba':
        print 'obsoleted'
        pass
        from scipy.stats import entropy
        # average_predict_proba
        average_predict_proba = joblib.load(os.path.join(anomaly_model_path, 'average_predict_proba.pkl'))
        testing_predict_proba = optimal_result['predict_proba']
        confidence = 0.0
        for iObs in range(len(testing_predict_proba)):
            confidence += entropy(testing_predict_proba[iObs,:], average_predict_proba[1][iObs,:])
        return None, None, confidence


    elif CONF_TYPE == 'hamming_distance_of_hidden_state_sequence':
        # hidden_state_sequence_of_training_trials
        hidden_stateSeq  = joblib.load(os.path.join(anomaly_model_path, 'hidden_stateSeq.pkl'))
        hidden_stateSeq  = np.append(hidden_stateSeq, [hidden_stateSeq[-1]]) # add one item, because for autoregressive model, I had deleted one data point
        totalLen         = len(hidden_stateSeq)

        testing_stateSeq = optimal_result['hidden_stateSeq']
        testing_stateSeq = np.append(testing_stateSeq, [testing_stateSeq[-1]])
        tLen             = len(testing_stateSeq)
        hidden_stateSeq  = hidden_stateSeq.reshape(totalLen/tLen, tLen)

        from scipy.spatial.distance import hamming
        confidence = 0
        for iTrial in range(totalLen/tLen):
            confidence += hamming(testing_stateSeq, hidden_stateSeq[iTrial,:])
        return None, None, confidence
    else:
        print ("without the confidence_metric as: " + CONF_TYPE)
        pass
