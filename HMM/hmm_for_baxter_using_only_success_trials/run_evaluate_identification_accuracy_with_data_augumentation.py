'''
@HongminWu April, 16, 2018
function: evaluate the perfermance of data augumentation methods;
The implementatiing steps:
0. load the dataset and split it into training(20%) and testing(80%) 
1. training the model with variable data augumation parameters
2. testing the identification accuracy w.r.t the trained model
'''
import os, sys, glob
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import shutil
import training_config
import hmm_model_training
import util
import generate_synthetic_data
import ipdb

def load_dataset(testing_ratio=0.8):
    path = os.path.join(training_config.anomaly_data_path, 'anomalies', 'TOOL_COLLISION')
    anomaly_data_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, path)
    data_in_list = []
    for filename in anomaly_data_group_by_folder_name:
        data_in_list.append(anomaly_data_group_by_folder_name[filename][1])
    length    = len(data_in_list)
    test_len  = int(testing_ratio * length)
    x_test  = data_in_list[:test_len]
    x_train = data_in_list[test_len:]
    print ('dataset_len:%s' %len(data_in_list))
    print ('test_len %s; train_len: %s' % (len(x_test),len(x_train)))
    return x_train, x_test

def run():
    output_dir = os.path.join(training_config.anomaly_data_path, 'synthetic_data')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    x_train, x_test = load_dataset(testing_ratio=0.9)
    
    # calculate the threshold for identification based on real training trials
    print 'calculate the threshold for identification based on real training trials'
    lengths = []
    for idx in range(len(x_train)):
        lengths.append(x_train[idx].shape[0])
        if idx == 0:
            train_data = x_train[idx]
        else:
            train_data = np.concatenate((train_data, x_train[idx]), axis=0)
    best_model, model_id = hmm_model_training.train_hmm_model(train_data, lengths)
    all_log_curves = []
    for itrial in range(len(x_train)):
        one_log_curve = util.fast_log_curve_calculation(x_train[itrial], best_model['model'])
        all_log_curves.append(one_log_curve)
    np_matrix_of_all_log_curves = np.matrix(all_log_curves)
    plt.figure()
    plt.subplot(211)
    plt.title('All log likelihood curves and calculated threshold')
    for no in range(np_matrix_of_all_log_curves.shape[0]):
        plt.plot(np_matrix_of_all_log_curves[no].tolist()[0], linestyle='--', color='gray', label = 'testing_trial')
    colors = iter(cm.rainbow(np.linspace(0,1,5)))
    for c in np.arange(0, 10, 2):
        plt.plot((np_matrix_of_all_log_curves.mean(0) - c * np_matrix_of_all_log_curves.std(0)).tolist()[0], label = 'mean-%s*std' %(c,), linestyle='solid', color = next(colors))
    plt.legend(loc='best')
    threshold_c = 2
    threshold_for_log_likelihood  = (np_matrix_of_all_log_curves.mean(0) - threshold_c * np_matrix_of_all_log_curves.std(0)).tolist()[0]

    # train the model with data augmentation and test it
    print "train the model with data augmentation and test it"
    num_data_list = range(2, 20, 2)
    acc_list      = []
    for num_data in num_data_list:
        old_files = glob.glob(os.path.join(output_dir, '*'))
        for old_file in old_files: os.remove(old_file)
        for i in range(len(x_train)):
            print ('Generating synthetic data from real_{0}'.format(i))
            df = pd.DataFrame(x_train[i], columns=training_config.interested_data_fields)
            df.to_csv(os.path.join(output_dir, 'real_' + str(i) + '.csv'))
            generate_synthetic_data.run_finite_differece_matrix(df=df, num_data = num_data, csv_save_path=output_dir, trial_name='real_'+str(i))
        anomaly_data_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, output_dir)
        list_of_trials = anomaly_data_group_by_folder_name.values()
        lengths = []
        for idx in range(len(list_of_trials)):
            lengths.append(list_of_trials[idx][1].shape[0])
            if idx == 0:
                train_data = list_of_trials[idx][1]
            else:
                train_data = np.concatenate((train_data, list_of_trials[idx][1]), axis=0)
        best_model, model_id = hmm_model_training.train_hmm_model(train_data, lengths)

        FP = 0.0
        for itest in range(len(x_test)):
            one_log_curve = util.fast_log_curve_calculation(x_test[itest], best_model['model'])
            if one_log_curve[-1] > threshold_for_log_likelihood[-1]: FP +=1 
        idfyRate = FP / len(x_test)
        acc_list.append(idfyRate)
        print idfyRate
    plt.subplot(212)
    plt.title('accuracy vs num synthetic data')
    plt.plot(num_data_list, acc_list, 'o-')
    plt.show()
            
if __name__=="__main__":
    sys.exit(run())
