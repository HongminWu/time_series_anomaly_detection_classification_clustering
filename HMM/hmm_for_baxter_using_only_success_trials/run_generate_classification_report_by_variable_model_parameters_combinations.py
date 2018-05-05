import sys
import os
import pandas as pd
import numpy as np
import shutil
import training_config
import hmm_model_training
import util
import itertools
from sklearn.externals import joblib
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger('HongminWu-logging')
logger.setLevel(logging.INFO)
hdlr = logging.FileHandler('../../log/bnpy_classification_report.log') # , mode='w')       
hdlr.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(hdlr)
import ipdb

def generate_performance_logging_report_with_varible_model_parameters():
    import model_generation
    # load the train/test/labels file
    TRAIN_TEST_DATASET_PATH = training_config.anomaly_data_path
    x_train_path = os.path.join(TRAIN_TEST_DATASET_PATH, "X_train.npy")
    y_train_path = os.path.join(TRAIN_TEST_DATASET_PATH, "y_train.npy")
    x_test_path  = os.path.join(TRAIN_TEST_DATASET_PATH,  "X_test.npy")
    y_test_path  = os.path.join(TRAIN_TEST_DATASET_PATH,  "y_test.npy")
    labels_path  = os.path.join(TRAIN_TEST_DATASET_PATH,  "labels_list.npy")
    try:
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_test  = np.load(x_test_path)
        y_test  = np.load(y_test_path)
        labels  = np.load(labels_path)
    except IOError:
        print ('Error occured trying to read the file, please check the path: ' + TRAIN_TEST_DATASET_PATH)
        sys.exit()
    x_train =  x_train.transpose((0,2,1))
    x_test  =  x_test.transpose((0,2,1))    
    y_train =  y_train.reshape(-1,).tolist()
    y_test  =  y_test.reshape(-1,).tolist()
    class_names = labels.tolist()

    train_data_by_class = {}
    train_lengths_by_class = {}
    for idx, class_name in enumerate(class_names):
        indices = [i for i, label in enumerate(y_train) if label == idx]
        train_data = x_train[indices] 
        lengths = []
        for i in range(len(train_data)):
            lengths.append(train_data[i].shape[0])
            if i == 0:
                data_tempt = train_data[i]
            else:
                data_tempt = np.concatenate((data_tempt, train_data[i]), axis=0)
        train_data = data_tempt
        train_data_by_class[class_name] = train_data
        lengths[-1] -= 1
        train_lengths_by_class[class_name] = lengths

    model_generator = model_generation.get_model_generator(training_config.model_type_chosen, training_config.model_config)
    for model, now_model_config in model_generator:
        logger.info(now_model_config)
        model_collection_for_all_classes = {}
        for idx, _name in enumerate(class_names):
            fitted_model = model.fit(train_data_by_class[_name], lengths=train_lengths_by_class[_name])  # n_samples, n_features
            # --- dump model and load it, confuse on this, but it works
            anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,'temp_classification_report_model', _name)
            if not os.path.isdir(anomaly_model_path):
                os.makedirs(anomaly_model_path)
            joblib.dump( fitted_model, os.path.join(anomaly_model_path, "model_s%s.pkl"%(1,)))
            model_collection_for_all_classes[_name] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))

        y_pred = []
        for i in range(len(x_test)):
            calc_cofidence_resourse = []
            for idx, model_label in enumerate(class_names):
                one_log_curve_of_this_model = util.fast_log_curve_calculation(x_test[i], model_collection_for_all_classes[model_label])
                calc_cofidence_resourse.append({
                    'model_idx'         : idx,
                    'model_label'       : model_label,
                    'culmulative_loglik': one_log_curve_of_this_model[-1],
                    })
            sorted_list = sorted(calc_cofidence_resourse, key=lambda x:x['culmulative_loglik'])
            optimal_result = sorted_list[-1]
            classified_idx = optimal_result['model_idx']
            y_pred.append(classified_idx)
        # for confusion matrix
        _clf_report = classification_report(y_test, y_pred, target_names=[l for l in class_names])
        logger.info(_clf_report)
        
if __name__ == '__main__':
    logger.info("-------------------------generate_classification_performance_logging_report_with_varible_model_parameters---------------------------------------------")
    sys.exit(generate_performance_logging_report_with_varible_model_parameters())
