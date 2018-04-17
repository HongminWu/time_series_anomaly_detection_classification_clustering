import sys
import os
import pandas as pd
import numpy as np
import shutil
import training_config
import hmm_model_training
import generate_synthetic_data
import util
import itertools
from sklearn.externals import joblib
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ipdb

DO_TRAINING = True
CLASSIFIER_TYPE_OPTIONS = ['HMMClassifier', 'MLPClassifierHiddenSeq']
CLASSIFIER_TYPE = CLASSIFIER_TYPE_OPTIONS[0]

def train_model(x_train, y_train, class_names):
    '''
    function: train all the anomalious models
    '''
    for idx, model_name in enumerate(class_names):
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
        best_model, model_id = hmm_model_training.train_hmm_model(train_data, lengths)
        anomaly_model_path = os.path.join(training_config.anomaly_model_save_path, 
                                                   model_name, 
                                                   training_config.config_by_user['data_type_chosen'], 
                                                   training_config.config_by_user['model_type_chosen'], 
                                                   )
        
        if not os.path.isdir(anomaly_model_path):
            os.makedirs(anomaly_model_path)
        joblib.dump(
            best_model['model'],
            os.path.join(anomaly_model_path, "model_s%s.pkl"%(1,))
        )
        
        model = best_model['model']
        zhidden = model.decode(train_data, lengths=lengths)
        zhat = []
        trial_len = len(zhidden) / len(indices)
        for iTrial in range(len(indices)):
            zseq = zhidden[iTrial * trial_len : (iTrial + 1) * trial_len]
            zhat.append(zseq.tolist() + [zseq[-1]])
        zdf = pd.DataFrame(zhat)
        zdf.to_csv(anomaly_model_path + '/zhat.csv', index = False)        

class HMMClassifier():

    def __init__(self):
        self.__name__ = 'HMMClassifier'

    def predict_proba(self, x_test, class_names):
        # load trained anomaly models
        anomaly_model_group_by_label = {}
        for fo in class_names:
            anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                                   fo,
                                                   training_config.config_by_user['data_type_chosen'],
                                                   training_config.config_by_user['model_type_chosen'],
                                                   )
            try:
                anomaly_model_group_by_label[fo] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))
            except IOError:
                print 'anomaly model of  %s not found'%(fo,)
                raw_input("sorry! cann't load the anomaly model")
                continue
        predict_score = []
        for i in range(len(x_test)):
            temp_loglik = []
            for model_label in class_names:
                one_log_curve_of_this_model = util.fast_log_curve_calculation(x_test[i], anomaly_model_group_by_label[model_label])
                temp_loglik.append(one_log_curve_of_this_model[-1])
            temp_score = temp_loglik / np.sum(temp_loglik)
            predict_score.append(temp_score)
        return np.array(predict_score)
    
    def predict(self, x_test, class_names):
        # load trained anomaly models
        anomaly_model_group_by_label = {}
        for fo in class_names:
            anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                                   fo,
                                                   training_config.config_by_user['data_type_chosen'],
                                                   training_config.config_by_user['model_type_chosen'],
                                                   )
            try:
                anomaly_model_group_by_label[fo] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))
            except IOError:
                print 'anomaly model of  %s not found'%(fo,)
                raw_input("sorry! cann't load the anomaly model")
                continue
        y_pred = []
        for i in range(len(x_test)):
            # plot
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # from matplotlib.pyplot import cm
            # color = iter(cm.rainbow(np.linspace(0, 1, len(anomaly_model_group_by_label))))
            calc_cofidence_resourse = []
            for idx, model_label in enumerate(class_names):
                one_log_curve_of_this_model = util.fast_log_curve_calculation(x_test[i], 
                                                                            anomaly_model_group_by_label[model_label])
                calc_cofidence_resourse.append({
                    'model_idx'         : idx,
                    'model_label'       : model_label,
                    'culmulative_loglik': one_log_curve_of_this_model[-1],
                    })
            #     c = next(color)
            #     plot_line, = ax.plot(one_log_curve_of_this_model, linestyle="solid", color = c)
            #     plot_line.set_label(model_label)
            # title = ('Anomaly_identification for ' + fo)
            # ax.set_title(title)
            # plt.savefig('images/'+str(i) + '.png', dpi=120)
            sorted_list = sorted(calc_cofidence_resourse, key=lambda x:x['culmulative_loglik'])
            optimal_result = sorted_list[-1]
            classified_idx = optimal_result['model_idx']
            y_pred.append(classified_idx)
        return y_pred

class MLPClassifierHiddenSeq():
    def __init__(self):
        self.__name__ = 'MLPClassifierHiddenSeq'

    def fit(self, class_names):
        from sklearn.neural_network import MLPClassifier
        x_train = []
        y_train = []
        for idx, fo in enumerate(class_names):
            anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                                  fo,
                                                  training_config.config_by_user['data_type_chosen'],
                                                  training_config.config_by_user['model_type_chosen'],
                                                 )
            try:
                zdf  = pd.read_csv(anomaly_model_path + '/zhat.csv')
                temp = zdf.values
                if idx == 0:
                    x_train = temp
                else:
                    x_train = np.concatenate((x_train, temp), axis = 0 )
                y_train += [idx] * temp.shape[0]
            except IOError:
                print 'hidden state sequence of  %s not found'%(fo,)
                raw_input("sorry! cann't load the hidden state sequence")
                sys.exit()
        MLPclf = MLPClassifier(solver='sgd', alpha=1e-5, tol=1e-9, hidden_layer_sizes = (100,100,100), max_iter = 1000, random_state = 1)
        MLPclf.fit(x_train, y_train)
        self.MLPclf = MLPclf

    def predict(self, x_test, class_names):
        anomaly_model_group_by_label = {}
        y_pred = []
        for fo in class_names:
            anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                                   fo,
                                                   training_config.config_by_user['data_type_chosen'],
                                                   training_config.config_by_user['model_type_chosen'],
                                                   )
            try:
                anomaly_model_group_by_label[fo] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))
            except IOError:
                print 'anomaly model of  %s not found'%(fo,)
                raw_input("sorry! cann't load the anomaly model")
                sys.exit()
        for i in range(x_test.shape[0]):
            all_zhat_for_one_test = []
            for f in class_names:
                zhat = anomaly_model_group_by_label[f].decode(x_test[i], lengths=len(x_test[i])-1)
                zhat_plus = zhat.tolist() + [zhat[-1]]
                all_zhat_for_one_test.append(zhat_plus)
#            print self.MLPclf.predict_log_proba(all_zhat_for_one_test)
#            max_in_row = np.amax(self.MLPclf.predict_proba(all_zhat_for_one_test), axis = 1)
#            y_pred.append(np.argmax(max_in_row))

def run():
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
    if DO_TRAINING: train_model(x_train, y_train, class_names)
    from evaluate_metrics import (plot_confusion_matrix, plot_roc_for_multiple_classes, \
                                      plot_precision_recall)
    if CLASSIFIER_TYPE == 'HMMClassifier':
        # for confusion matrix
        classifier = HMMClassifier()
        y_pred = classifier.predict(x_test, class_names)
        plot_confusion_matrix.run(y_test, y_pred, class_names)
        # for plot roc
        y_score = classifier.predict_proba(x_test, class_names)
        plot_roc_for_multiple_classes.run(y_score, y_test, class_names)
        # for plot precision-recall curve
        plot_precision_recall.run(y_score, y_test, class_names)
        
    elif CLASSIFIER_TYPE == 'MLPClassifierHiddenSeq':
        # for confusion matrix
        classifier = MLPClassifierHiddenSeq()
        classifier.fit(x_train,class_names)
        y_pred = classifier.predict(x_test, class_names)
        plot_confusion_matrix.run( y_test, y_pred, class_names)
    else:
        print ("sorry, we don't have such a classifier %s" % CLASSIFIER_TYPE)

    plt.show()
        
if __name__ == '__main__':
    sys.exit(run())
    
