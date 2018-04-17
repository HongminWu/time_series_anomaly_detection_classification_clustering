from HMM.hmm_for_baxter_using_only_success_trials import training_config, run_anomaly_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import plot_confusion_matrix
import os,sys
import numpy as np
import ipdb

# load the train/test/labels file
TRAIN_TEST_DATASET_PATH = training_config.anomaly_data_path
x_path       = os.path.join(TRAIN_TEST_DATASET_PATH, "X.npy")
y_path       = os.path.join(TRAIN_TEST_DATASET_PATH, "y.npy")
labels_path  = os.path.join(TRAIN_TEST_DATASET_PATH,  "labels_list.npy")
try:
    x       = np.load(x_path)
    y       = np.load(y_path)
    labels  = np.load(labels_path)
except IOError:
    print ('Error occured trying to read the file, please check the path: ' + TRAIN_TEST_DATASET_PATH)
    sys.exit()

 # transpose the observation to n_samples * n_features
X = []
for i in range(x.shape[0]):
    X.append(x[i].T) 
x = np.array(X)
y =  y.reshape(-1,)
class_names = labels.tolist()

skf = StratifiedKFold(n_splits = 10)
all_conf_mat = None
for train_index, test_index in skf.split(x, y):
    print ("%s %s" %(train_index, test_index))
    x_train, y_train, x_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index]
    run_anomaly_classification.train_model(x_train, y_train, class_names)
    y_pred = run_anomaly_classification.predict(x_test, class_names)
    conf_mat = confusion_matrix(y_test, y_pred)
    if all_conf_mat is None:
        all_conf_mat = conf_mat
    else:
        all_conf_mat += conf_mat

import matplotlib.pyplot as plt
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(all_conf_mat, classes=class_names, title='nonParametricHMM: Confusion Matrix after 10-fold Cross Validation')
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(all_conf_mat, classes=class_names, normalize=True, title='nonParametricHMM: Confusion Matrix after 10-fold Cross Validation')
plt.savefig('images/nonParametricHMMConfusionMatrixafter10-foldCrossValidation.png')
plt.show()
    
    
