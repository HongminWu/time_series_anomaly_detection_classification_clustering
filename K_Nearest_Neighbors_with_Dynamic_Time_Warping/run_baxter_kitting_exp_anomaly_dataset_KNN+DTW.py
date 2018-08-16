import os, sys
import numpy as np
import itertools
import collections
import matplotlib.pyplot as plt
from util import KnnDtw
from sklearn.externals import joblib
from visualizing_high_dimensional_data import plot_with_t_SNE
from HMM.hmm_for_baxter_using_only_success_trials import training_config
import ipdb

def plot_in_high_dimensional_space(x_train, x_test, y_train, y_test):
    x = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))    
    
    y = y.reshape(1, -1)[0]
    plot_with_t_SNE.run(x,y)
    

if __name__=='__main__':

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
        print('Error occured trying to read the file, please check the path: ' + TRAIN_TEST_DATASET_PATH)        
        
    # flatten the observation
    X = []
    for i in range(x_train.shape[0]):
        X.append(x_train[i].flatten('C')) # default 'C' flatten in row order, 'F' flatten in column order
    x_train = []
    x_train = np.array(X)
        
    X = []
    for i in range(x_test.shape[0]):
        X.append(x_test[i].flatten('C'))   # default 'C' flatten in row order, 'F' flatten in column order
    x_test = []
    x_test = np.array(X)
    labels = labels.tolist()
    
    classes = {}
    for num, className in enumerate(labels):
        classes[num] = className
    print classes

    #---
    #plot_in_high_dimensional_space(x_train, x_test, y_train, y_test)
    #print 'only for plotting test'
    #sys.exit()
    
    # Visualizing sample observations from the HAR dataset
    plt.figure(figsize=(11,37))
    colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD','#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

    for i, r in enumerate(np.random.randint(0, y_train.shape[0], 6).tolist()):
        plt.subplot(6,1,i+1)
        plt.plot(x_train[r][:], label=classes[y_train[r][0]], color=colors[i], linewidth=2)
        plt.legend(loc='upper left')
        plt.tight_layout()
    plt.show()

    # Model Performance
    try:
        model = joblib.load('KNN+DTW-anomaly_clf_model.pkl')
    except IOError as e:
        m = KnnDtw(n_neighbors=1, max_warping_window=10)
        m.fit(x_train, y_train)
        joblib.dump(m, 'KNN+DTW-anomaly_clf_model.pkl')
        model = joblib.load('KNN+DTW-anomaly_clf_model.pkl')
        
    y_pred, proba = model.predict(x_test)

    from evaluate_metrics import plot_confusion_matrix
    plot_confusion_matrix.run(y_test, y_pred, labels)
    
    # Time preformance
    '''
    import time
    time_taken = []
    windows = [1,2,5,10,50,100,500,1000,5000]

    for w in windows:
        begin = time.time()

        t = KnnDtw(n_neighbors=1, max_warping_window=w)
        t.fit(x_train, y_train)
        label, proba = t.predict(x_test)

        end = time.time()
        time_taken.append(end - begin)

    fig = plt.figure(figsize=(12,5))
    _ = plt.plot(windows, [t/400. for t in time_taken], lw=4)
    plt.title('DTW Execution Time with \nvarying Max Warping Window')
    plt.ylabel('Execution Time (seconds)')
    plt.xlabel('Max Warping Window')
    plt.xscale('log')    
    '''
