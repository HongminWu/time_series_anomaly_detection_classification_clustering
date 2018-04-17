import numpy as np
import itertools
import collections
import matplotlib.pyplot as plt
from util import KnnDtw, ProgressBar

if __name__=='__main__':
    
    #TEST-SyntheticData: Measuring the DTW distance
    time = np.linspace(0,20,1000)
    amplitude_a = 5*np.sin(time)
    amplitude_b = 3*np.sin(time + 1)
    m = KnnDtw()
    distance = m._dtw_distance(amplitude_a, amplitude_b)
    fig = plt.figure(figsize=(12,4))
    _ = plt.plot(time, amplitude_a, label='A')
    _ = plt.plot(time, amplitude_b, label='B')
    _ = plt.title('DTW distance between A and B is %.2f' % distance)
    _ = plt.ylabel('Amplitude')
    _ = plt.xlabel('Time')
    _ = plt.legend()
    plt.show()

    #TEST-SyntheticData: Compute the distance between each pair of two collections of inputs
    disMx = m._dist_matrix(np.random.random((4,50)), np.random.random((4,50)))
    print '\n'
    print disMx

    #RealData-Human Activity Recognition Dataset
    # Import the HAR dataset
    x_train_file = open('data/UCI-HAR-Dataset/train/X_train.txt', 'r')
    y_train_file = open('data/UCI-HAR-Dataset/train/y_train.txt', 'r')

    x_test_file = open('data/UCI-HAR-Dataset/test/X_test.txt', 'r')
    y_test_file = open('data/UCI-HAR-Dataset/test/y_test.txt', 'r')

    # Create empty lists
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Mapping table for classes
    labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',
              4:'SITTING', 5:'STANDING', 6:'LAYING'}

    # Loop through datasets
    for x in x_train_file:
        x_train.append([float(ts) for ts in x.split()])

    for y in y_train_file:
        y_train.append(int(y.rstrip('\n')))

    for x in x_test_file:
        x_test.append([float(ts) for ts in x.split()])

    for y in y_test_file:
        y_test.append(int(y.rstrip('\n')))

    # Convert to numpy for efficiency
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Visualizing sample observations from the HAR dataset
    plt.figure(figsize=(11,7))
    colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
              '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

    for i, r in enumerate([0,27,65,100,145,172]):
        plt.subplot(3,2,i+1)
        plt.plot(x_train[r][:100], label=labels[y_train[r]], color=colors[i], linewidth=2)
        plt.xlabel('Samples @50Hz')
        plt.legend(loc='upper left')
        plt.tight_layout()
    plt.show()

    # Model Performance
    skip_step = 20
    m = KnnDtw(n_neighbors=1, max_warping_window=10)
    m.fit(x_train[::skip_step], y_train[::skip_step]) # from beginning to end with step skip_step
    label, proba = m.predict(x_test[::skip_step])

    #evaluate the results
    from sklearn.metrics import classification_report, confusion_matrix
    print classification_report(label, y_test[::skip_step],
                                target_names=[l for l in labels.values()])

    conf_mat = confusion_matrix(label, y_test[::skip_step])

    fig = plt.figure(figsize=(6,6))
    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c>0:
                plt.text(j-.2, i+.1, c, fontsize=16)

    cb = fig.colorbar(res)
    plt.title('Confusion Matrix')
    _ = plt.xticks(range(6), [l for l in labels.values()], rotation=90)
    _ = plt.yticks(range(6), [l for l in labels.values()])
    plt.show()

    # Time preformance
    import time

    time_taken = []
    windows = [1,2,5,10,50,100,500,1000,5000]

    for w in windows:
        begin = time.time()

        t = KnnDtw(n_neighbors=1, max_warping_window=w)
        t.fit(x_train[:20], y_train[:20])
        label, proba = t.predict(x_test[:20])

        end = time.time()
        time_taken.append(end - begin)

    fig = plt.figure(figsize=(12,5))
    _ = plt.plot(windows, [t/400. for t in time_taken], lw=4)
    plt.title('DTW Execution Time with \nvarying Max Warping Window')
    plt.ylabel('Execution Time (seconds)')
    plt.xlabel('Max Warping Window')
    plt.xscale('log')    
