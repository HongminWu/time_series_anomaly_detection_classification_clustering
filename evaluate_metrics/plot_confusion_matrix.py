import numpy as np
import itertools
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import sys
import ipdb

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    try:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    except FloatingPointError:
        print ('Error occurred: invalid value encountered in divide')
        sys.exit()
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def run(y_test, y_pred, class_names):
    #evaluate the results
    print 'classification_report:\n'
    print classification_report(y_test, y_pred, target_names=[l for l in class_names])
    print 'average-weighted:\n'
    print precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print 'average-micro:\n'    
    print precision_recall_fscore_support(y_test, y_pred, average='micro')
    print 'average-macro:\n'    
    print precision_recall_fscore_support(y_test, y_pred, average='macro')    
    # Plot non-normalized confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    # import datetime
    # ctim = datetime.datetime.now().strftime("%Yy%mm%dd%HH%MM%SS")
    # plt.savefig('images/'+ ctim + '.png', dpi=120)    
