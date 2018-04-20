from HMM.hmm_for_baxter_using_only_success_trials import (training_config, util)
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_generator(base_folder):
    folders = os.listdir(base_folder)
    for fo in folders:
        print "plotting the " + fo
        X, y= [],[]
        path = os.path.join(base_folder, fo)
        if not os.path.isdir(path):
            continue
        anomaly_data_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, path)
        temp = anomaly_data_group_by_folder_name.values()
        file_names = anomaly_data_group_by_folder_name.keys()
        for i in range(len(temp)):
            X.append(temp[i][1])
            y.append(fo)
        yield X, y, fo, file_names 

def plot_anomaly_by_label(base_folder):
    X_train, X_test, y_train, y_test, class_names = [], [], [], [], []
    for X, y, fo, file_names in data_generator(base_folder):
        X=np.concatenate(X)
        per_anomaly_len  = len(X)/len(y)
        Xdf = pd.DataFrame(X, columns=training_config.interested_data_fields)
        ax = Xdf.plot(legend=True, title=fo)
        ax.set_xlabel('Time')
        for xc in range(per_anomaly_len, len(X), per_anomaly_len):
            ax.axvline(x=xc, color='k', linestyle='--')
            ax.legend(loc='best', framealpha=0.2)
        plt.savefig('images/'+ fo + '.png')
    plt.show()
    print 'generated plots been placed in folder /images'

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog -d base_folder_path"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--base-folder",
        action="store", type="string", dest="base_folder",
        help="provide a base folder which will have this structure: \
            ./01, ./01/*.csv, ./02, ./02/*.csv, ...")
 
    (options, args) = parser.parse_args()

    if options.base_folder is None:
        parser.error("no base_folder, please add -d dir or -h for detailed help")
    else:
        base_folder = options.base_folder
    
    sys.exit(plot_anomaly_by_label(base_folder))
