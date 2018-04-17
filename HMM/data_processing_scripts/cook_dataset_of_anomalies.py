import pandas as pd
import ipdb
import os
from shutil import copyfile

ano_keyword_to_label = {
    "left": 0,
    "right": 1,
}

def get_label(f):
    global ano_keyword_to_label
    for key in ano_keyword_to_label:
        if key in f:
            return ano_keyword_to_label[key]
    raise Exception("cannot get label from \"%s\""%f)


if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog -d base_folder_path"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--base-folder",
        action="store", type="string", dest="base_folder",
        help="the folder contains anomaly csv.")
    (options, args) = parser.parse_args()

    if options.base_folder is None:
        parser.error("no base_folder")
    base_folder = options.base_folder

    resampled_anomalies_dir = os.path.join(base_folder, 'resampled_anomalies_dir')

    dataset_of_resampled_anomalies_dir = os.path.join(base_folder, 'dataset_of_resampled_anomalies_dir')
    if not os.path.isdir(dataset_of_resampled_anomalies_dir):
        os.makedirs(dataset_of_resampled_anomalies_dir)

    files = os.listdir(resampled_anomalies_dir)
    files.sort()
    for f in files:
        label = get_label(f)
        file_name = "label_(%s)_from_(%s)"%(label, f)
        copyfile(
            src=os.path.join(resampled_anomalies_dir, f),     
            dst=os.path.join(dataset_of_resampled_anomalies_dir, file_name+'.csv')
        )
    
