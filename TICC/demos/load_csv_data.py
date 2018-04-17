import pandas as pd
import os

def _load_csv_data(csv_file_path,  interested_data_fields):
    df = pd.read_csv(csv_file_path, sep=',')
    df = df[interested_data_fields]
    one_trial_data = df.values
    return one_trial_data

def run(base_path, interested_data_fields):
    folders = os.listdir(base_path)
    all_trial_data = {}
    for fo in folders:
        fo_path = os.path.join(base_path, fo)
        if not os.path.isdir(fo_path):
            continue
        files = os.listdir(fo_path)
        for f in files:
            if os.path.isfile(os.path.join(fo_path, f)):
                csv_file_path = os.path.join(fo_path, f)
                one_trial_data = _load_csv_data(csv_file_path, interested_data_fields)
            file_name = fo + ':' + f
            all_trial_data[file_name] = one_trial_data
    return all_trial_data
                
