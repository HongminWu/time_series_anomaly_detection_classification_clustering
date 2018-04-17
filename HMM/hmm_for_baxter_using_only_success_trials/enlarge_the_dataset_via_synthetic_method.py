import sys
import os
import pandas as pd
import shutil
import training_config
import generate_synthetic_data
import util

def run():
    folders = os.listdir(training_config.anomaly_data_path)
    for fo in folders:
        path = os.path.join(training_config.anomaly_data_path, fo)
        if not os.path.isdir(path):
            continue
        anomaly_data_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, path)
        for trial_name in anomaly_data_group_by_folder_name:
            df = pd.DataFrame(anomaly_data_group_by_folder_name[trial_name][1], columns=training_config.interested_data_fields)
            output_dir =  os.path.join(training_config.config_by_user['base_path'], 'synthetic_anomalies', fo)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)            
            
            shutil.copy2(os.path.join(path, trial_name), output_dir)

            print 'synthetic data generation'
            generate_synthetic_data.run_finite_differece_matrix(df=df, num_data = 10, csv_save_path=output_dir, trial_name=trial_name)
            
if __name__ == '__main__':
    sys.exit(run())
