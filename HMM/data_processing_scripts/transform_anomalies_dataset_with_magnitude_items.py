import os
import pandas as pd
import numpy as np
import ipdb

def load_data_folder(folder_path):
    print 'loading the folders'
    if not os.path.isdir(folder_path):
        raise 'Please check the base path'
    files = os.listdir(folder_path)

    pandadf_group_by_filename = {}
    for f in files:
        path = os.path.join(folder_path, f)
        df = pd.read_csv(path, sep=',')
        pandadf_group_by_filename[f] = df
    return pandadf_group_by_filename
    

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

    folders = os.listdir(base_folder)
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        df_group_by_filename = load_data_folder(folder_path)
        for f in df_group_by_filename:
            print "Processing " + f
            df = df_group_by_filename[f]
    
            df['.wrench_stamped.wrench.force.magnitude'] = np.sqrt(df['.wrench_stamped.wrench.force.x']**2 +\
                                                           df['.wrench_stamped.wrench.force.y']**2 +\
                                                           df['.wrench_stamped.wrench.force.z']**2)

            df['.wrench_stamped.wrench.torque.magnitude'] = np.sqrt(df['.wrench_stamped.wrench.torque.x']**2 +\
                                                            df['.wrench_stamped.wrench.torque.y']**2 +\
                                                            df['.wrench_stamped.wrench.torque.z']**2)

    
            df['.endpoint_state.wrench.force.magnitude'] = np.sqrt(df['.endpoint_state.wrench.force.x']**2 +\
                                                           df['.endpoint_state.wrench.force.y']**2 +\
                                                           df['.endpoint_state.wrench.force.z']**2)

            df['.endpoint_state.wrench.torque.magnitude'] = np.sqrt(df['.endpoint_state.wrench.torque.x']**2 +\
                                                            df['.endpoint_state.wrench.torque.y']**2 +\
                                                            df['.endpoint_state.wrench.torque.z']**2)

                                                           
            df['.endpoint_state.twist.linear.magnitude'] = np.sqrt(df['.endpoint_state.twist.linear.x']**2 +\
                                                           df['.endpoint_state.twist.linear.y']**2 +\
                                                           df['.endpoint_state.twist.linear.z']**2)


            df['.endpoint_state.twist.angular.magnitude'] = np.sqrt(df['.endpoint_state.twist.angular.x']**2 +\
                                                            df['.endpoint_state.twist.angular.y']**2 +\
                                                            df['.endpoint_state.twist.angular.z']**2)
                                                           
            df.to_csv(os.path.join(folder_path, f))
            print 'finish processing ' + f

