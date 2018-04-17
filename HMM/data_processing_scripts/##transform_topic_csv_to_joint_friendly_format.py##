import os
import pandas as pd
import load_data_folder
import numpy as np
import ipdb

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
        parser.error("no base_folder")
    else:
        base_folder = options.base_folder

    files = os.listdir(base_folder)
    df_group_by_foldername = load_data_folder.run(base_folder)

    for f in df_group_by_foldername:
        print "doing "+f
        path = os.path.join(base_folder, f)
        df = df_group_by_foldername[f]

        from dateutil import parser
        df['time'] = df['time'].apply(lambda x: parser.parse(x))
        trial_start_datetime = df['time'][0]
        df['time'] = df['time'].apply(lambda x: x-trial_start_datetime)
        df['time'] = df['time'].apply(lambda x: x/np.timedelta64(1, 's'))
        df = df.set_index('time')

        column_names = df.columns
        if ".joint_state.name" in column_names\
            and ".joint_state.velocity" in column_names\
            and ".joint_state.effort" in column_names\
            and ".joint_state.position" in column_names:
            pass
        else:
            raise Exception("%s doesn't contain \
                \".joint_state.name\", \".joint_state.position\", \
                \".joint_state.velocity\" or \".joint_state.effort\"")

        list_of_joint_name_json_str = df[".joint_state.name"].unique()
        list_of_joint_name = [] 
        for json_str in list_of_joint_name_json_str:
            list_of_joint_name += eval(json_str)
        for joint_name in list_of_joint_name:
            df['.joint_state.position.'+joint_name] = np.nan
            df['.joint_state.velocity.'+joint_name] = np.nan
            df['.joint_state.effort.'+joint_name] = np.nan
    
        column_names = df.columns
        def apply_func(x):
            x = eval(x)
            return [
                [column_names.get_loc('.joint_state.position.'+i) for i in x],
                [column_names.get_loc('.joint_state.velocity.'+i) for i in x],
                [column_names.get_loc('.joint_state.effort.'+i) for i in x],
            ]
        df["joint_idx"] = df[".joint_state.name"].copy()
        df["joint_idx"] = df["joint_idx"].apply(apply_func)
        df[".joint_state.position"] = df[".joint_state.position"].apply(lambda x: eval(x))
        df[".joint_state.velocity"] = df[".joint_state.velocity"].apply(lambda x: eval(x))
        df[".joint_state.effort"] = df[".joint_state.effort"].apply(lambda x: eval(x))

        column_names = df.columns
        joind_idx_idx = column_names.get_loc('joint_idx')
        pos_idx = column_names.get_loc('.joint_state.position')
        vel_idx = column_names.get_loc('.joint_state.velocity')
        eff_idx = column_names.get_loc('.joint_state.effort')
        def apply_func(row):
            joint_idx = row[joind_idx_idx]
            list_of_joint_position = row[pos_idx]
            list_of_joint_velocity = row[vel_idx]
            list_of_joint_effort = row[eff_idx]
            for i in range(len(joint_idx[0])):
                row[joint_idx[0][i]] = list_of_joint_position[i]
                row[joint_idx[1][i]] = list_of_joint_velocity[i]
                row[joint_idx[2][i]] = list_of_joint_effort[i]
            return row

        df = df.apply(apply_func, axis=1)
        df = df.interpolate(method='linear', limit_direction='both')
        df.to_csv(os.path.join(path, "joint_friendly_tag_multimodal_tranformed_from_"+f+".csv")) 


