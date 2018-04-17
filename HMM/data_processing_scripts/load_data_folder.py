import os
import pandas as pd
import ipdb

def load_data_of_ben_struct(base_folder):
    print "load_data_of_ben_struct"
    files = os.listdir(base_folder)
    pandadf_group_by_foldername = {}
    for f in files:
        path = os.path.join(base_folder, f)
        if not os.path.isdir(path):
            continue
        print "processing", f
        
        if os.path.isfile(os.path.join(path, 'joint_friendly_tag_multimodal_tranformed_from_' + f + '.csv')):
            csv_file_path = os.path.join(path, 'joint_friendly_tag_multimodal_tranformed_from_'+ f + '.csv')
        elif os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            csv_file_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            csv_file_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have csv file."%(path,))

        df = pd.read_csv(csv_file_path, sep=',')
        pandadf_group_by_foldername[f] = df
        
    return pandadf_group_by_foldername

def load_data_of_rcbht_struct(base_folder):
    files = os.listdir(base_folder)
    pandadf_group_by_foldername = {}
    for f in files:
        path = os.path.join(base_folder, f)
        if not os.path.isdir(path):
            continue

        if os.path.isfile(os.path.join(path, 'R_Torques.dat')):
            torque_file_path = os.path.join(path, 'R_Torques.dat')
        else:
            print("folder %s doesn't have R_Torques.dat file."%(path,))
            continue

        if os.path.isfile(os.path.join(path, 'R_State.dat')):
            state_file_path = os.path.join(path, 'R_State.dat')
        else:
            print("folder %s doesn't have R_State.dat file."%(path,))
            continue


        df = pd.read_csv(
            torque_file_path, 
            delimiter=' *\t *', 
            header=None, 
        )

        df.columns = [
            'time', 
            '.wrench_stamped.wrench.force.x',
            '.wrench_stamped.wrench.force.y',
            '.wrench_stamped.wrench.force.z',
            '.wrench_stamped.wrench.torque.x',
            '.wrench_stamped.wrench.torque.y',
            '.wrench_stamped.wrench.torque.z',
        ]
        
        df['.tag'] = 0

        state_info = pd.read_csv(
            state_file_path, 
            delimiter=' *\t *', 
            header=None, 
        )
        state_start_at = 0
        if state_info[0][0] == 0:
            start_idx = 1
        else:
            start_idx = 0
        state_count = 1
        for idx in range(start_idx, len(state_info[0])):
            state_end_at = state_info[0][idx]
            df.loc[(df['time']>=state_start_at) & (df['time']<=state_end_at), '.tag'] = state_count 
            state_start_at = state_end_at
            state_count += 1
        df.loc[(df['time']>=state_start_at), '.tag'] = state_count 

        pandadf_group_by_foldername[f] = df
        
    return pandadf_group_by_foldername


def run(base_folder):
    if 'hiro' in base_folder.lower():
        return load_data_of_rcbht_struct(base_folder)
    else:
        return load_data_of_ben_struct(base_folder)
