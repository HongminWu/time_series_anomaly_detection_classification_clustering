import os
import pandas as pd
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
import load_data_folder
import plot_data_in_panda_df
import matplotlib.pyplot as plt
import ipdb
import matplotlib.dates as mdates 
import numpy as np
from matplotlib.pyplot import cm 

def trim_non_trial_data(tag_multimodal_df, hmm_online_result_df):
    state_df = tag_multimodal_df[tag_multimodal_df['.tag'] != 0]
    row = state_df.head(1)
    trial_start_time = row['time'].values[0]
    row = state_df.tail(1)
    trial_end_time = row['time'].values[0]

    return tag_multimodal_df[(tag_multimodal_df['time']>=trial_start_time) & (tag_multimodal_df['time']<=trial_end_time)], \
        hmm_online_result_df[(hmm_online_result_df['time']>=trial_start_time) & (hmm_online_result_df['time']<=trial_end_time)]

def color_anomaly_pos(tag_multimodal_df, list_of_anomaly_time_range):
    fig = plt.figure()
    pos_plot = fig.add_subplot(111, projection='3d')
    pos_plot.plot(
        tag_multimodal_df['.endpoint_state.pose.position.x'].tolist(), 
        tag_multimodal_df['.endpoint_state.pose.position.y'].tolist(), 
        tag_multimodal_df['.endpoint_state.pose.position.z'].tolist(), 
        color='gray',
    )

    anomaly_amount = len(list_of_anomaly_time_range)

    color=iter(cm.rainbow(np.linspace(0, 1, anomaly_amount)))
    for i in range(len(list_of_anomaly_time_range)):
        start_time = list_of_anomaly_time_range[i][0]
        end_time = list_of_anomaly_time_range[i][1]
        anomaly_df = tag_multimodal_df[(tag_multimodal_df['time']>=start_time) & (tag_multimodal_df['time']<=end_time)]
        pos_plot.plot(
            anomaly_df['.endpoint_state.pose.position.x'].tolist(), 
            anomaly_df['.endpoint_state.pose.position.y'].tolist(), 
            anomaly_df['.endpoint_state.pose.position.z'].tolist(), 
            color=next(color),
        )

def get_anomaly_range(tag_df, flag_df):
    list_of_idx_range = []
    list_of_anomaly_time_range = []
    
    flag_start_idx = 0
    tag_start_idx = 0

    flag_df_length = flag_df.shape[0]
    tag_df_length = tag_df.shape[0]

    anomaly_start_time = None
    anomaly_backtracked_start_time = None
    anomaly_end_time = flag_df['time'][0]
    anomaly_detected = False
    while True:
        if not anomaly_detected:
            print 'start new ano search...',
            row = flag_df[flag_df['time']>anomaly_end_time].head(1)
            if len(row) == 0:
                raise Exception("flag_df failed to sync by anomaly_end_time")
            flag_start_idx = row.index[0] 
            for idx in range(flag_start_idx, flag_df_length):
                if flag_df['.event_flag'][idx] == 0:
                    backtracked_idx = idx 
                    while flag_df['.event_flag'][backtracked_idx] <= flag_df['.event_flag'][idx]:
                        backtracked_idx -= 1

                    anomaly_detected = True
                    anomaly_start_time = flag_df['time'][idx]
                    anomaly_backtracked_start_time = flag_df['time'][backtracked_idx]
                    print 'got one...',
                    break
            if not anomaly_detected:
                break
        else:
            row = tag_df[tag_df['time']>anomaly_start_time].head(1)
            if len(row) == 0:
                raise Exception("tag_df failed to sync by anomaly_start_time")
            tag_start_idx = row.index[0] 

            search_secs = 3
            search_end_time = row['time'].values[0]+np.timedelta64(search_secs, 's') 

            row = tag_df[tag_df['time']>=search_end_time].head(1)
            if len(row) == 0:
                raise Exception("tag_df failed to sync by anomaly_start_time+%ss"%(search_secs,))
            tag_end_idx = row.index[0]


            for idx in range(tag_start_idx, tag_end_idx):
                now_state = tag_df['.tag'][idx]
                if now_state == 0:
                    anomaly_detected = False 
                    anomaly_end_time = tag_df['time'][idx]
                    list_of_anomaly_time_range.append([
                        anomaly_backtracked_start_time,
                        anomaly_end_time,
                    ])
                    print 'ano added'
                    break
            if anomaly_detected:
                anomaly_detected = False 
                anomaly_end_time = anomaly_start_time
                print 'smach did go into recov in %ss, we will skip this ano'%(search_secs,)

    print 'done search'
    return list_of_anomaly_time_range 

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog -d base_folder_path"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--base-folder",
        action="store", type="string", dest="base_folder",
        help="provide a base folder which will have this structure: ./01, ./01/*.csv, ./02, ./02/*.csv, ...")
    (options, args) = parser.parse_args()

    if options.base_folder is None:
        parser.error("no base_folder")

    base_folder = options.base_folder

    files = os.listdir(base_folder)
    for f in files:
        path = os.path.join(base_folder, f)
        if not os.path.isdir(path):
            continue

        if os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            tag_multimodal_csv_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            tag_multimodal_csv_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have tag_multimodal csv file."%(path,))

        if os.path.isfile(os.path.join(path, f+'-hmm_online_result.csv')):
            hmm_online_result_csv_path = os.path.join(path, f+'-hmm_online_result.csv')
        elif os.path.isfile(os.path.join(path, 'hmm_online_result.csv')):
            hmm_online_result_csv_path = os.path.join(path, 'hmm_online_result.csv')
        else:
            raise Exception("folder %s doesn't have hmm_online_result csv file."%(path,))

        tag_multimodal_df = pd.read_csv(tag_multimodal_csv_path, sep=',')
        hmm_online_result_df = pd.read_csv(hmm_online_result_csv_path, sep=',')


        from dateutil import parser
        tag_multimodal_df['time'] = tag_multimodal_df['time'].apply(lambda x: parser.parse(x))
        hmm_online_result_df['time'] = hmm_online_result_df['time'].apply(lambda x: parser.parse(x))


        tag_multimodal_df, hmm_online_result_df = trim_non_trial_data(tag_multimodal_df, hmm_online_result_df)


        print
        print '-'*20
        print f
        list_of_anomaly_time_range = get_anomaly_range(
            tag_multimodal_df,
            hmm_online_result_df,
        )
        print list_of_anomaly_time_range
        print '-'*20

        color_anomaly_pos(tag_multimodal_df, list_of_anomaly_time_range)

        fig = plt.figure()
        deri_of_diff = fig.add_subplot(111)
        deri_of_diff.plot(
            hmm_online_result_df['time'].tolist(),
            hmm_online_result_df['.deri_of_diff_btw_curlog_n_thresh.data'].tolist(),
            marker='o',
        )
    
        for i in hmm_online_result_df[hmm_online_result_df['.event_flag'] == 0].index:
            deri_of_diff.plot(
                [hmm_online_result_df['time'][i]],
                [hmm_online_result_df['.deri_of_diff_btw_curlog_n_thresh.data'][i]],
                marker='o',
                color = 'red',
                linestyle='None',
            )


        for i in range(len(list_of_anomaly_time_range)):
            start_time = list_of_anomaly_time_range[i][0]
            end_time = list_of_anomaly_time_range[i][1]
            tag_multimodal_df[(tag_multimodal_df['time']>=start_time) & (tag_multimodal_df['time']<=end_time)].to_csv(os.path.join(path, 'extracted_anomaly_%s.csv'%(i,)))            
            xmin = mdates.date2num(start_time)
            xmax = mdates.date2num(end_time)
            deri_of_diff.axvspan(xmin, xmax, facecolor='red', alpha=0.25)

        title = 'anomaly range of trial %s'%(f,)
        deri_of_diff.set_title(title)
        fig.savefig(os.path.join(path, title+".png"), format="png", dpi=900)


        fig = plt.figure()
        bbox_extra_artists = []
        pos_plot = fig.add_subplot(111, projection='3d')
        pos_plot.plot(
            tag_multimodal_df['.endpoint_state.pose.position.x'].tolist(), 
            tag_multimodal_df['.endpoint_state.pose.position.y'].tolist(), 
            tag_multimodal_df['.endpoint_state.pose.position.z'].tolist(), 
            color='gray',
        )

        recovery_df = tag_multimodal_df[tag_multimodal_df['.tag'] == 0]
        pos_plot.plot(
            recovery_df['.endpoint_state.pose.position.x'].tolist(), 
            recovery_df['.endpoint_state.pose.position.y'].tolist(), 
            recovery_df['.endpoint_state.pose.position.z'].tolist(), 
            marker='.',
            markersize=0.25,
            linestyle='None',
            color='black',
        )
        

        anomaly_amount = len(list_of_anomaly_time_range)
        color=iter(cm.rainbow(np.linspace(0, 1, anomaly_amount)))
        for i in range(len(list_of_anomaly_time_range)):
            start_time = list_of_anomaly_time_range[i][0]
            end_time = list_of_anomaly_time_range[i][1]
            anomaly_df = tag_multimodal_df[(tag_multimodal_df['time']>=start_time) & (tag_multimodal_df['time']<=end_time)]
            pos_plot.plot(
                anomaly_df['.endpoint_state.pose.position.x'].tolist(), 
                anomaly_df['.endpoint_state.pose.position.y'].tolist(), 
                anomaly_df['.endpoint_state.pose.position.z'].tolist(), 
                color=next(color),
                label='extracted_anomaly_%s'%i,
                marker='.',
                markersize=0.25,
                linestyle='None',
            )
        title = 'colored anomaly pos of trial %s'%(f,)
        pos_plot.set_title(title)
        lgd = pos_plot.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)
        fig.savefig(os.path.join(path, title+".png"), format="png", dpi=900, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')




