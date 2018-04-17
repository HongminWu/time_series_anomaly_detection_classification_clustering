import pandas as pd
import ipdb
import os
from shutil import copyfile
import dtw
import matplotlib.pyplot as plt
import birl.robot_introspection_pkg.multi_modal_config as mmc
import copy
import numpy as np
import shutil
from sklearn.preprocessing import normalize, MinMaxScaler

PLOT_VERIFICATION = True 

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
        help="the folder contains lfd csv.")
    (options, args) = parser.parse_args()

    if options.base_folder is None:
        parser.error("no base_folder")
    base_folder = options.base_folder

    resampled_lfd_dir = os.path.join(base_folder, 'resampled_lfd_dir')

    dataset_of_resampled_DTWed_lfd_dir = os.path.join(base_folder, 'dataset_of_resampled_DTWed_lfd_dir')
    if not os.path.isdir(dataset_of_resampled_DTWed_lfd_dir):
        os.makedirs(dataset_of_resampled_DTWed_lfd_dir)
    else:
        shutil.rmtree(dataset_of_resampled_DTWed_lfd_dir)
        os.makedirs(dataset_of_resampled_DTWed_lfd_dir)
         

    df_group_by_label = {}

    files = os.listdir(resampled_lfd_dir)
    files.sort()
    for f in files:
        label = get_label(f)
        if label not in df_group_by_label:
            df_group_by_label[label] = []

        lfd_df = pd.read_csv(os.path.join(resampled_lfd_dir, f), sep=',')
        df_group_by_label[label].append([f, lfd_df])

    for label in df_group_by_label:
        list_of_df = df_group_by_label[label]


        list_of_preprocessed_df = []
        scaler = MinMaxScaler()
        for i in range(0, len(list_of_df)):
            f, df = list_of_df[i]
            preprosessed_df = df.copy()
            mat = preprosessed_df.values[:, 1:]
            mat = scaler.fit_transform(mat)
            preprosessed_df[preprosessed_df.columns[1:]] = mat
            list_of_preprocessed_df.append([f, preprosessed_df]) 

        ref_idx = np.argmin([len(i[1]) for i in list_of_preprocessed_df])
        ref_f, ref_df = list_of_preprocessed_df[ref_idx]
        ref_mat = ref_df.values[:, 1:]

        list_of_dtwed_df = []
        for i in range(0, len(list_of_preprocessed_df)):
            f, raw_df = list_of_df[i]
            f, preprocessed_df = list_of_preprocessed_df[i]
            preprocessed_mat = preprocessed_df.values[:, 1:]

            dist, cost, acc, path = dtw.dtw(ref_mat, preprocessed_mat, dist=lambda x, y: np.linalg.norm(x - y, ord=2))
            list_of_ref_t = path[0]
            list_of_target_t = path[1]

            dtwed_df = ref_df.copy()
            tx = 0
            for idx, t in enumerate(list_of_ref_t):
                if t == tx:
                    ty = list_of_target_t[idx]
                    dtwed_df.values[tx] = raw_df.values[ty]
                    tx += 1
                else:
                    continue
            list_of_dtwed_df.append([f, dtwed_df])

        for i in range(len(list_of_dtwed_df)):
            f, df = list_of_dtwed_df[i]
            file_name = "label_(%s)_from_(%s)"%(label, f)
            df.to_csv(os.path.join(dataset_of_resampled_DTWed_lfd_dir, file_name+".csv"))


        if not PLOT_VERIFICATION:
            continue
        visualization_by_dimension_dir = os.path.join(base_folder, 'visualization_by_dimension_dir')
        DTWed_resampled_lfd_dir = os.path.join(visualization_by_dimension_dir, "DTWed_resampled_lfd_dir", "label_%s"%(label, )) 
        if not os.path.isdir(DTWed_resampled_lfd_dir):
            os.makedirs(DTWed_resampled_lfd_dir)
        else:
            shutil.rmtree(DTWed_resampled_lfd_dir)
            os.makedirs(DTWed_resampled_lfd_dir)

        dimensions = copy.deepcopy(mmc.interested_data_fields)
        if '.tag' in dimensions:
            idx_to_del = dimensions.index('.tag')
            del dimensions[idx_to_del]
        for dim in dimensions:
            lfd_amount = len(list_of_df)
            dtwed_lfd_fig, dtwed_lfd_axs = plt.subplots(nrows=lfd_amount, ncols=3, sharex=True, sharey=True)
            if lfd_amount == 1:
                dtwed_lfd_axs = dtwed_lfd_axs.reshape((1, 3))
            for i in range(0, len(list_of_df)):
                f, raw_df = list_of_df[i]
                f, preprocessed_df = list_of_preprocessed_df[i]
                f, dtwed_df = list_of_dtwed_df[i]

                ax = dtwed_lfd_axs[i, 0] 
                time_x = raw_df.index-raw_df.index[0]
                ax.plot(
                    time_x.tolist(),
                    raw_df[dim].tolist(), 
                )
                title = 'raw_df_from_%s'%(f,)
                ax.set_title(title)

                ax = dtwed_lfd_axs[i, 1] 
                time_x = preprocessed_df.index-preprocessed_df.index[0]
                ax.plot(
                    time_x.tolist(),
                    preprocessed_df[dim].tolist(), 
                )
                title = 'preprocessed_df'
                ax.set_title(title)

                ax = dtwed_lfd_axs[i, 2] 
                time_x = dtwed_df.index-dtwed_df.index[0]
                ax.plot(
                    time_x.tolist(),
                    dtwed_df[dim].tolist(), 
                )
                title = 'dtwed_df_from_%s'%(f,)
                ax.set_title(title)

            dtwed_lfd_fig.set_size_inches(16,4*lfd_amount)
            dtwed_lfd_fig.suptitle('label_'+str(label)+'_'+dim)
            dtwed_lfd_fig.savefig(os.path.join(DTWed_resampled_lfd_dir, 'label_'+str(label)+'_'+dim+'.png'))
            plt.close(dtwed_lfd_fig)

