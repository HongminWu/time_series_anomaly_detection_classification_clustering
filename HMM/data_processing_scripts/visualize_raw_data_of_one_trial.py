import os
import pandas as pd
from datetime import datetime
import numpy as np
import load_data_folder
import plot_data_in_panda_df


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

    df_group_by_foldername = load_data_folder.run(options.base_folder)
    
    f, df = df_group_by_foldername.iteritems().next()
    
    df = df.loc[df['.tag'] != 0]
    df.index = np.arange(1, len(df)+1)

    state_amount = len(df['.tag'].unique())
    from matplotlib.pyplot import cm 
    color=iter(cm.rainbow(np.linspace(0, 1, state_amount)))

    plot_data_in_panda_df.init_plots()
    for state_no in df['.tag'].unique():
        c=next(color)
        state_df = df.loc[df['.tag'] == state_no]
        plot_data_in_panda_df.plot_one_df(state_df, color=c, label=state_no)
    plot_data_in_panda_df.plot_legend()
    plot_data_in_panda_df.show_plots()

