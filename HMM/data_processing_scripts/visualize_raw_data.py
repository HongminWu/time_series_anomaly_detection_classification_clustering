import os
import pandas as pd
from datetime import datetime
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


    files = os.listdir(options.base_folder)

    from matplotlib.pyplot import cm 
    import numpy as np
    color=iter(cm.rainbow(np.linspace(0, 1, len(files))))
    df_group_by_foldername = load_data_folder.run(options.base_folder)

    plot_data_in_panda_df.init_plots()
    for f, df in df_group_by_foldername.iteritems():
        df = df.loc[df['.tag'] != 0]
        df.index = np.arange(1, len(df)+1)
        c=next(color)
        plot_data_in_panda_df.plot_one_df(df, color=c, label=f)
    
    plot_data_in_panda_df.plot_legend()
    plot_data_in_panda_df.show_plots()

