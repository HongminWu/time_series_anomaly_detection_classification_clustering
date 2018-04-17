import os
import ipdb

def get_model_score(files_in_model_id_folder):
    import re

    prog = re.compile('model_s(\d+)_std_mean_ratio_(\d+\.\d+).pkl')
    
    score_group_by_state = {}
    for f in files_in_model_id_folder:
        m = prog.match(f)
        if m:
            state_no = int(m.group(1))
            score = float(m.group(2))
            score_group_by_state[state_no] = score

    return score_group_by_state

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog -d data_folder_path --model-id-prefix model_id_prefix"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--data-folder",
        action="store", type="string", dest="data_folder",
        help="provide a data folder which will have this structure: ./model, ./success, ./figure")

    parser.add_option("--model-id-prefix",
        action="store", type="string", dest="model_id_prefix",
        default='',
        help="model_id_prefix to filter models")
    (options, args) = parser.parse_args()

    if options.data_folder is None:
        parser.error("no data_folder")

    model_folder = os.path.join(options.data_folder, 'model')

    if not os.path.isdir(model_folder):
        raise Exception('model folder \"%s\" not found.'%(model_folder,))

    y_multi = []
    labels = []

    for data_type in os.listdir(model_folder):
        print ' -> data_type', data_type
        dt_path = os.path.join(model_folder, data_type)
        if not os.path.isdir(dt_path):
            continue

        for model_type in os.listdir(dt_path):
            print ' \t -> model_type', model_type
            mt_path = os.path.join(dt_path, model_type)
            if not os.path.isdir(mt_path):
                continue

            for model_id in os.listdir(mt_path):
                print ' \t\t -> model_id', model_id
                if not model_id.startswith(options.model_id_prefix):
                    print 'prefix mismatch, skipped'
                    continue
                mi_path = os.path.join(mt_path, model_id)
                if not os.path.isdir(mi_path):
                    continue
    
                
                score_group_by_state = get_model_score(os.listdir(mi_path))
                score_list = []
                for state_no in sorted(score_group_by_state):
                    score_list.append(score_group_by_state[state_no])

                if len(score_list) == 0:
                    continue
                y_multi.append(score_list)
                labels.append("%s->%s->%s"%(data_type, model_type, model_id))

    model_amount = len(y_multi)
    state_amount = len(y_multi[0]) 

    width = 0.7/model_amount
    x = range(1, state_amount+1)

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm 
    import numpy as np

    color=iter(cm.rainbow(np.linspace(0,1,model_amount)))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(y_multi)):
        try:
            c=next(color)
            ax.bar([j+i*width for j in x], y_multi[i], width=width, label=labels[i], color=c)
        except AssertionError:
            print labels[i], y_multi[i], ' is problematic.'

    ax.legend(loc=3)
    plt.show()







