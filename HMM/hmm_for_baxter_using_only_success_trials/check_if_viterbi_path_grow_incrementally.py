#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from math import (
    log,
    exp
)
from matplotlib import pyplot as plt
import time
import util

def output_growing_viterbi_path_img(
    list_of_growing_viterbi_paths, 
    hidden_state_amount, 
    output_file_path,
):
    from matplotlib.pyplot import cm
    import numpy as np

    height = len(list_of_growing_viterbi_paths)
    width = len(list_of_growing_viterbi_paths[-1])

    colors = [tuple((256*i).astype(int)) for i in cm.rainbow(np.linspace(0, 1, hidden_state_amount))]

    output_pixels = []

    for vp in list_of_growing_viterbi_paths:
        black_to_append = width-len(vp)
        row = [colors[i] for i in vp]+[(0,0,0) for i in range(black_to_append)]
        output_pixels += row

    from PIL import Image

    output_dir = os.path.dirname(output_file_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_img = Image.new("RGB", (width, height)) # mode,(width,height)
    output_img.putdata(output_pixels)
    output_img.save(
        output_file_path,
    )

    
def run(model_save_path, 
    figure_save_path,
    trials_group_by_folder_name,
    options,
):

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    threshold_constant = 10
    threshold_offset = 10

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        try:
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        except IOError:
            print 'model of state %s not found'%(state_no,)
            continue

    for state_no in model_group_by_state:

        curve_owner = []
        for trial_name in trials_group_by_folder_name:
            curve_owner.append(trial_name)
            
            list_of_growing_viterbi_paths, n_samples, n_components = util.fast_growing_viterbi_paths_cal(
                trials_group_by_folder_name[trial_name][state_no],
                model_group_by_state[state_no]
            )

            output_growing_viterbi_path_img(
                list_of_growing_viterbi_paths, 
                n_components,
                os.path.join(
                    figure_save_path, 
                    'check_if_viterbi_path_grow_incrementally',
                    "state_%s"%state_no, 
                    "%s.png"%trial_name,
                ), 
            )

