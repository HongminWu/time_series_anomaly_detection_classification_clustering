import sys
sys.path.append("/home/birl_wu/TICC")
import TICC_solver as TICC
import numpy as np
import sys
import load_csv_data
import ipdb
base_path = 'anomaly_data_mix'
interested_data_fields = [
         '.wrench_stamped.wrench.force.x',
         '.wrench_stamped.wrench.force.y',
         '.wrench_stamped.wrench.force.z',
         '.wrench_stamped.wrench.torque.x',
         '.wrench_stamped.wrench.torque.y',
         '.wrench_stamped.wrench.torque.z',
        ]
all_trial_data = load_csv_data.run(base_path, interested_data_fields)
all_trial_data = all_trial_data.values()
for i in range(len(all_trial_data)):
    if i == 0:
        _temp = all_trial_data[i]
    else:
        _temp = np.concatenate((_temp, all_trial_data[i]), axis=0)
np.savetxt('anomaly_data.txt', _temp, delimiter=',')
fname = 'anomaly_data.txt'

(cluster_assignment, cluster_MRFs) = TICC.solve(window_size = 5,number_of_clusters = 2, lambda_parameter = 11e-2, beta = 100, maxIters = 100, threshold = 2e-5, write_out_file = False, input_file = fname, prefix_string = "output_folder/", num_proc=1)

print cluster_assignment
np.savetxt('Results.txt', cluster_assignment, fmt='%d', delimiter=',')
