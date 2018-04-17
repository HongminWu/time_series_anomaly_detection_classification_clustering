import os
import util
from model_config_store import model_store

score_metric_options = [
    '_score_metric_last_time_stdmeanratio_',
    '_score_metric_worst_stdmeanratio_in_10_slice_',
    '_score_metric_sum_stdmeanratio_using_fast_log_cal_',
    '_score_metric_mean_of_std_using_fast_log_cal_',
    '_score_metric_hamming_distance_using_fast_log_cal_',
    '_score_metric_std_of_std_using_fast_log_cal_',
    '_score_metric_mean_of_std_divied_by_final_log_mean_',
    '_score_metric_mean_of_std_of_gradient_divied_by_final_log_mean_',
    '_score_metric_minus_diff_btw_1st_2ed_emissionprob_',
    '_score_metric_minus_diff_btw_1st_2ed(>=0)_divide_maxeprob_emissionprob_',
    '_score_metric_minus_diff_btw_1st_2ed(delete<0)_divide_maxeprob_emissionprob_',
    '_score_metric_mean_of_(std_of_(max_emissionprob_of_trials))_',
    '_score_metric_duration_of_(diff_btw_1st_2ed_emissionprob_<_10)_',
]

anomaly_detection_metric_options = [
    'loglik<threshold=(mean-c*std)',
    'gradient<threshold=(min-range/2)',
    'deri_of_diff',
]

dataset_path_options = [
    '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/baxter_pnp_anomalies/'
    ]


modalities_store = {
    "endpoint_state_pose": [
        '.endpoint_state.pose.position.x',
        '.endpoint_state.pose.position.y',
        '.endpoint_state.pose.position.z',
        '.endpoint_state.pose.orientation.x',
        '.endpoint_state.pose.orientation.y',
        '.endpoint_state.pose.orientation.z',
        '.endpoint_state.pose.orientation.w',
    ],

    'endpoint_state_twist':[
        '.endpoint_state.twist.linear.x',
        '.endpoint_state.twist.linear.y',
        '.endpoint_state.twist.linear.z',
        '.endpoint_state.twist.angular.x',
        '.endpoint_state.twist.angular.y',
        '.endpoint_state.twist.angular.z',
    ],

    'wrench': [
         '.wrench_stamped.wrench.force.x',
         '.wrench_stamped.wrench.force.y',
         '.wrench_stamped.wrench.force.z',
         '.wrench_stamped.wrench.torque.x',
         '.wrench_stamped.wrench.torque.y',
         '.wrench_stamped.wrench.torque.z',
    ],

    'delta_wrench': [
         '.delta_wrench.force.x',
         '.delta_wrench.force.y',
         '.delta_wrench.force.z',
         '.delta_wrench.torque.x',
         '.delta_wrench.torque.y',
         '.delta_wrench.torque.z',
    ],
    
    'joint_position': [
    '.joint_state.position.right_s0',
    '.joint_state.position.right_s1',
    '.joint_state.position.right_e0',
    '.joint_state.position.right_e1',
    '.joint_state.position.right_w0',
    '.joint_state.position.right_w1',
    '.joint_state.position.right_w2',
    ],

    'joint_velocity': [
    '.joint_state.velocity.right_s0',
    '.joint_state.velocity.right_s1',
    '.joint_state.velocity.right_e0',
    '.joint_state.velocity.right_e1',
    '.joint_state.velocity.right_w0',
    '.joint_state.velocity.right_w1',
    '.joint_state.velocity.right_w2',
    ],

    'joint_effort': [
    '.joint_state.effort.right_s0',
    '.joint_state.effort.right_s1',
    '.joint_state.effort.right_e0',
    '.joint_state.effort.right_e1',
    '.joint_state.effort.right_w0',
    '.joint_state.effort.right_w1',
    '.joint_state.effort.right_w2',
    ],

#the following items for IIWA-ROBOT
    'CartesianWrench':[
    '.CartesianWrench.wrench.force.x',
    '.CartesianWrench.wrench.force.y',
    '.CartesianWrench.wrench.force.z',
    '.CartesianWrench.wrench.torque.x',
    '.CartesianWrench.wrench.torque.y',
    '.CartesianWrench.wrench.torque.z',
    ],

    'JointPosition':[
    '.JointPosition.position.a1',
    '.JointPosition.position.a2',
    '.JointPosition.position.a3',
    '.JointPosition.position.a4',
    '.JointPosition.position.a5',
    '.JointPosition.position.a6',
    '.JointPosition.position.a7',
    ],

	'tactile_sensor_data':[
		'.tactile_values.tactile_values_0',
		'.tactile_values.tactile_values_1',
		'.tactile_values.tactile_values_2',
		'.tactile_values.tactile_values_3',
		'.tactile_values.tactile_values_4',
		'.tactile_values.tactile_values_5',
		'.tactile_values.tactile_values_6',
		'.tactile_values.tactile_values_7',
		'.tactile_values.tactile_values_8',
		'.tactile_values.tactile_values_9',
		],

    # for the magnitude i.e norm
    'wrench_magnitude':[
        '.wrench_stamped.wrench.force.magnitude',
        '.wrench_stamped.wrench.torque.magnitude',
        ],

    'endpoint_state_twist_magnitude':[
        '.endpoint_state.twist.linear.magnitude',
        '.endpoint_state.twist.angular.magnitude',
        ],

    'endpoint_state_wrench_magnitude':[
        '.endpoint_state.wrench.force.magnitude',
        '.endpoint_state.wrench.torque.magnitude',
        ],
    
        
}

model_type_options = [
    'BNPY\'s HMM',
    'hmmlearn\'s HMM',
    'hmmlearn\'s GMMHMM',
    'PYHSMM\'s HMM',
]

modality_chosen = [ 'endpoint_state_twist', 'wrench']
interested_data_fields = []
for modality in modality_chosen:
    interested_data_fields += modalities_store[modality]
interested_data_fields.append('.tag')


# config provided by the user
config_by_user = {

    # config for dataset folder
    'base_path'               : '../',
    'dataset_path'            : dataset_path_options[-1],

    # config for types
    'nFeatures'               : len(interested_data_fields) - 1,
    'interested_data_fields'  :  interested_data_fields,
    'data_type_chosen'        :  ', '.join(modality_chosen),    
    'model_type_chosen'       :  model_type_options[1],
    'score_metric'            : '_score_metric_last_time_stdmeanratio_',
    'anomaly_detection_metric': anomaly_detection_metric_options[1],

    # config for preprocessing
    'preprocessing_scaling'   : False,   # scaled data has zero mean and unit variance
    'preprocessing_normalize' : False, # normalize the individual samples to have unit norm "l1" or 'l2'
    'norm_style'              : 'l2',
    'pca_components'          : 0, # cancel the pca processing

    # threshold of derivative used in hmm online anomaly detection
    'deri_threshold'          : 200,
    
    # threshold training c value in threshold=mean-c*std
    'threshold_c_value'       : 5
}

model_config_set_name = model_store[config_by_user['model_type_chosen']]['use']
model_config          = model_store[config_by_user['model_type_chosen']]['config_set'][model_config_set_name]
model_id              = util.get_model_config_id(model_config)
model_id              = config_by_user['score_metric']+model_id
norm_style            = config_by_user['norm_style']

success_path = os.path.join(config_by_user['dataset_path'], "success")
test_success_data_path = os.path.join(config_by_user['dataset_path'], "success_for_test")
model_save_path = os.path.join(config_by_user['base_path'], "model", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)
figure_save_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)

# for anomaly analysis
anomaly_data_path = config_by_user['dataset_path']
anomaly_raw_data_path = os.path.join(config_by_user['dataset_path'], 'anomalies')
anomaly_model_save_path = os.path.join(config_by_user['base_path'], "anomaly_models")
anomaly_identification_figure_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'])

exec '\n'.join("%s=%r"%i for i in config_by_user.items())
