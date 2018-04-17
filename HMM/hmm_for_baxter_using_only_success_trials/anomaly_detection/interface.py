from sklearn.externals import joblib
import util
import os
import ipdb
import Detectors

def get_anomaly_detector(
    model_save_path, 
    state_amount,
    anomaly_detection_metric,
):

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))

    if anomaly_detection_metric == 'loglik<threshold=(mean-c*std)':
        threshold_pkl_data = joblib.load(os.path.join(model_save_path, 'threshold_for_log_likelihood.pkl'))
        return Detectors.DetectorBasedOnLoglikCurve(model_group_by_state, threshold_pkl_data)
    elif anomaly_detection_metric == 'gradient<threshold=(min-range/2)':
        threshold_pkl_data = joblib.load(os.path.join(model_save_path, 'threshold_for_gradient_of_log_likelihood.pkl'))
        return Detectors.DetectorBasedOnGradientOfLoglikCurve(model_group_by_state, threshold_pkl_data)
    elif anomaly_detection_metric == 'deri_of_diff':
        threshold_pkl_data = joblib.load(os.path.join(model_save_path, 'threshold_for_deri_of_diff.pkl'))
        mean_curve_group_by_state = joblib.load(os.path.join(model_save_path, 'mean_curve_group_by_state.pkl'))
        return Detectors.DetectorBasedOnDeriOfDiff(model_group_by_state, threshold_pkl_data, mean_curve_group_by_state)
