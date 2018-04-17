import log_likelihood_incremental_calculator.interface
import numpy as np
import ipdb

class BaseDetector(object):
    def __init__(self, model_group_by_state):
        self.model_group_by_state = model_group_by_state
        
        loglik_incre_cal_group_by_state = {}
        for state_no in model_group_by_state:
            loglik_incre_cal_group_by_state[state_no] = log_likelihood_incremental_calculator.interface.get_calculator(model_group_by_state[state_no])
            
        self.loglik_incre_cal_group_by_state = loglik_incre_cal_group_by_state

        prev_loglik_group_by_state = {}
        for state_no in model_group_by_state:
            prev_loglik_group_by_state[state_no] = 0 
        self.prev_loglik_group_by_state = prev_loglik_group_by_state

        now_gradient_group_by_state = {}
        for state_no in model_group_by_state:
            now_gradient_group_by_state[state_no] = 0
        self.now_gradient_group_by_state = now_gradient_group_by_state


        self.metric_observation = []
        self.metric_threshold = []
        self.anomaly_point = []

    def identify_skill(self, sample):
        for state_no in self.loglik_incre_cal_group_by_state:
            calculator = self.loglik_incre_cal_group_by_state[state_no]
            now_loglik = calculator.add_one_sample_and_get_loglik(sample)
            self.now_gradient_group_by_state[state_no] = now_loglik-self.prev_loglik_group_by_state[state_no] 
            self.prev_loglik_group_by_state[state_no] = now_loglik

        most_probable_skill, its_gradient = max(self.now_gradient_group_by_state.items(), key=lambda x:x[1])

        return most_probable_skill
        
    def plot_metric_data(self, ax, plot_metric_observation_only=False):
        ax.plot(self.metric_observation, color='blue')
    
        if plot_metric_observation_only:
            return
    
        ax.plot(self.metric_threshold, color='red')

        for point in self.anomaly_point:
            ax.plot(point[0], point[1], marker='o', color='red', linestyle='None')

    def reset(self):
        pass

class DetectorBasedOnLoglikCurve(BaseDetector):
    def __init__(self, model_group_by_state, threshold_curve_group_by_state):
        BaseDetector.__init__(self, model_group_by_state)
        self.threshold_curve_group_by_state = threshold_curve_group_by_state
        self.prev_skill = None
        self.calculator = None
        self.now_skill_t = None

    def reset(self):
        self.prev_skill = None

    def add_one_smaple_and_identify_skill_and_detect_anomaly(self, sample, now_skill=None):
        if now_skill is None:
            now_skill = BaseDetector.identify_skill(self, sample)
        prev_skill = self.prev_skill
        self.prev_skill = now_skill

        now_loglik = None
        now_threshold = None

        if now_skill is None:
            print 'now_skill is None so we can\'t perform anomaly detection.'
            self.metric_observation.append(now_loglik)
            self.metric_threshold.append(now_threshold)
            return now_skill, None, now_loglik, now_threshold
    
        if now_skill != prev_skill:
            print "now_skill != prev_skill, gonna switch model and restart anomaly detection."
            self.calculator = log_likelihood_incremental_calculator.interface.get_calculator(self.model_group_by_state[now_skill])
            self.now_skill_t = 0

        t = self.now_skill_t
        self.now_skill_t += 1

        now_loglik = self.calculator.add_one_sample_and_get_loglik(sample)

        threshold_curve = self.threshold_curve_group_by_state[now_skill]

        if t >= len(threshold_curve):
            print 'input data is longer than threshold curve so we can\'t perform anomaly detection.'
            self.metric_observation.append(now_loglik)
            self.metric_threshold.append(now_threshold)
            return now_skill, None, now_loglik, now_threshold

        now_threshold = threshold_curve[t]
        anomaly_detected = False
        if now_loglik <= now_threshold:
            anomaly_detected = True

        if anomaly_detected:
            self.anomaly_point.append([len(self.metric_observation), now_loglik])

        self.metric_observation.append(now_loglik)
        self.metric_threshold.append(now_threshold)
        return now_skill, anomaly_detected, now_loglik, now_threshold

class DetectorBasedOnGradientOfLoglikCurve(BaseDetector):
    def __init__(self, model_group_by_state, threshold_constant_group_by_state):
        BaseDetector.__init__(self, model_group_by_state)
        self.threshold_constant_group_by_state = threshold_constant_group_by_state
        self.prev_skill = None
        self.prev_loglik = None
        self.calculator = None

    def reset(self):
        self.prev_skill = None

    def add_one_smaple_and_identify_skill_and_detect_anomaly(self, sample, now_skill=None):
        if now_skill is None:
            now_skill = BaseDetector.identify_skill(self, sample)
        prev_skill = self.prev_skill
        self.prev_skill = now_skill

        now_gradient = None
        now_threshold = None

        if now_skill is None:
            print 'now_skill is None so we can\'t perform anomaly detection.'
            self.metric_observation.append(now_gradient)
            self.metric_threshold.append(now_threshold)
            return now_skill, None, now_gradient, now_threshold
    
        if now_skill != prev_skill:
            print "now_skill != prev_skill, gonna switch model and restart anomaly detection."
            self.calculator = log_likelihood_incremental_calculator.interface.get_calculator(self.model_group_by_state[now_skill])
            self.prev_loglik = None

        now_loglik = self.calculator.add_one_sample_and_get_loglik(sample)
        prev_loglik = self.prev_loglik
        self.prev_loglik = now_loglik

        threshold_constant = self.threshold_constant_group_by_state[now_skill]

        if prev_loglik is None:
            print 'we don\' have prev_loglik for now_skill, gonna wait one more run.'
            self.metric_observation.append(now_gradient)
            self.metric_threshold.append(now_threshold)
            return now_skill, None, now_gradient, now_threshold

        now_gradient = now_loglik-prev_loglik
        now_threshold = threshold_constant
        anomaly_detected = False
        if now_gradient <= now_threshold:
            anomaly_detected = True

        if anomaly_detected:
            self.anomaly_point.append([len(self.metric_observation), now_gradient])

        self.metric_observation.append(now_gradient)
        self.metric_threshold.append(now_threshold)
        return now_skill, anomaly_detected, now_gradient, now_threshold
        
class DetectorBasedOnDeriOfDiff(BaseDetector):
    def __init__(
        self, 
        model_group_by_state, 
        threshold_curve_group_by_state,
        mean_curve_group_by_state
):
        BaseDetector.__init__(self, model_group_by_state)
        self.threshold_curve_group_by_state = threshold_curve_group_by_state
        self.mean_curve_group_by_state = mean_curve_group_by_state
        self.prev_skill = None

        self.calculator = None
        self.now_skill_t = None
        self.prev_diff = None

    def reset(self):
        self.prev_skill = None

    def add_one_smaple_and_identify_skill_and_detect_anomaly(self, sample, now_skill=None):
        if now_skill is None:
            now_skill = BaseDetector.identify_skill(self, sample)
        prev_skill = self.prev_skill
        self.prev_skill = now_skill

        now_deri_of_diff = None
        now_threshold = None

        if now_skill is None:
            print 'now_skill is None so we can\'t perform anomaly detection.'
            self.metric_observation.append(now_deri_of_diff)
            self.metric_threshold.append(now_threshold)
            return now_skill, None, now_deri_of_diff, now_threshold
    
        if now_skill != prev_skill:
            print "now_skill != prev_skill, gonna switch model and restart anomaly detection."
            self.calculator = log_likelihood_incremental_calculator.interface.get_calculator(self.model_group_by_state[now_skill])
            self.now_skill_t = 0
            self.prev_diff = None

        t = self.now_skill_t
        self.now_skill_t += 1

        threshold_constant = self.threshold_curve_group_by_state[now_skill]
        mean_curve = self.mean_curve_group_by_state[now_skill]

        if t >= mean_curve.shape[1]:
            print 'input data is longer than threshold curve so we can\'t perform anomaly detection.'
            self.metric_observation.append(now_deri_of_diff)
            self.metric_threshold.append(now_threshold)
            return now_skill, None, now_deri_of_diff, now_threshold

        now_threshold = threshold_constant
        now_mean = mean_curve[0, t]

        now_loglik = self.calculator.add_one_sample_and_get_loglik(sample)
        now_diff = now_loglik-now_mean
        prev_diff = self.prev_diff
        self.prev_diff = now_diff

        if prev_diff is None:
            print 'we don\' have prev_diff for now_skill, gonna wait one more run.'
            self.metric_observation.append(now_deri_of_diff)
            self.metric_threshold.append(now_threshold)
            return now_skill, None, now_deri_of_diff, now_threshold

        now_deri_of_diff = now_diff-prev_diff
        anomaly_detected = False
        if now_deri_of_diff <= now_threshold:
            anomaly_detected = True

        if anomaly_detected:
            self.anomaly_point.append([len(self.metric_observation), now_deri_of_diff])

        self.metric_observation.append(now_deri_of_diff)
        self.metric_threshold.append(now_threshold)
        return now_skill, anomaly_detected, now_deri_of_diff, now_threshold
