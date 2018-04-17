import multiprocessing
from sklearn.externals import joblib
import rospy
from birl_sim_examples.msg import (
    Hmm_Log,
)
import constant
import util

class AnomalyDetector(multiprocessing.Process):
    def __init__(self, model_save_path, state_amount, deri_threshold, com_queue):
        multiprocessing.Process.__init__(self)     

        self.deri_threshold = deri_threshold
        self.com_queue = com_queue


        list_of_expected_log = joblib.load(model_save_path+'/expected_log.pkl')
        list_of_threshold = joblib.load(model_save_path+'/threshold.pkl')
        list_of_std_of_log = joblib.load(model_save_path+"/std_of_log.pkl")

        model_group_by_state = {}
        expected_log_group_by_state = {}
        threshold_group_by_state = {} 
        std_log_group_by_state = {}

        for state_no in range(1, state_amount+1):
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))

            # the counterpart simply pushes these data into a list, so for state 1, its data is located in 0.
            expected_log_group_by_state[state_no] = list_of_expected_log[state_no-1]
            threshold_group_by_state[state_no] = list_of_threshold[state_no-1]
            std_log_group_by_state[state_no] = list_of_std_of_log[state_no-1]

        self.model_group_by_state = model_group_by_state
        self.expected_log_group_by_state = expected_log_group_by_state
        self.std_log_group_by_state = std_log_group_by_state 
        self.threshold_group_by_state = threshold_group_by_state 

    def get_anomaly_detection_msg(self, arrived_data, arrived_state, data_header):

        hmm_log = Hmm_Log()

        arrived_length = len(arrived_data)
        if arrived_length < 10:
            return hmm_log

        if arrived_state <= 0:
            return hmm_log 

        try:    
            self.expected_log_group_by_state[arrived_state][arrived_length-1]
            self.threshold_group_by_state[arrived_state][arrived_length-1]
            log_curve = util.fast_log_curve_calculation(arrived_data, self.model_group_by_state[arrived_state])
            
            idx = arrived_length-2
            prev_threshold = self.threshold_group_by_state[arrived_state][idx]
            prev_log_lik = log_curve[idx]
            prev_diff = prev_log_lik-prev_threshold

            idx = arrived_length-1
            now_threshold = self.threshold_group_by_state[arrived_state][idx]
            now_log_lik = log_curve[idx]
            now_diff = now_log_lik-now_threshold

            deri_of_diff = now_diff-prev_diff
    
            if abs(deri_of_diff) < self.deri_threshold:
                hmm_log.event_flag = 1
            else:
                hmm_log.event_flag = 0
            hmm_log.current_log.data = now_log_lik
            hmm_log.expected_log.data = self.expected_log_group_by_state[arrived_state][idx]
            hmm_log.threshold.data = now_threshold 
            hmm_log.diff_btw_curlog_n_thresh.data = now_diff
            hmm_log.deri_of_diff_btw_curlog_n_thresh.data = deri_of_diff
            hmm_log.header = data_header 
            hmm_log.header.stamp = rospy.Time.now() 
        except IndexError:
            rospy.loginfo('received data is longer than the threshold. DTW needed.')

        return hmm_log

    def run(self):

        rospy.init_node("", anonymous=True)
        anomaly_topic_pub = rospy.Publisher("/hmm_online_result", Hmm_Log, queue_size=1000)
        rospy.loginfo('/hmm_online_result published')

        arrived_state = 0 
        arrived_data = []
        while not rospy.is_shutdown():
            try:
                latest_data_tuple = self.com_queue.get(1)
            except multiprocessing.Queue.Empty:
                continue

            data_frame = latest_data_tuple[constant.data_frame_idx]
            smach_state = latest_data_tuple[constant.smach_state_idx]
            data_header = latest_data_tuple[constant.data_header_idx]
            if smach_state <= 0:
                arrived_state = 0 
                arrived_data = []
            elif smach_state != arrived_state:
                arrived_state = smach_state
                arrived_data = [data_frame]
            else:
                arrived_data.append(data_frame)
            hmm_log = self.get_anomaly_detection_msg(arrived_data, arrived_state, data_header)
            anomaly_topic_pub.publish(hmm_log)
            print 'pub'
            
