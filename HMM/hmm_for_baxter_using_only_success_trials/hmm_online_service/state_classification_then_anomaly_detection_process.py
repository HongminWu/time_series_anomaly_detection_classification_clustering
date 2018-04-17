import multiprocessing
from sklearn.externals import joblib
import rospy
import constant
import util
import anomaly_detection.interface
import numpy as np
from std_msgs.msg import (
    Header,
    Int8,
    Float64,
)

class IdSkillThenDetectAnomaly(multiprocessing.Process):
    def __init__(
        self, 
        model_save_path, 
        state_amount, 
        anomaly_detection_metric,
        com_queue,
    ):
        multiprocessing.Process.__init__(self)     

        self.com_queue = com_queue
        self.detector = anomaly_detection.interface.get_anomaly_detector(
            model_save_path, 
            state_amount,
            anomaly_detection_metric,
        )
        self.anomaly_detection_metric = anomaly_detection_metric

    def run(self):

        rospy.init_node("", anonymous=True)
        anomaly_detection_signal_pub = rospy.Publisher("/anomaly_detection_signal", Header, queue_size=100)
        anomaly_detection_metric_pub = rospy.Publisher("/anomaly_detection_metric_%s"%self.anomaly_detection_metric, Float64, queue_size=100)
        anomaly_detection_threshold_pub = rospy.Publisher("/anomaly_detection_threshold_%s"%self.anomaly_detection_metric, Float64, queue_size=100)
        identified_skill_pub = rospy.Publisher("/identified_skill_%s"%self.anomaly_detection_metric, Int8, queue_size=100)
        rospy.loginfo('/hmm_online_result published')

        arrived_state = 0 
        while not rospy.is_shutdown():
            try:
                latest_data_tuple = self.com_queue.get(1)
            except multiprocessing.Queue.Empty:
                continue


            smach_state = latest_data_tuple[constant.smach_state_idx]
            if smach_state <= 0:
                self.detector.reset()
                continue

            data_frame = latest_data_tuple[constant.data_frame_idx]
            data_header = latest_data_tuple[constant.data_header_idx]

            now_skill, anomaly_detected, metric, threshold = self.detector.add_one_smaple_and_identify_skill_and_detect_anomaly(np.array(data_frame).reshape(1,-1), now_skill=smach_state)

            rospy.loginfo("anomaly_detected:%s"%anomaly_detected)
            if anomaly_detected:
                rospy.loginfo("anomaly_detected:%s"%anomaly_detected)
                anomaly_detection_signal_pub.publish(data_header) 
    
            if now_skill is not None:
                identified_skill_pub.publish(now_skill) 

            if metric is not None and threshold is not None:
                anomaly_detection_metric_pub.publish(metric)
                anomaly_detection_threshold_pub.publish(threshold)
