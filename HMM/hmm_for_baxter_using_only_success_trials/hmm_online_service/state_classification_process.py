
class HMMThreadForStateClassification(threading.Thread):
    def __init__(self, model_save_path, state_amount):
        threading.Thread.__init__(self) 

        model_group_by_state = {}
        for state_no in range(1, state_amount+1):
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        self.model_group_by_state = model_group_by_state


    def get_state_classification_msgs_group_by_state(self):
        global data_arr
        global hmm_state

        data_arr_copy = copy.deepcopy(data_arr)

        data_index = len(data_arr_copy)
        if data_index == 0:
            return None

        hmm_state_copy = hmm_state

        if hmm_state_copy <= 0:
            return None

        msgs_group_by_state = {}
        for state_no in self.model_group_by_state:
            msgs_group_by_state[state_no] = self.model_group_by_state[state_no].score(data_arr_copy)
        return msgs_group_by_state

    def run(self):
        publishing_rate = 10 
        r = rospy.Rate(publishing_rate)

        state_log_curve_pub = {}
        for state_no in self.model_group_by_state:
            topic_name = "/hmm_log_curve_of_state_%s"%(state_no,)
            state_log_curve_pub[state_no] = rospy.Publisher(topic_name, Float32, queue_size=10)
            rospy.loginfo('%s published'%(topic_name,))
        

        while not rospy.is_shutdown():

            msgs_group_by_state = self.get_state_classification_msgs_group_by_state()
            if msgs_group_by_state is not None:
                for state_no in msgs_group_by_state:
                    state_log_curve_pub[state_no].publish(msgs_group_by_state[state_no]) 
            else:
                for state_no in state_log_curve_pub:
                    state_log_curve_pub[state_no].publish(0) 

            r.sleep()

        return 0
