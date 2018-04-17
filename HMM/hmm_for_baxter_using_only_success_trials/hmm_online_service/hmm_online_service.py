#!/usr/bin/env python

import rospy
def run(
    interested_data_fields, 
    model_save_path, 
    state_amount, 
    anomaly_detection_metric,
):
    from multiprocessing import Queue
    import data_stream_handler_process
    com_queue_of_receiver = Queue()
    process_receiver = data_stream_handler_process.TagMultimodalTopicHandler(
        interested_data_fields,
        com_queue_of_receiver,
    )

    import state_classification_then_anomaly_detection_process
    com_queue_of_anomaly_detection = Queue()
    process_anomaly_detection = state_classification_then_anomaly_detection_process.IdSkillThenDetectAnomaly(
        model_save_path,
        state_amount,
        anomaly_detection_metric,
        com_queue_of_anomaly_detection,    
    )



    process_receiver.start()
    process_anomaly_detection.start()

    while not rospy.is_shutdown():
        try:
            latest_data_tuple = com_queue_of_receiver.get(1)
        except Queue.Empty:
            continue
        latest_data_tuple = com_queue_of_receiver.get()
        com_queue_of_anomaly_detection.put(latest_data_tuple)


    process_receiver.shutdown()
    process_anomaly_detection.shutdown()
