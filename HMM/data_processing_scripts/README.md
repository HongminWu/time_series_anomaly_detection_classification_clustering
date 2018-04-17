# Why this repo exists

I'm currently collecting robot data from Rethink Robotisc Baxter. And this data belongs to a task named "pick and place". And I need a place to save them. So here it is.

### What is this data used for?

This data will be used for training models that can detect anomalies during task execution. 
# data_processing_scripts INFO
* All the script can be -h command to find the specific usage.

1. Transfer the rosbag *.bag file to *.csv files
>> python process_rosbag_patch -d ./PATH_TO_FOLDER

