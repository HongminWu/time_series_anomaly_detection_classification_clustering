from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
     load_robot_execution_failures

import ipdb     
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()
ipdb.set_trace()
