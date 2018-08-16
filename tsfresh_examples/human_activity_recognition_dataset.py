# This script for the high level convenience function to extract time series from timeseries_container
from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset, load_har_classes
from tsfresh import extract_relevant_features
import coloredlogs, logging
coloredlogs.install()
import ipdb

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler('tsfresh.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info('downloading har dataset')
download_har_dataset()

logger.info('loading har dataset')
df = load_har_dataset()
print df.shape

logger.info('loading har classes')
y = load_har_classes()
print y


'''
# extraction, imputing and filtering at the same time
X = extract_relevant_features(timeseries, y, column_id = 'id', column_sort = 'time')
logger.debug('Congratulations! We can use the features_filtered to train the classification models')
'''
