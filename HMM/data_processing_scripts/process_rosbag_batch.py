import os
import rospy

def put_bag_into_folder(base_folder):
    files = os.listdir(options.base_folder)
    for f in files:
        if not f.endswith('bag'):
            continue
        folder = os.path.join(base_folder, f[:-4])
        if not os.path.isdir(folder):
            os.makedirs(folder)

        now_path = os.path.join(base_folder, f)
        new_path = os.path.join(folder, f)
    
        from shutil import copyfile
        copyfile(now_path, new_path)

def process_bag_to_csv(base_folder, topic_names):
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    files = os.listdir(options.base_folder)
    for f in files:
        path = os.path.join(base_folder, f)
        if not os.path.isdir(path):
            continue
        if f.startswith("bad"):
            continue

        bag_file_path = os.path.join(path, f+'.bag')

        import tuned_rosbag_to_csv
        
        fake_options = Bunch(
            start_time = None,
            end_time = None,
            topic_names = topic_names, 
            output_file_format="%t.csv",
            header = True)

        print 'gonna process', bag_file_path
        tuned_rosbag_to_csv.bag_to_csv(fake_options, os.path.abspath(bag_file_path))

if __name__ == "__main__":
    rospy.init_node("rosbag_batch_processor")
    from optparse import OptionParser
    usage = "usage: %prog -d base_folder_path"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--base-folder",
        action="store", type="string", dest="base_folder",
        help="provide a base folder which will have this structure: ./*.bag")
    parser.add_option("-t", "--topic",
        action="store", type="string", dest="topic",
        help="topic seperated by comma")
    (options, args) = parser.parse_args()

    (options, args) = parser.parse_args()
    if options.base_folder is None:
        parser.error("no base_folder")

    topic_names = ['/'+i for i in options.topic.split(',')]

    print topic_names

    put_bag_into_folder(options.base_folder)
    process_bag_to_csv(options.base_folder, topic_names)
    
