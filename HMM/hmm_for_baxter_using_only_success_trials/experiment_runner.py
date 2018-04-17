from optparse import OptionParser
import training_config
import util

def build_parser():
    parser = OptionParser()

    parser.add_option(
        "--test-if-parallelity-can-be-restored",
        action="store_true", 
        dest="test_if_parallelity_can_be_restored",
        default = False,
        help="read the option name please.")

    parser.add_option(
        "--tamper-transmat",
        action="store_true", 
        dest="tamper_transmat",
        default = False,
        help="read the option name please.")

    parser.add_option(
        "--tamper-startprob",
        action="store_true", 
        dest="tamper_startprob",
        default = False,
        help="read the option name please.")


    parser.add_option(
        "--test-if-gradient-can-detect-state-switch",
        action="store_true", 
        dest="test_if_gradient_can_detect_state_switch",
        default = False,
        help="read the option name please.")

    parser.add_option(
        "--compare-anomay-model-with-normal-model",
        action="store_true", 
        dest="compare_anomay_model_with_normal_model",
        default = False,
        help="read the option name please.")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()

    if options.test_if_parallelity_can_be_restored:
        print 'gonna test_if_parallelity_can_be_restored'
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import experiment_scripts.test_if_parallelity_can_be_restored
        experiment_scripts.test_if_parallelity_can_be_restored.run(
            model_save_path = training_config.model_save_path,
            trials_group_by_folder_name = trials_group_by_folder_name,
            parsed_options=options)

    if options.test_if_gradient_can_detect_state_switch:
        print 'gonna test_if_gradient_can_detect_state_switch'
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import experiment_scripts.test_if_gradient_can_detect_state_switch
        experiment_scripts.test_if_gradient_can_detect_state_switch.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            trials_group_by_folder_name = trials_group_by_folder_name,
            state_order_group_by_folder_name = state_order_group_by_folder_name,
            parsed_options=options)

    if options.compare_anomay_model_with_normal_model:
        print 'gonna compare_anomay_model_with_normal_model'
        normal_trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)
        anomaly_trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config, data_class='anomaly')

        import experiment_scripts.compare_anomay_model_with_normal_model
        experiment_scripts.compare_anomay_model_with_normal_model.run(
            training_config=training_config,
            parsed_options=options)
