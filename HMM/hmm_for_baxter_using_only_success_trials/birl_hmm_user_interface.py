from optparse import OptionParser
import training_config
import util
import os
import ipdb

def warn(*args, **kwargs):
    if 'category' in kwargs and kwargs['category'] == DeprecationWarning:
        pass
    else:
        for arg in args:
            print arg
import warnings
warnings.warn = warn
        
def build_parser():
    parser = OptionParser()
    parser.add_option(
        "--train-model",
        action="store_true", 
        dest="train_model",
        default = False,
        help="True if you want to train HMM models.")

    parser.add_option(
        "--train-anomaly-model",
        action="store_true", 
        dest="train_anomaly_model",
        default = False,
        help="True if you want to train HMM anomaly models.")


    parser.add_option(
        "--learn_threshold_for_log_likelihood",
        action="store_true", 
        dest="learn_threshold_for_log_likelihood",
        default = False,
        help="True if you want to learn_threshold_for_log_likelihood.")

    parser.add_option(
        "--learn_threshold_for_gradient_of_log_likelihood",
        action="store_true", 
        dest="learn_threshold_for_gradient_of_log_likelihood",
        default = False,
        help="True if you want to learn_threshold_for_gradient_of_log_likelihood.")

    parser.add_option(
        "--learn_threshold_for_deri_of_diff",
        action="store_true", 
        dest="learn_threshold_for_deri_of_diff",
        default = False,
        help="True if you want to learn_threshold_for_deri_of_diff.")

    parser.add_option(
        "--train-derivative-threshold",
        action="store_true", 
        dest="train_derivative_threshold",
        default = False,
        help="True if you want to train derivative threshold.")

    parser.add_option(
        "--online-service",
        action="store_true", 
        dest="online_service",
        default = False,
        help="True if you want to run online anomaly detection and online state classification.")

    parser.add_option(
        "--hidden-state-log-prob-plot",
        action="store_true", 
        dest="hidden_state_log_prob_plot",
        default = False,
        help="True if you want to plot hidden state log prob.")

    parser.add_option(
        "--trial-log-likelihood-plot",
        action="store_true", 
        dest="trial_log_likelihood_plot",
        default = False,
        help="True if you want to plot trials' log likelihood.")

    parser.add_option(
        "--emission-log-prob-plot",
        action="store_true", 
        dest="emission_log_prob_plot",
        default = False,
        help="True if you want to plot emission log prob.")

    parser.add_option(
        "--trial-log-likelihood-gradient-plot",
        action="store_true", 
        dest="trial_log_likelihood_gradient_plot",
        default = False,
        help="True if you want to plot trials' log likelihood gradient.")

    parser.add_option(
        "--check-if-score-metric-converge-loglik-curves",
        action="store_true", 
        dest="check_if_score_metric_converge_loglik_curves",
        default = False,
        help="True if you want to check_if_score_metric_converge_loglik_curves.")

    parser.add_option(
        "--check_if_viterbi_path_grow_incrementally",
        action="store_true", 
        dest="check_if_viterbi_path_grow_incrementally",
        default = False,
        help="True if you want to check_if_viterbi_path_grow_incrementally.")

    parser.add_option(
        "--plot_skill_identification_and_anomaly_detection",
        action="store_true", 
        dest="plot_skill_identification_and_anomaly_detection",
        default = False,
        help="True if you want to plot_skill_identification_and_anomaly_detection.")

    parser.add_option("--trial_class",
        action="store", 
        type="string", 
        dest="trial_class",
        default = None,)

    parser.add_option("--skill_identification",
        action="store_true", 
        dest="skill_identification",
        default = False,
        help = 'True for skill identification')

    parser.add_option("--anomaly_identification",
                      action = "store_true",
                      dest = "anomaly_identification",
                      default = False,
                      help = "True for anomaly identification")
    return parser

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()

    util.inform_config(training_config)

    if options.train_model is True:
        print "gonna train HMM model."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import hmm_model_training
        hmm_model_training.run(
            model_save_path = training_config.model_save_path,
            model_type = training_config.model_type_chosen,
            model_config = training_config.model_config,
            score_metric = training_config.score_metric,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.train_anomaly_model is True:
        import hmm_model_training
        print "gonna train HMM anomaly_model."

        folders = os.listdir(training_config.anomaly_raw_data_path)
        for fo in folders:
            path = os.path.join(training_config.anomaly_raw_data_path, fo)
            if not os.path.isdir(path):
                continue
            data_path = os.path.join(training_config.anomaly_raw_data_path, fo)
            anomaly_trials_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, data_path) 
            anomaly_model_path = os.path.join(training_config.anomaly_model_save_path, 
                                                   fo, 
                                                   training_config.config_by_user['data_type_chosen'], 
                                                   training_config.config_by_user['model_type_chosen'], 
                                                   training_config.model_id)
            hmm_model_training.run(
                model_save_path = anomaly_model_path,
                model_type = training_config.model_type_chosen,
                model_config = training_config.model_config,
                score_metric = training_config.score_metric,
                trials_group_by_folder_name =  anomaly_trials_group_by_folder_name)

            import learn_threshold_for_log_likelihood
            print "gonna calculate the expected log-likelihood of all the non-anomalous trials"
            learn_threshold_for_log_likelihood.run(model_save_path = anomaly_model_path,
                                                   figure_save_path = training_config.anomaly_identification_figure_path,
                                                   threshold_c_value = training_config.threshold_c_value,
                                                   trials_group_by_folder_name = anomaly_trials_group_by_folder_name)

    if options.anomaly_identification is True:
        print "gonna test the anomaly identification"
        import anomaly_identification
        anomaly_identification.run(
            anomaly_data_path_for_testing = training_config.anomaly_data_path_for_testing,
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.anomaly_identification_figure_path,)
        
    if options.learn_threshold_for_log_likelihood is True:
        print "gonna learn_threshold_for_log_likelihood."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import learn_threshold_for_log_likelihood
        learn_threshold_for_log_likelihood.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.learn_threshold_for_gradient_of_log_likelihood is True:
        print "gonna learn_threshold_for_gradient_of_log_likelihood."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import learn_threshold_for_gradient_of_log_likelihood
        learn_threshold_for_gradient_of_log_likelihood.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.learn_threshold_for_deri_of_diff is True:
        print "gonna learn_threshold_for_deri_of_diff."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import learn_threshold_for_deri_of_diff
        learn_threshold_for_deri_of_diff.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.train_derivative_threshold is True:
        print "gonna train derivative threshold."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import derivative_threshold_training 
        derivative_threshold_training.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.online_service is True:
        print "gonna run online service."
        import hmm_online_service.hmm_online_service as hmm_online_service

        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)
        one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
        state_amount = len(one_trial_data_group_by_state)

        hmm_online_service.run(
            interested_data_fields = training_config.interested_data_fields,
            model_save_path = training_config.model_save_path,
            state_amount = state_amount,
            anomaly_detection_metric = training_config.anomaly_detection_metric,
        ) 
            
    if options.hidden_state_log_prob_plot is True:
        print "gonna plot hidden state log prob."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import hidden_state_log_prob_plot 
        hidden_state_log_prob_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.trial_log_likelihood_plot is True:
        print "gonna plot trials' log likelihood."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import trial_log_likelihood_plot
        trial_log_likelihood_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)



    if options.trial_log_likelihood_gradient_plot is True:

        if options.trial_class is None:
            raise Exception("options.trial_class is needed for options.trial_log_likelihood_gradient_plot")

        data_class = options.trial_class

        print "gonna do trial_log_likelihood_gradient_plot."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config, data_class=data_class)

        import trial_log_likelihood_gradient_plot 
        trial_log_likelihood_gradient_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name,
            data_class=data_class,
        )

    if options.check_if_score_metric_converge_loglik_curves is True:
        print "gonna check_if_score_metric_converge_loglik_curves."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import check_if_score_metric_converge_loglik_curves
        check_if_score_metric_converge_loglik_curves.run(
            model_save_path = training_config.model_save_path,
            model_type = training_config.model_type_chosen,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.check_if_viterbi_path_grow_incrementally is True:
        print "gonna check_if_viterbi_path_grow_incrementally."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import check_if_viterbi_path_grow_incrementally
        check_if_viterbi_path_grow_incrementally.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            trials_group_by_folder_name = trials_group_by_folder_name,
            options=options,
        )

    if options.emission_log_prob_plot is True:
        print "gonna plot emission log prob."
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import emission_log_prob_plot 
        emission_log_prob_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.plot_skill_identification_and_anomaly_detection is True:
        if options.trial_class is None:
            raise Exception("options.trial_class is needed for options.plot_skill_identification_and_anomaly_detection")

        data_class = options.trial_class
        if data_class == 'success':
            data_path = training_config.success_path
        elif data_class == 'anomaly':
            data_path = training_config.anomaly_data_path
        elif data_class == 'test_success':
            data_path = training_config.test_success_data_path
        else:
            raise Exception("unknown data class %s"%data_class)

        
        import plot_skill_identification_and_anomaly_detection
        plot_skill_identification_and_anomaly_detection.run(
            model_save_path = training_config.model_save_path, 
            figure_save_path = training_config.figure_save_path,
            anomaly_detection_metric = training_config.anomaly_detection_metric,
            trial_class=options.trial_class,
            data_path=data_path,
            interested_data_fields = training_config.interested_data_fields,
        )

    if options.skill_identification is True:
        trials_group_by_folder_name, state_order_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)
        import skill_identification
        skill_identification.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            trials_group_by_folder_name = trials_group_by_folder_name,)

    

