import util
import numpy as np
import ipdb

def score(score_metric, model, X, lengths):
    if score_metric == '_score_metric_worst_stdmeanratio_in_10_slice_':
        slice_10_time_step_log_lik = [[model.score(X[i:i+k*(j-i)/10]) for k in range(1, 11, 1)] for i, j in util.iter_from_X_lengths(X, lengths)]
        matrix = np.matrix(slice_10_time_step_log_lik)
        slice_10_means = abs(matrix.mean(0))
        slice_10_std = matrix.std(0)
        slice_10_stme_ratio = slice_10_std/slice_10_means
        score = slice_10_stme_ratio.max()

    elif score_metric == '_score_metric_last_time_stdmeanratio_':
        final_time_step_log_lik = [
            model.score(X[i:j]) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        matrix = np.matrix(final_time_step_log_lik)
        mean = abs(matrix.mean())
        std = matrix.std()
        score = std/mean

    elif score_metric == '_score_metric_sum_stdmeanratio_using_fast_log_cal_':
        final_time_step_log_lik = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(final_time_step_log_lik) 
        mean_of_log_curve = curve_mat.mean(0)
        std_of_log_curve = curve_mat.std(0)
        score = abs(std_of_log_curve/mean_of_log_curve).mean()

    elif score_metric == '_score_metric_mean_of_std_using_fast_log_cal_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(log_curves_of_all_trials) 
        std_of_log_curve = curve_mat.std(0)
        score = std_of_log_curve.mean()

    elif score_metric == '_score_metric_hamming_distance_using_fast_log_cal_':
        import scipy.spatial.distance as sp_dist
        log_lik = [util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        log_mat         = np.matrix(log_lik)
        std_of_log_mat  = log_mat.std(0)
        mean_of_log_mat = log_mat.mean(0)
        lower_bound     = mean_of_log_mat - 20 * std_of_log_mat
        ipdb.set_trace()
        hamming_score   = sp_dist.hamming(mean_of_log_mat, lower_bound)
        score  = hamming_score
    elif score_metric == '_score_metric_std_of_std_using_fast_log_cal_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(log_curves_of_all_trials) 
        std_of_log_curve = curve_mat.std(0)
        score = std_of_log_curve.std()

    elif score_metric == '_score_metric_mean_of_std_divied_by_final_log_mean_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(log_curves_of_all_trials) 
        std_of_log_curve = curve_mat.std(0)
        mean_of_std = std_of_log_curve.mean()
        final_log_mean = curve_mat.mean(0)[0, -1]
        score = abs(mean_of_std/final_log_mean)
    elif score_metric == '_score_metric_mean_of_std_of_gradient_divied_by_final_log_mean_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        curve_mat = np.matrix(log_curves_of_all_trials) 
        gradient_mat = curve_mat[:, 1:]-curve_mat[:, :-1]
        std_of_log_curve = gradient_mat.std(0)
        mean_of_std = std_of_log_curve.mean()
        final_log_mean = gradient_mat.mean(0)[0, -1]
        score = abs(mean_of_std/final_log_mean)

    elif score_metric == '_score_metric_minus_diff_btw_1st_2ed_emissionprob_':
        score_of_trials = []
        for i, j in util.iter_from_X_lengths(X, lengths):
            framelogprob = util.get_emission_log_prob_matrix(X[i:j], model)

            if framelogprob.shape[1] == 1:
                print 'hidden state amount = 1, but _score_metric_minus_diff_btw_1st_2ed_emissionprob_ wants hidden state amount > 1, so no score for this turn'
                return None

            framelogprob.sort(1)
            diff_btw_1st_2ed_eprob = framelogprob[:, -1]-framelogprob[:, -2]
            score_of_trials.append(np.sum(diff_btw_1st_2ed_eprob)/(j-i))
        score = -np.array(score_of_trials).mean()

    elif score_metric == '_score_metric_minus_diff_btw_1st_2ed(>=0)_divide_maxeprob_emissionprob_':
       
        score_of_trials = []
        for i, j in util.iter_from_X_lengths(X, lengths):
            framelogprob = util.get_emission_log_prob_matrix(X[i:j], model)

            if framelogprob.shape[1] == 1:
                print 'hidden state amount = 1, but _score_metric_minus_diff_btw_1st_2ed_emissionprob_ wants hidden state amount > 1, so no score for this turn'
                return None

            framelogprob.sort(1)
            eprob_2ed = framelogprob[:, -2].clip(min=0)
            eprob_1st = framelogprob[:, -1].clip(min=0)

            max_eprob = np.max(eprob_1st)
            if max_eprob == 0:
                print 'max_eprob = 0, so no score for this turn'
                return None

            diff_btw_1st_2ed_eprob = eprob_1st-eprob_2ed
            score_of_trials.append(np.sum(diff_btw_1st_2ed_eprob)/(max_eprob*(j-i)))

        score = -np.array(score_of_trials).mean()

    elif score_metric == '_score_metric_minus_diff_btw_1st_2ed(delete<0)_divide_maxeprob_emissionprob_':
       
        score_of_trials = []
        for i, j in util.iter_from_X_lengths(X, lengths):
            framelogprob = util.get_emission_log_prob_matrix(X[i:j], model)

            if framelogprob.shape[1] == 1:
                print 'hidden state amount = 1, but _score_metric_minus_diff_btw_1st_2ed_emissionprob_ wants hidden state amount > 1, so no score for this turn'
                return None

            framelogprob.sort(1)
            eprob_2ed = framelogprob[:, -2]
            eprob_1st = framelogprob[:, -1]

            entry_filter = eprob_2ed > 0
            eprob_2ed = eprob_2ed[entry_filter]
            eprob_1st = eprob_1st[entry_filter]

            entry_length = len(eprob_2ed)
            if entry_length == 0:
                print 'entry_length = 0, so no score for this turn'
                return None

            max_eprob = np.max(eprob_1st)
            if max_eprob == 0:
                print 'max_eprob = 0, so no score for this turn'
                return None

            diff_btw_1st_2ed_eprob = eprob_1st-eprob_2ed
            score_of_trials.append(np.sum(diff_btw_1st_2ed_eprob)/(max_eprob*entry_length))

        score = -np.array(score_of_trials).mean()

    elif score_metric == '_score_metric_mean_of_(std_of_(max_emissionprob_of_trials))_':
      
        mat = []  
        for i, j in util.iter_from_X_lengths(X, lengths):
            framelogprob = util.get_emission_log_prob_matrix(X[i:j], model)

            if framelogprob.shape[1] == 1:
                print 'hidden state amount = 1, but _score_metric_minus_diff_btw_1st_2ed_emissionprob_ wants hidden state amount > 1, so no score for this turn'
                return None

            max_omissionprb = framelogprob.max(1)
            mat.append(max_omissionprb)
        mat = np.matrix(mat)
        std_list = mat.std(0)
        score = std_list.mean()
    elif score_metric == '_score_metric_duration_of_(diff_btw_1st_2ed_emissionprob_<_10)_':
       
        score_of_trials = []
        for i, j in util.iter_from_X_lengths(X, lengths):
            framelogprob = util.get_emission_log_prob_matrix(X[i:j], model)

            if framelogprob.shape[1] == 1:
                print 'hidden state amount = 1, but _score_metric_minus_diff_btw_1st_2ed_emissionprob_ wants hidden state amount > 1, so no score for this turn'
                return None

            framelogprob.sort(1)
            diff_btw_1st_2ed_eprob = framelogprob[:, -1]-framelogprob[:, -2]
            duration = (diff_btw_1st_2ed_eprob<10).sum()
            
            score_of_trials.append(float(duration)/(j-i))

        score = np.array(score_of_trials).mean()
    else:
        raise Exception('unknown score metric \'%s\''%(score_metric,))

    return score
    
