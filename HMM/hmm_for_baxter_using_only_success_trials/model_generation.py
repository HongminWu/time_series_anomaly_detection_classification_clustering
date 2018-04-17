import numpy as np
import ipdb
score_hist_stack = []


def _debug_score_level(s):
    print 'socre level %s: %s'%(len(score_hist_stack), s)

def update_now_score(_now_score):
    record_of_last_level = score_hist_stack[-1]
    record_of_last_level['now_score'] = _now_score
    _debug_score_level('update %s'%(_now_score,))

def init_new_score_level():
    global score_hist_stack
    score_hist_stack.append({
        "best": None,
        "now_score": None,
        "bad_score_count": 0,
    })
    _debug_score_level('new score level')

def update_last_score_level():
    global score_hist_stack

    _debug_score_level('gonna update')

    record_of_this_level = score_hist_stack[-1]
    if record_of_this_level['now_score'] is None:
        _debug_score_level('no now_score for this level, abort update')
        pass
    else:
        if record_of_this_level['best'] is None\
            or record_of_this_level['now_score'] < record_of_this_level['best']:
            _debug_score_level('now_score(%s) < best(%s), will update best'%(record_of_this_level['now_score'], record_of_this_level['best']))
            record_of_this_level['best'] = record_of_this_level['now_score']
        else:
            _debug_score_level('now_score(%s) >= best(%s), bad_score_count %s->%s'%(record_of_this_level['now_score'], record_of_this_level['best'], record_of_this_level['bad_score_count'], record_of_this_level['bad_score_count']+1))
            record_of_this_level['bad_score_count'] += 1

def does_bad_score_count_hit(bad_score_count_max):
    record_of_this_level = score_hist_stack[-1]
    return record_of_this_level['bad_score_count'] >= bad_score_count_max

def clear_last_score_level():
    global score_hist_stack

    _debug_score_level('gonna clear_last_score_level')

    # pass the best to upper level,
    # the upper level will need the best of this level to make a decision

    if len(score_hist_stack) > 1:
        _debug_score_level('will assign best of this level(%s) to now_score of upper level'%(score_hist_stack[-1]['best'],))
        score_hist_stack[-2]['now_score'] = score_hist_stack[-1]['best']

    del score_hist_stack[-1]

def get_model_generator(model_type, model_config):
    global score_hist_stack

    score_hist_stack = []
    if model_type == 'hmmlearn\'s HMM':
        import hmmlearn.hmm
        if type(model_config['hmm_max_train_iteration']) is not list:
            model_config['hmm_max_train_iteration'] = [model_config['hmm_max_train_iteration']]

        if type(model_config['gaussianhmm_covariance_type_string']) is not list:
            model_config['gaussianhmm_covariance_type_string'] = [model_config['gaussianhmm_covariance_type_string']]

        if 'hmm_max_hidden_state_amount' in model_config:
            model_config['hmm_hidden_state_amount'] = range(1, model_config['hmm_max_hidden_state_amount']+1)
        else:
            if type(model_config['hmm_hidden_state_amount']) is not list:
                model_config['hmm_hidden_state_amount'] = [model_config['hmm_hidden_state_amount']]


        for covariance_type in model_config['gaussianhmm_covariance_type_string']:
            for n_iter in model_config['hmm_max_train_iteration']:

                init_new_score_level()
                for n_components in model_config['hmm_hidden_state_amount']:
                    update_last_score_level()
                    if does_bad_score_count_hit(2) and n_components>5:
                        clear_last_score_level()
                        break

                    model = hmmlearn.hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        params="mct",
                        init_params="cmt",
                        n_iter=n_iter)
                    start_prob = np.zeros(n_components)
                    start_prob[0] = 1
                    model.startprob_ = start_prob

                    now_model_config = {
                        "hmm_hidden_state_amount": n_components,
                        "gaussianhmm_covariance_type_string": covariance_type,
                        "hmm_max_train_iteration": n_iter,
                    }
                    yield model, now_model_config

    elif model_type == 'hmmlearn\'s GMMHMM':
        import hmmlearn.hmm
        if type(model_config['hmm_max_train_iteration']) is not list:
            model_config['hmm_max_train_iteration'] = [model_config['hmm_max_train_iteration']]

        if type(model_config['gaussianhmm_covariance_type_string']) is not list:
            model_config['gaussianhmm_covariance_type_string'] = [model_config['gaussianhmm_covariance_type_string']]

        if 'GMM_max_state_amount' in model_config:
            model_config['GMM_state_amount'] = range(1, model_config['GMM_max_state_amount']+1)
        else:
            if type(model_config['GMM_state_amount']) is not list:
                model_config['GMM_state_amount'] = [model_config['GMM_state_amount']]

        if 'hmm_max_hidden_state_amount' in model_config:
            model_config['hmm_hidden_state_amount'] = range(1, model_config['hmm_max_hidden_state_amount']+1)
        else:
            if type(model_config['hmm_hidden_state_amount']) is not list:
                model_config['hmm_hidden_state_amount'] = [model_config['hmm_hidden_state_amount']]


        for covariance_type in model_config['gaussianhmm_covariance_type_string']:
            for n_iter in model_config['hmm_max_train_iteration']:

                init_new_score_level()
                for n_mix in model_config['GMM_state_amount']:
                    update_last_score_level()
                    if does_bad_score_count_hit(2) and n_mix>5:
                        clear_last_score_level()
                        break

                    init_new_score_level()
                    for n_components in model_config['hmm_hidden_state_amount']:
                        update_last_score_level()
                        if does_bad_score_count_hit(2) and n_components>5:
                            clear_last_score_level()
                            break

                        model = hmmlearn.hmm.GMMHMM(
                            n_components=n_components,
                            n_mix=n_mix,
                            covariance_type=covariance_type,
                            params="mct",
                            init_params="cmt",
                            n_iter=n_iter)
                        start_prob = np.zeros(n_components)
                        start_prob[0] = 1
                        model.startprob_ = start_prob

                        now_model_config = {
                            "hmm_hidden_state_amount": n_components,
                            "gaussianhmm_covariance_type_string": covariance_type,
                            "hmm_max_train_iteration": n_iter,
                            "GMM_state_amount": n_mix,
                        }

                        yield model, now_model_config

    elif model_type == 'BNPY\'s HMM':
        import HMM.hmm_wrapper.bnpy_hmm as bnpy_hmm

        if type(model_config['hmm_max_train_iteration']) is not list:
            model_config['hmm_max_train_iteration'] = [model_config['hmm_max_train_iteration']]

        if type(model_config['alloModel']) is not list:
            model_config['alloModel'] = [model_config['alloModel']]

        if type(model_config['obsModel']) is not list:
            model_config['obsModel'] = [model_config['obsModel']]

        if type(model_config['ECovMat']) is not list:
            model_config['ECovMat'] = [model_config['ECovMat']]

        if type(model_config['varMethod']) is not list:
            model_config['varMethod'] = [model_config['varMethod']]

        if 'hmm_max_hidden_state_amount' in model_config:
            model_config['hmm_hidden_state_amount'] = range(1, model_config['hmm_max_hidden_state_amount']+1)
        else:
            if type(model_config['hmm_hidden_state_amount']) is not list:
                model_config['hmm_hidden_state_amount'] = [model_config['hmm_hidden_state_amount']]

        for alloModel in model_config['alloModel']:
            for obsModel in model_config['obsModel']:
                for ECovMat in model_config['ECovMat']:
                    for varMethod in model_config['varMethod']:
                        for n_iter in model_config['hmm_max_train_iteration']:

                            init_new_score_level()
                            for n_components in model_config['hmm_hidden_state_amount']:
                                update_last_score_level()
                                if does_bad_score_count_hit(2) and n_components>5:
                                    clear_last_score_level()
                                    break

                                model = bnpy_hmm.HongminHMM(
                                    alloModel   = alloModel,
                                    obsModel    = obsModel,
                                    ECovMat     = ECovMat,
                                    varMethod   = varMethod,
                                    n_iteration = n_iter,
                                    K           = n_components
                                )

                                now_model_config = {
                                    'alloModel': alloModel,
                                    'obsModel':  obsModel,
                                    'ECovMat' :  ECovMat,
                                    'varMethod': varMethod,
                                    'hmm_hidden_state_amount': n_components,
                                    'hmm_max_train_iteration': n_iter,
                                }

                                yield model, now_model_config

    elif model_type == 'PYHSMM\'s HMM':
        import HMM.hmm_wrapper.pyhsmm_hmm as pyhsmm_hmm
        if type(model_config['hmm_max_train_iteration']) is not list:
            model_config['hmm_max_train_iteration'] = [model_config['hmm_max_train_iteration']]
        if type(model_config['hmm_hidden_state_amount']) is not list:
            model_config['hmm_hidden_state_amount'] = [model_config['hmm_hidden_state_amount']]
        if type(model_config['max_duration_length']) is not list:
            model_config['max_duration_length'] = [model_config['max_duration_length']]

        for hmm_max_train_iteration in  model_config['hmm_max_train_iteration']:
            for hmm_hidden_state_amount in model_config['hmm_hidden_state_amount']:
                for max_duration_length in model_config['max_duration_length']:
                    init_new_score_level()
                    model = hmm_wrapper.pyhsmm_hmm.HongminHSMM(
                        SAVE_FIGURES = False,
                        hmm_max_train_iteration = hmm_max_train_iteration,
                        hmm_hidden_state_amount = hmm_hidden_state_amount,
                        max_duration_length     = max_duration_length,
                        )

                    now_model_config = {
                        'SAVE_FIGURES' : False,
                        'hmm_max_train_iteration' :hmm_max_train_iteration,
                        'hmm_hidden_state_amount' :hmm_hidden_state_amount,
                        'max_duration_length'     :max_duration_length,
                    }
                    yield model, now_model_config
