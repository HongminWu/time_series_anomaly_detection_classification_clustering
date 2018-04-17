import pyhsmm
import pyhsmm.basic.distributions as distributions
from  pyhsmm.util.text import progprint_xrange
import numpy as np
import copy, os
import ipdb

class HongminHSMM():
    def __init__(
        self, 
        SAVE_FIGURES,
        hmm_max_train_iteration,
        hmm_hidden_state_amount,
        max_duration_length,
        ):
        self.SAVE_FIGURES = SAVE_FIGURES
        self.hmm_max_train_iteration = hmm_max_train_iteration
        self.hmm_hidden_state_amount = hmm_hidden_state_amount
        self.max_duration_length = max_duration_length
        
    def get_current_state_number(state_no):
        return state_no

    def fit(self, data, lengths):
        obs_dim = data.shape[1]
        Nmax = self.hmm_hidden_state_amount
        dur_length = self.max_duration_length
        maxIter = self.hmm_max_train_iteration
        data_length = lengths

        obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}
        dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}
        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]

        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6., # thesecan matter; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
        
        posteriormodel.add_data(data,trunc = dur_length) # duration truncation speeds things up when it's possible
        for idx in progprint_xrange(maxIter):
            posteriormodel.resample_model()
            print ('-resampling-%d/%d'%(idx, maxIter))

        self.model = posteriormodel

        return self

    def score(self, data, **kwargs):
        if isinstance(data, np.ndarray):
            self.model.add_data(data=data, generate=False, **kwargs)
            return self.model.states_list.pop().log_likelihood()  
        else:
            print 'Please convert the input data to np.ndarray'
            return 'Fail to calculate the log_likelihood using hdp-hsmm model'

    def calc_log(self, data, **kwargs):
        loglik = list()
        data = list(data)
        if data is not None and isinstance(data, list):
            for d in data:
                self.model.add_data(data=d, generate=False, **kwargs)
                loglik.append(self.model.states_list.pop().log_likelihood())
            log_curve = np.cumsum(loglik)
            return log_curve
        else:
            print 'Please convert the input data to list'
            return 'Fail to calculate the log_likelihood using hdp-hsmm model'
#        log_curve = [logsumexp(log[i]) for i in range(len(log))]
#        log_curve = np.cumsum(log_curve)
