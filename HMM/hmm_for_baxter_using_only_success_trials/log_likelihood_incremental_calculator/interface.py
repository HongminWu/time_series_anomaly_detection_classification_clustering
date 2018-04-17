import numpy as np
import util
from scipy.special import logsumexp

class HmmlearnModelIncrementalLoglikCalculator(object):
    def __init__(self, model):
        self.model = model
        self.n_components = model.n_components
        self.log_transmat = util.log_mask_zero(model.transmat_)
        self.log_startprob = util.log_mask_zero(model.startprob_)
        self.fwdlattice = None
        self.work_buffer = np.zeros(self.n_components)

    def add_one_sample_and_get_loglik(self, sample):
        framelogprob = self.model._compute_log_likelihood(sample)
        if self.fwdlattice is None:
            self.fwdlattice = np.zeros((1, self.n_components))
            for i in range(self.n_components):
                self.fwdlattice[0, i] = self.log_startprob[i] + framelogprob[0, i]
        else:
            self.fwdlattice = np.append(self.fwdlattice, np.zeros((1, self.n_components)), axis=0)
            for j in range(self.n_components):
                for i in range(self.n_components):
                    self.work_buffer[i] = self.fwdlattice[-2, i] + self.log_transmat[i, j]

                self.fwdlattice[-1, j] = logsumexp(self.work_buffer) + framelogprob[0, j]

        return logsumexp(self.fwdlattice[-1])

def get_calculator(model):
    import hmmlearn.hmm
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        return HmmlearnModelIncrementalLoglikCalculator(model)

    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))
