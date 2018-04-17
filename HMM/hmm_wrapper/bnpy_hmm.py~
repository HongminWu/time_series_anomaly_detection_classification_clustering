import bnpy
import numpy as np
from scipy.misc import logsumexp
import ipdb

def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        a_log = np.log(a)
        a_log[a <= 0] = 0.0
        return a_log

class HongminHMM():
    def __init__(
        self, 
        alloModel,
        obsModel,
        ECovMat,
        varMethod,
        n_iteration,
        K,
        nTask = 1,
        nBatch = 10,
        convergethr = 0.000000001, #for memoVB
        alpha = 0.5,
        gamma = 5.0,
        sF = 1.0,
        initname = 'randexamples'):

        self.alloModel = alloModel
        self.obsModel = obsModel
        self.ECovMat   = ECovMat
        self.varMethod = varMethod
        self.n_iteration = n_iteration
        self.nTask = nTask
        self.nBatch = nBatch
        self.convergethr = convergethr
        self.alpha = alpha
        self.gamma = gamma
        self.sF = sF
        self.K = K
        self.initname = initname
        self.fwdlattice = None
        self.preSample = None
        self.work_buffer = np.zeros(K)

    def fit(self, X, lengths):
        '''
        # Load dataset from file
        import os
        dataset_path = os.path.join(bnpy.DATASET_PATH, 'mocap6')
        mocap6_dataset = bnpy.data.GroupXData.read_npz(os.path.join(dataset_path, 'dataset.npz'))
        ipdb.set_trace()
        '''
        Xprev      = X[:-1,:]
        X          = X[1:,:]
        doc_range  = list([0])
        doc_range += (np.cumsum(lengths).tolist())
        dataset    = bnpy.data.GroupXData(X, doc_range, None, Xprev)

        # -set the hyperparameters
        model, model_info = bnpy.run(
            dataset,
            self.alloModel,
            self.obsModel,
            self.varMethod,
            nLap = self.n_iteration,
            nTask = self.nTask,
            nBatch = self.nBatch,
            convergethr = self.convergethr,
            alpha = self.alpha,
            gamma = self.gamma,
            sF = self.sF,
            ECovMat = self.ECovMat,
            K = self.K,
            initname = self.initname)
#       self.log_startprob = log_mask_zero(model.allocModel.get_init_prob_vector())
        self.log_startprob = model.allocModel.get_active_comp_probs()
        self.log_startprob = self.log_startprob / sum(self.log_startprob)
        self.log_transmat  = model.allocModel.get_trans_prob_matrix()
        self.model = model
        return self

    def add_one_sample_and_get_loglik(self, sample):
#        if np.array([sample]).shape[0] == 1:
#            sample = np.append([sample],[sample], axis=0)
        if self.preSample is None:
            self.preSample = sample
            return 0
        else:
            Xprev  = np.array([self.preSample])
            X      = np.array([sample])
            self.preSample = sample 
        length = 1
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)

        LP = self.model.calc_local_params(dataset)
        framelogprob = LP['E_log_soft_ev'] #probability of per-component under the posterior
        if self.fwdlattice is None:
            self.fwdlattice = np.zeros((1, self.K))
            for i in range(self.K):
                self.fwdlattice[0,i] = self.log_startprob[i] + framelogprob[0,i]
            print "I am here once"
        else:
            self.fwdlattice = np.append(self.fwdlattice, np.zeros((1, self.K)), axis=0)
            for j in range(self.K):
                for i in range(self.K):
                    self.work_buffer[i] = self.fwdlattice[-2,i] + self.log_transmat[i,j]
                self.fwdlattice[-1,j] = logsumexp(self.work_buffer) + framelogprob[0,j]
        curr_log = logsumexp(self.fwdlattice[-1])
        return curr_log

    def decode(self, X, lengths):
        Xprev      = X[:-1,:]
        X          = X[1:,:]
        doc_range  = list([0])
        doc_range += (np.cumsum(lengths).tolist())
        dataset    = bnpy.data.GroupXData(X, doc_range, None, Xprev)     
   
        from bnpy.allocmodel.hmm.HMMUtil import runViterbiAlg
        from bnpy.util import StateSeqUtil
        initPi =  self.model.allocModel.get_init_prob_vector()
        transPi = self.model.allocModel.get_trans_prob_matrix()
        LP = self.model.calc_local_params(dataset)
        Lik = LP['E_log_soft_ev']
        zHatBySeq = list()
        for n in range(dataset.nDoc):
            start = dataset.doc_range[n]
            stop  = dataset.doc_range[n + 1]
            zHat = runViterbiAlg(Lik[start:stop], initPi, transPi)
            zHatBySeq.append(zHat)
        zHatFlat = StateSeqUtil.convertStateSeq_list2flat(zHatBySeq, dataset)
        return zHatFlat        

    def score(self, X):
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)

        LP = self.model.calc_local_params(dataset)
        log_probability = LP['evidence'] # by HongminWu 28.07-2017
        return log_probability
                   
    def predict_proba(self, X):
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        LP = self.model.calc_local_params(dataset)
        # HongminWu 24.02-2018
        log_probability = LP['resp'] #probability of per-component under the posterior
        return log_probability  

    def calc_log(self, X):
        from scipy.misc import logsumexp
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        LP = self.model.calc_local_params(dataset)
        log = LP['E_log_soft_ev']
        log_curve = [logsumexp(log[i]) for i in range(len(log))]
        log_curve = np.cumsum(log_curve)
        return log_curve
        
