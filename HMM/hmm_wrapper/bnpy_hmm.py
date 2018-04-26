import bnpy
import numpy as np
from scipy.special import logsumexp
import bnpy
import ipdb
  
class HongminHMM():
    def __init__(
        self, 
        alloModel,
        obsModel,
        ECovMat,
        varMethod,
        n_iteration,
        K,
        nTask       = 1,
        nBatch      = 10,
        convergethr = 0.000000001, #for memoVB
        alpha       = 0.5, 
        gamma       = 5.0,  # top-level Dirichlet concentration parameter
        transAlpha  = 5.0,  # trans-level Dirichlet concentration parameter transAlpha
        startAlpha  = 10.0, # starting-state Dirichlet concentration parameter startAlpha
        sF          = 1.0,
        hmmKappa    = 50.0,
        initname    = 'randexamples',
        printEvery  = 10):

        self.alloModel    = alloModel
        self.obsModel     = obsModel
        self.ECovMat      = ECovMat
        self.varMethod    = varMethod
        self.n_iteration  = n_iteration
        self.nTask        = nTask
        self.nBatch       = nBatch
        self.convergethr  = convergethr
        self.alpha        = alpha
        self.gamma        = gamma
        self.transAlpha   = transAlpha
        self.startAlpha   = startAlpha
        self.hmmKappa     = hmmKappa
        self.sF           = sF
        self.K            = K
        self.initname     = initname
        self.fwdlattice   = None
        self.preSample    = None
        self.work_buffer  = np.zeros(K)
        self.printEvery   = printEvery

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
            nLap        = self.n_iteration,
            nTask       = self.nTask,
            nBatch      = self.nBatch,
            convergethr = self.convergethr,
            alpha       = self.alpha,
            gamma       = self.gamma,            
            transAlpha  = self.transAlpha,
            startAlpha  = self.startAlpha,
            hmmKappa    = self.hmmKappa, 
            sF          = self.sF,
            ECovMat     = self.ECovMat,
            K           = self.K,
            initname    = self.initname)
        self.dataset    = dataset
        self.log_startprob = np.log(model.allocModel.get_init_prob_vector())
        self.log_transmat  = np.log(model.allocModel.get_trans_prob_matrix())
        self.model         = model
        return self

    def add_one_sample_and_get_loglik(self, sample):
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
        framelogprob = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)
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
        from bnpy.util import StateSeqUtil
        zHatBySeq = list()
        probBySeq = list()
        logBySeq  = list()
        for n in range(dataset.nDoc):
            logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset.make_subset([n]))
            zHat =  self.runViterbiAlg(logSoftEv, self.log_startprob, self.log_transmat)
            zHatBySeq.append(zHat)
            fmsg, margPrObs = bnpy.allocmodel.hmm.HMMUtil.FwdAlg(np.exp(self.log_startprob), np.exp(self.log_transmat), np.exp(logSoftEv))
            
            loglik = np.log(margPrObs) # 1-D, T
            logBySeq.append(loglik)
            #FwdBwdAlg(PiInit, PiMat, logSoftEv)
            resp, respPair, logMargPrSeq = bnpy.allocmodel.hmm.HMMUtil.FwdBwdAlg(np.exp(self.log_startprob), np.exp(self.log_transmat), logSoftEv)
            probBySeq.append(resp)
        # zHatFlat = StateSeqUtil.convertStateSeq_list2flat(zHatBySeq, dataset)
        return zHatBySeq, probBySeq, logBySeq
                   
    def predict_proba(self, X):
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)            
        #FwdBwdAlg(PiInit, PiMat, logSoftEv)
        resp, respPair, logMargPrSeq = bnpy.allocmodel.hmm.HMMUtil.FwdBwdAlg(np.exp(self.log_startprob), np.exp(self.log_transmat), logSoftEv)
        return resp

    def get_emission_log_prob_matrix(self, X):    
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)            
        return logSoftEv
    
    def score(self, X):
        '''
        Compute the the log-likelihood p(x_T | x_1, x_2,....,x_{T-1})
        '''
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset   = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)
        # FwdAlg(PiInit, PiMat, SoftEv)        
        fmsg, margPrObs = bnpy.allocmodel.hmm.HMMUtil.FwdAlg(np.exp(self.log_startprob), np.exp(self.log_transmat), np.exp(logSoftEv))
        loglik = np.log(margPrObs)
        log_curve = np.cumsum(loglik)
        logprob = log_curve[-1]
        return logprob
    
    def calc_log(self, X):
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        
        logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)
        # FwdAlg(PiInit, PiMat, SoftEv)
        fmsg, margPrObs = bnpy.allocmodel.hmm.HMMUtil.FwdAlg(np.exp(self.log_startprob), np.exp(self.log_transmat), np.exp(logSoftEv))
        loglik = np.log(margPrObs)
        log_curve = np.cumsum(loglik)        
        return log_curve

    def runViterbiAlg(self, logSoftEv, logPi0, logPi):
        
        ''' Run viterbi algorithm to estimate MAP states for single sequence.
        Args
        ------
        logSoftEv : 2D array, T x K
            log soft evidence matrix
            each row t := log p( x[t] | z[t]=k )
        pi0 : 1D array, length K
            initial state probability vector, sums to one
        pi : 2D array, shape K x K
            j-th row is is transition probability vector for state j
        Returns
        ------
        zHat : 1D array, length T, representing the MAP state sequence
            zHat[t] gives the integer label {1, 2, ... K} of state at timestep t
        '''

        from bnpy.util import EPS
        
        if np.any(logPi0 > 0):
            logPi0 = np.log(logPi0 + EPS)
        if np.any(logPi > 0):
            logPi = np.log(logPi + EPS)
        T, K = np.shape(logSoftEv)

        # ScoreTable : 2D array, shape T x K
        #   entry t,k gives the log probability of reaching state k at time t
        #   under the most likely path
        ScoreTable = np.zeros((T, K))

        # PtrTable : 2D array, size T x K
        #   entry t,k gives the integer id of the state j at timestep t-1
        #   which would be part of the most likely path to reaching k at t
        PtrTable = np.zeros((T, K))

        ScoreTable[0, :] = logSoftEv[0] + logPi0
        PtrTable[0, :] = -1

        ids0toK = range(K)
        for t in xrange(1, T):
            ScoreMat_t = logPi + ScoreTable[t - 1, :][:, np.newaxis]
            bestIDvec = np.argmax(ScoreMat_t, axis=0)

            PtrTable[t, :] = bestIDvec
            ScoreTable[t, :] = logSoftEv[t, :] \
                + ScoreMat_t[(bestIDvec, ids0toK)]

        # Follow backward pointers to construct most likely state sequence
        z = np.zeros(T)
        z[-1] = np.argmax(ScoreTable[-1])
        for t in reversed(xrange(T - 1)):
            z[t] = PtrTable[int(t + 1), int(z[t + 1])]
        return z

    
    def show_single_sequence(self,
            seq_id,
            zhat_T     = None,
            z_img_cmap = None,
            ylim       = [-12, 12],
            K          = 5,
            left       = 0.2,
            bottom     = 0.2,
            right      = 0.8,
            top        = 0.95):

        '''
        Usage:
        1. show the single sequence in dataset: show_single_sequence(0)
        2. show the sequence and zhat: show_single_sequence(0, zhat)
        '''

        import matplotlib
        from matplotlib import pylab
        dataset = self.dataset
        
        K = self.K
        if z_img_cmap is None:
            z_img_cmap = matplotlib.cm.get_cmap('Set1', K)
        if zhat_T is None:
            nrows = 1
        else:
            nrows = 2
        fig_h, ax_handles = pylab.subplots(
            nrows=nrows, ncols=1, sharex=True, sharey=False)
        ax_handles = np.atleast_1d(ax_handles).flatten().tolist()
        start = dataset.doc_range[seq_id]
        stop = dataset.doc_range[seq_id + 1]
        # Extract current sequence
        # as a 2D array : T x D (n_timesteps x n_dims)
        curX_TD = dataset.X[start:stop]
        for dim in xrange(dataset.dim):
            ax_handles[0].plot(curX_TD[:, dim], '.-')
        ax_handles[0].set_ylabel('angle')
        ax_handles[0].set_ylim(ylim)
        z_img_height = int(np.ceil(ylim[1] - ylim[0]))
        pylab.subplots_adjust(
            wspace=0.1,
            hspace=0.1,
            left=left, right=right,
            bottom=bottom, top=top)
        if zhat_T is not None:
            img_TD = np.tile(zhat_T, (z_img_height, 1))
            ax_handles[1].imshow(
                img_TD,
                interpolation='nearest',
                vmin=-0.5, vmax=(K-1)+0.5,
                cmap=z_img_cmap)
            ax_handles[1].set_ylim(0, z_img_height)
            ax_handles[1].set_yticks([])
            bbox = ax_handles[1].get_position()
            width = (1.0 - bbox.x1) / 3
            height = bbox.y1 - bbox.y0
            cax = fig_h.add_axes([right + 0.01, bottom, width, height])
            cbax_h = fig_h.colorbar(
                ax_handles[1].images[0], cax=cax, orientation='vertical')
            cbax_h.set_ticks(np.arange(K))
            cbax_h.set_ticklabels(np.arange(K))
            cbax_h.ax.tick_params(labelsize=9)
        ax_handles[-1].set_xlabel('time')
        pylab.show()
        return ax_handles

    
