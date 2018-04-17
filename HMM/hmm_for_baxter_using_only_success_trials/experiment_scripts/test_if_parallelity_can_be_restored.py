import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import hmmlearn.hmm
import HMM.hmm_wrapper.bnpy_hmm as bnpy_hmm
import bnpy
import util
import ipdb
import os

# don't wanna see scientific notation
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

def tamper_input_mat(X, all_Xs):
    list_of_tampered_range = []
    length = X.shape[0]
    list_of_tampered_range.append([int(length*0.1), int(length*0.2)])
    list_of_tampered_range.append([int(length*0.5), int(length*0.6)])

    std_mat = np.array(all_Xs).std(0)

    for r in list_of_tampered_range:
        X[r[0]:r[1]+1] -= 2*std_mat[r[0]:r[1]+1]

    return X, list_of_tampered_range

def profile_model(model, output_dir, output_prefix):
    output_prefix = 'model_profile_'+output_prefix
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        np.savetxt(
            os.path.join(output_dir, output_prefix+'_transmat.txt'),
            model.transmat_,
            fmt='%.6f')
        np.savetxt(
            os.path.join(output_dir, output_prefix+'_startprob.txt'),
            model.startprob_,
            fmt='%.6f')
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))

def color_txt_lines(txt_file_path, list_of_color_range):
    txt_file = open(txt_file_path, 'r')
    lines = txt_file.readlines()
    for r in list_of_color_range:
        for i in range(r[0], r[1]+1):
            lines[i] = "\033[1;34m%s\033[0m"%(lines[i],)

    tmp_file_path = os.path.join(os.path.dirname(txt_file_path), 'tmp.txt')
    tmp_file = open(tmp_file_path, 'w')
    for l in lines:
        tmp_file.write(l)
    tmp_file.close()
    txt_file.close()

    os.rename(tmp_file_path, txt_file_path)

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

def tab_sep_floats(list_to_print, annotation_list=None, highlight_maximum=False, highlight_attr='1;39', idx_to_hl=set()):
    s = ''
    max_idx = None
    if highlight_maximum:
        import operator
        max_idx, max_val = max(enumerate(list_to_print), key=operator.itemgetter(1))

    for idx in range(len(list_to_print)):
        if idx == max_idx or idx in idx_to_hl:
            s += '\033['+highlight_attr+'m'

        i = list_to_print[idx]
        if annotation_list is not None:
            j = annotation_list[idx]
            s += '%-24s' % ('%.2f(%s)'%(i, j))
        else:
            s += '%-24s' % ('%.2f'%i)

        if idx == max_idx or idx in idx_to_hl:
            s += '\033[0m'
    return s

def profile_log_curve_cal(X, model, output_dir, output_prefix, list_of_color_range=[]):
    output_prefix = 'log_curve_cal_profile_'+output_prefix
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from scipy.special import logsumexp
        from sklearn.utils import check_array, check_random_state

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])

        np.savetxt(
            os.path.join(output_dir, output_prefix+'_framelogprob.txt'),
            framelogprob,
            fmt='%.6f')
        color_txt_lines(
            os.path.join(output_dir, output_prefix+'_framelogprob.txt'),
            list_of_color_range)

        logprobij, _fwdlattice = model._do_forward_pass(framelogprob)
        np.savetxt(
            os.path.join(output_dir, output_prefix+'_fwdlattice.txt'),
            _fwdlattice,
            fmt='%.6f')
        color_txt_lines(
            os.path.join(output_dir, output_prefix+'_fwdlattice.txt'),
            list_of_color_range)

        n_samples, n_components = framelogprob.shape
        log_startprob = log_mask_zero(model.startprob_)
        log_transmat = log_mask_zero(model.transmat_)

        log_lik = [None]*n_samples
        log_lik[0] = logsumexp(_fwdlattice[0])


        color_t = set()
        for r in list_of_color_range:
            for t in range(r[0], r[1]+1):
                color_t.add(t)

        import sys

        orig_stdout = sys.stdout
        f = open(os.path.join(output_dir, output_prefix+'_graph_cal_check.txt'), 'w')
        sys.stdout = f

        print 0, '\t', 'lpi\t', tab_sep_floats(log_startprob)
        print 0, '\t', '\t', '+\t\t\t'*n_components
        print 0, '\t', 'lem\t', tab_sep_floats(framelogprob[0], highlight_maximum=True)
        print 0, '\t', '\t', '|\t\t\t'*n_components
        print 0, '\t', 'lf\t', tab_sep_floats(_fwdlattice[0], highlight_maximum=True, highlight_attr='1;39;4'), '->lse deri', 0


        color_list = range(31,37)
        from itertools import cycle
        colors = cycle(color_list)


        for t in range(1, n_samples):
            if t in color_t:
                fmt = "\033[1;"+str(next(colors))+"m%s\033[0m"
            else:
                fmt = "\033[0;39m%s\033[0m"
            print fmt%(t,), '\t', '\t', 'A*\t\t\t'*n_components


            intermediate_mat = log_transmat.copy()
            intermediate_mat += _fwdlattice[t-1].reshape(-1, 1)

            max_row_no = intermediate_mat.argmax(0)
            for i in range(n_components):
                idx_to_hl = set()
                for col_no in np.where(max_row_no == i)[0]:
                    idx_to_hl.add(col_no)
                annotation_list = []
                for j in range(n_components):
                    annotation_list.append('%.2f%+.2f'%(_fwdlattice[t-1][i], log_transmat[i][j]))
                print fmt%(t,), '\t', 'inmat\t', tab_sep_floats(intermediate_mat[i], annotation_list=annotation_list, idx_to_hl=idx_to_hl)
            print fmt%(t,), '\t', '\t', '|lse\t\t\t'*n_components
            print fmt%(t,), '\t', '\t', 'v\t\t\t'*n_components

            lAf = _fwdlattice[t]-framelogprob[t]
            log_lik[t] = logsumexp(_fwdlattice[t])

            annotation_list = []
            print fmt%(t,), '\t', 'lAf\t', tab_sep_floats(lAf, highlight_maximum=True)
            print fmt%(t,), '\t', '\t', '+\t\t\t'*n_components
            print fmt%(t,), '\t', 'lem\t', tab_sep_floats(framelogprob[t], highlight_maximum=True)
            print fmt%(t,), '\t', '\t', '|\t\t\t'*n_components
            print fmt%(t,), '\t', '\t', 'v\t\t\t'*n_components


            for i in range(n_components):
                annotation_list.append('%+.2f'%(_fwdlattice[t][i]-_fwdlattice[t-1][i], ))
            print fmt%(t,), '\t', 'lf\t', tab_sep_floats(_fwdlattice[t], annotation_list=annotation_list, highlight_maximum=True, highlight_attr='1;39;4'), '->lse deri', round(log_lik[t]-log_lik[t-1],2)


        sys.stdout = orig_stdout
        f.close()


    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))


def tamper_transmat(model):
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        hidden_state_amount = len(model.transmat_)
        if hidden_state_amount == 1:
            pass
        else:
            model.transmat_[:] = 0.9999999999/(hidden_state_amount-1)
            for i in range(hidden_state_amount):
                model.transmat_[i, i] = 0.0000000001
        pass
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))

def tamper_startprob(model):
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        hidden_state_amount = len(model.transmat_)
        model.startprob_[:] = 1.0/hidden_state_amount
        pass
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))

def delete_range_and_get_segments(complete_curve, list_of_range_to_delete):
    list_of_range_to_delete = sorted(list_of_range_to_delete, key=lambda x:x[0])
    list_of_xy = []
    start_idx = 0
    for r in list_of_range_to_delete:
        end_idx = r[0]
        if start_idx < end_idx:
            list_of_xy.append({
                'x': range(start_idx, end_idx),
                'y': complete_curve[start_idx:end_idx],
            })
        start_idx = r[1]+1

    curve_len = len(complete_curve)
    if start_idx < curve_len:
        list_of_xy.append({
            'x': range(start_idx, curve_len),
            'y': complete_curve[start_idx:curve_len],
        })

    return list_of_xy

def run(model_save_path,
    trials_group_by_folder_name,
    parsed_options):

    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        try:
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        except IOError:
            print 'model of state %s not found'%(state_no,)
            continue

    base_dir = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(base_dir, 'experiment_output', 'test_if_parallelity_can_be_restored')
    output_id = '(tamper_input)'


    tampered = False
    if parsed_options.tamper_transmat:
        output_id += '_(tamper_transmat)'
        tampered = True
    if parsed_options.tamper_startprob:
        output_id += '_(tamper_startprob)'
        tampered = True
    output_dir = os.path.join(exp_dir, output_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    for state_no in model_group_by_state:
        X = one_trial_data_group_by_state[state_no]
        all_Xs = [trials_group_by_folder_name[trial_name][state_no]\
                for trial_name in trials_group_by_folder_name]
        tampered_X, list_of_tampered_range = tamper_input_mat(X.copy(), all_Xs)

        model = model_group_by_state[state_no]
        profile_model(model, output_dir, 'state %s raw'%(state_no,))


        if parsed_options.tamper_transmat:
            tamper_transmat(model)
        if parsed_options.tamper_startprob:
            tamper_startprob(model)
        if tampered:
            profile_model(model, output_dir, 'state %s tampered'%(state_no,))

        log_transmat = util.get_log_transmat(model)


        log_lik_of_X = np.array(util.fast_log_curve_calculation(X, model))
        framelogprob_of_X = np.array(util.get_emission_log_prob_matrix(X, model))
        fwdlattice_of_X = util.get_hidden_state_log_prob_matrix(X, model)
        max_hstate_of_X = fwdlattice_of_X.argmax(1)

        the_term_of_X = [framelogprob_of_X[0][max_hstate_of_X[0]]]
        for t in range(1, len(max_hstate_of_X)):
            hs1 = max_hstate_of_X[t-1]
            hs2 = max_hstate_of_X[t]
            the_term_of_X.append(framelogprob_of_X[t][hs2]+log_transmat[hs1][hs2])

        profile_log_curve_cal(X, model, output_dir, 'state %s X'%(state_no,), list_of_tampered_range)



        log_lik_of_tampered_X = np.array(util.fast_log_curve_calculation(tampered_X, model))
        framelogprob_of_tampered_X = np.array(util.get_emission_log_prob_matrix(tampered_X, model))
        fwdlattice_of_tampered_X = util.get_hidden_state_log_prob_matrix(tampered_X, model)
        max_hstate_of_tampered_X = fwdlattice_of_tampered_X.argmax(1)

        the_term_of_tampered_X = [framelogprob_of_tampered_X[0][max_hstate_of_tampered_X[0]]]
        for t in range(1, len(max_hstate_of_tampered_X)):
            hs1 = max_hstate_of_tampered_X[t-1]
            hs2 = max_hstate_of_tampered_X[t]
            the_term_of_tampered_X.append(framelogprob_of_tampered_X[t][hs2]+log_transmat[hs1][hs2])

        profile_log_curve_cal(tampered_X, model, output_dir, 'state %s tampered_X'%(state_no,), list_of_tampered_range)





        deri_of_X = log_lik_of_X.copy()
        deri_of_X[1:] = log_lik_of_X[1:]-log_lik_of_X[:-1]
        deri_of_X[0] = 0

        deri_of_tampered_X = log_lik_of_tampered_X.copy()
        deri_of_tampered_X[1:] = log_lik_of_tampered_X[1:]-log_lik_of_tampered_X[:-1]
        deri_of_tampered_X[0] = 0

        diff = log_lik_of_X-log_lik_of_tampered_X



        fig = plt.figure()
        bbox_extra_artists = []

        ax = fig.add_subplot(411)
        title = "log lik"
        ax.set_title(title)
        ax.plot(log_lik_of_X, color='black', marker='None', linestyle='solid', label='Normal')
        ax.plot(log_lik_of_tampered_X, color='blue', marker='None', linestyle='solid', label='Tampered')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)


        ax = fig.add_subplot(412)
        title = "1st deri"
        ax.set_title(title)
        ax.plot(deri_of_X, color='black', marker='None', linestyle='solid', label='Normal')
        ax.plot(deri_of_tampered_X, color='blue', marker='None', linestyle='solid', label='Tampered')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)


        ax = fig.add_subplot(413)
        title = "1st deri and max emission prob of Normal"
        ax.set_title(title)
        ax.plot(deri_of_X, color='black', marker='None', linestyle='solid', label='Normal 1st deri')
        ax.plot(the_term_of_X, color='red', marker='None', linestyle='solid', label='Normal the term')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)


        ax = fig.add_subplot(414)
        title = "1st deri and max emission prob of Tampered"
        ax.set_title(title)
        ax.plot(deri_of_tampered_X, color='blue', marker='None', linestyle='solid', label='Tampered 1st deri')
        ax.plot(the_term_of_tampered_X, color='red', marker='None', linestyle='solid', label='Tampered the term')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)


        title = "output_id %s state %s"%(output_id, state_no)
        fig.suptitle(title)

        plt.tight_layout()

        fig.savefig(os.path.join(output_dir, title+".eps"), format="eps", bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, title+".png"), format="png", bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
