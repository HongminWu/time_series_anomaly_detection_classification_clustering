model_store = {
    'hmmlearn\'s HMM': {
        'use': 'default',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 10000,
                'hmm_hidden_state_amount': 5,
                'gaussianhmm_covariance_type_string': 'diag',
            },
            'c1': {
                'hmm_max_train_iteration': 100000,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            'c1_less_iter': {
                'hmm_max_train_iteration': 1000,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            'c1_less_iter_less_maxhstate': {
                'hmm_max_train_iteration': 1000,
                'hmm_max_hidden_state_amount': 10,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            'a1': {
                'hmm_max_train_iteration': [100, 1000],
                'hmm_hidden_state_amount': [1,2,3,4,5,6,7],
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            '201709051536': {
                'hmm_max_train_iteration': 100000,
                'hmm_hidden_state_amount': 5,
                'gaussianhmm_covariance_type_string': 'full',
            },
            'config_that_make_state_1_diverge_for_20170711data': {
                'hmm_max_train_iteration': 100000,
                'hmm_hidden_state_amount': 2,
                'gaussianhmm_covariance_type_string': 'diag',
            },
        }
    },

    'hmmlearn\'s GMMHMM': {
        'use': 'default',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 10000,
                'hmm_hidden_state_amount': 5,
                'gaussianhmm_covariance_type_string': 'diag',
                'GMM_state_amount': 10,
            },
            'haha': {
                'hmm_max_train_iteration': 100000,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'full',
                'GMM_state_amount': 10,
            },
            'c1': {
                'hmm_max_train_iteration': 100000,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
                'GMM_state_amount': [1,2,3,4,5,6,7,8,9,10],
            },
            'd1': {
                'hmm_max_train_iteration': 100,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
                'GMM_max_state_amount': 100,
            },
        }
    },
  

    'BNPY\'s HMM': {
        'use': 'gauss',
        'config_set': {
            'ar': {
                'hmm_max_train_iteration': 1000,
                'hmm_hidden_state_amount': 5,
                'alloModel' : 'HDPHMM',     
                'obsModel'  : 'AutoRegGauss',
                'ECovMat'   : ['covdata'],
               # 'ECovMat'   : ['covdata', 'diagcovdata','covfirstdiff', 'diagcovfirstdiff'],
                'varMethod' : 'memoVB',
            },
            
            'gauss': {
                'hmm_max_train_iteration': 1000,
                'hmm_hidden_state_amount': 5,
                'alloModel' : 'HDPHMM',     
                'obsModel'  : ['DiagGauss'],  
                'ECovMat'   : ['covdata'],
                'varMethod' : 'memoVB',
            },


            'auto': {
                'hmm_max_train_iteration': 10000,
                'hmm_max_hidden_state_amount': 20,
                'alloModel' : 'HDPHMM',     
                'obsModel'  : ['Gauss', 'DiagGauss', 'ZeroMeanGauss'],
                'ECovMat'   : ['eye', 'covdata', 'diagcovdata'],
                'varMethod' : ['moVB', 'memoVB', 'VB'],
            },
        }
    },
    
    'PYHSMM\'s HMM': {
        'use': 'default',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 1000,
                'hmm_hidden_state_amount': 10,
                'max_duration_length' : 10,     
            },
        }
    },
}
