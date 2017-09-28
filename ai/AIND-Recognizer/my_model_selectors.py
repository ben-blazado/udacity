import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float("Inf")
        best_num_states = 0
        curr_num_states = self.min_n_components

        #--- logN is used in BIC score
        logN = np.log(len(self.X))
        BIC = []
        model = []

        try:

            while curr_num_states <= self.max_n_components:
                
                curr_model = GaussianHMM(
                    n_components    = curr_num_states, 
                    covariance_type = "diag", 
                    n_iter          = 1000,
                    random_state    = self.random_state, 
                    verbose         = False)

                curr_model.fit(self.X, self.lengths)
                logL = curr_model.score(self.X, self.lengths)

                #--- curr score is BIC score: -2 * logL + p * logN
                #--- number of parameters(p) = size of transition matrix + size of means + size of covariance matrix
                p = curr_model.transmat_.size + curr_model.means_.size + curr_model.covars_.size
                BIC.append(-2.0 * logL + p * logN)
                model.append(curr_model)

                curr_num_states += 1
        
        except:
            pass
        
        if (BIC):
            best_model=model[np.argmin(BIC)]
        
        return best_model

    
class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        warnings.filterwarnings("ignore", category=RuntimeWarning)        
        best_model = None
        curr_num_states = self.min_n_components
        
        try:
            logL = []
            model = []

            while curr_num_states <= self.max_n_components:
                
                curr_model = GaussianHMM(
                    n_components    = curr_num_states, 
                    covariance_type = "diag", 
                    n_iter          = 5000,
                    random_state    = self.random_state, 
                    verbose         = False)

                curr_model.fit(self.X, self.lengths)
                logL.append(curr_model.score(self.X, self.lengths))
                model.append(curr_model)
                curr_num_states += 1
                
        except:
            pass
        
        model_idx = range(len(model))
        
        if (model_idx):
            avg_other_logL = [np.mean(logL[:i] + logL[i + 1:]) for i in model_idx]
            DIC            = [logL[i] - avg_other_logL[i] for i in model_idx]
            best_model     = model[np.argmax(DIC)]
        
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_model = None
        curr_nstate = self.min_n_components
        
        score = []
        model = []

        try:
            X_train, lengths_train, X_test, lengths_test = create_sets(self)

            while curr_nstate <= self.max_n_components:

                curr_model = GaussianHMM(
                    n_components    = curr_nstate, 
                    covariance_type = "diag", 
                    n_iter          = 1000,
                    random_state    = self.random_state, 
                    verbose         = False)

                curr_model.fit(X_train, lengths_train)
                
                score.append(curr_model.score(X_test, lengths_test))
                model.append(curr_model)

                curr_nstate += 1
        
        except:
            pass
        
        if (score):
            best_model = model[np.argmax(score)]
        
        return best_model
    
    
#--- helper function
def create_sets (model_selector):

    split_method = KFold(n_splits=min(len(model_selector.sequences),3))

    train_idx_combined = []
    test_idx_combined = []

    for train_idx, test_idx in split_method.split(model_selector.sequences):
        train_idx_combined += train_idx.tolist()
        test_idx_combined += test_idx.tolist()

    X_train, lengths_train = combine_sequences (train_idx_combined, model_selector.sequences)
    X_test, lengths_test = combine_sequences (test_idx_combined, model_selector.sequences)
    
    return X_train, lengths_train, X_test, lengths_test

    
