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

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict,
                 this_word: str, n_constant=3, min_n_components=2,
                 max_n_components=10, random_state=14, verbose=False):
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

    def base_model(self, num_states, X, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except Exception as e:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word,
                                                            num_states))
            return None

    def score_model(self, num_components):
        raise NotImplementedError

    def select_best(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        components = range(self.min_n_components, self.max_n_components + 1)

        scores = [(c, self.score_model(c)) for c in components]
        scores = [s for s in scores if s[-1] is not None]
        scores.sort(key=lambda v: v[1], reverse=True)

        if scores:
            best_num_components = scores[0][0]
        else:
            best_num_components = self.n_constant
        return self.base_model(best_num_components, self.X, self.lengths)


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components, self.X, self.lengths)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def score_model(self, num_components):
        try:
            model = self.base_model(num_components, self.X, self.lengths)
            if model is not None:
                log_L = model.score(self.X, self.lengths)
                N, p = self.X.shape

                return -2 * log_L + p * np.log(N)
        except Exception as e:
            return None

    def select(self):
        return self.select_best()


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application
    to hmm topology optimization." Document Analysis and Recognition, 2003.
    Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def score_model(self, num_components):
        try:
            model = self.base_model(num_components, self.X, self.lengths)
            if model is not None:
                log_L = model.score(self.X, self.lengths)

                log_P = [model.score(X, lengths) for w, (X, lengths)
                         in self.hwords.items() if w != self.this_word]
                M = len(log_P)
                return log_L - sum(log_P) / (M - 1)
        except Exception as e:
            return None

    def select(self):
        return self.select_best()


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of
    cross-validation folds

    '''

    def score_model(self, num_components):
        n_splits = 3
        if len(self.sequences) < n_splits:
            return None

        kf = KFold(n_splits=n_splits)

        scores = []
        for train_index, test_index in kf.split(self.sequences):
            X_train, lengths_train = combine_sequences(train_index,
                                                       self.sequences)
            model = self.base_model(num_components, X_train, lengths_train)
            if model is None:
                continue

            X_test, lengths_test = combine_sequences(test_index,
                                                     self.sequences)

            try:
                scores.append(model.score(X_test, lengths_test))
            except Exception as e:
                continue
                # print('Error: {}'.format(e))

        return np.mean(scores)

    def select(self):
        return self.select_best()
