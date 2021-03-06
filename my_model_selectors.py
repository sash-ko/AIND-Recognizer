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

    def select_best(self, maximaze=True):
        """ Helper functions that loops between min and max number of
        components and to find out how many states leads to the best solution
        and than train a model with this number of components
        """

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # check all possible numbers components
        components = range(self.min_n_components, self.max_n_components + 1)

        scores = [(c, self.score_model(c)) for c in components]
        scores = [s for s in scores if s[-1] is not None]

        # minimize or maximaze the score
        scores.sort(key=lambda v: v[1][0], reverse=maximaze)

        model = None
        if scores:
            best_num_components = scores[0][0]
            model = scores[0][1][1]
        else:
            # fallback to the constant if nothing scored
            best_num_components = self.n_constant

        # Some selectors, like SelectorCV, don't return a model so it needs
        # to be trained here. Those selectora just perform looping and
        # selection of best number of components
        if model is None:
            model = self.base_model(best_num_components, self.X, self.lengths)
        return model


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
    """ select the model with the lowest Baysian Information Criterion
    (BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def score_model(self, num_components):
        try:
            model = self.base_model(num_components, self.X, self.lengths)
            if model is not None:
                log_L = model.score(self.X, self.lengths)
                N, features = self.X.shape

                # https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt
                # p = n^2 + 2*d*n - 1
                # d - number of features
                # n - number of HMM states
                p = num_components ** 2 + 2 * features * num_components - 1

                BIC = -2 * log_L + p * np.log(N)
                return BIC, model
        except Exception as e:
            return None

    def select(self):
        return self.select_best(maximaze=False)


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

                scores = []
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        try:
                            score = model.score(X, lengths)
                        except Exception:
                            continue

                        scores.append(score)

                M = len(scores)
                DIC = log_L - sum(scores) / (M - 1)
                return DIC, model
        except Exception as e:
            return None

    def select(self):
        # loop between min and max number of components and
        # find out how many states leads to the best solution

        # maximize the score
        return self.select_best(maximaze=True)


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
            try:
                X_train, lengths_train = combine_sequences(train_index,
                                                           self.sequences)
                model = self.base_model(num_components, X_train, lengths_train)
                if model is None:
                    continue

                X_test, lengths_test = combine_sequences(test_index,
                                                         self.sequences)

                scores.append(model.score(X_test, lengths_test))
            except Exception as e:
                continue

        return np.mean(scores), None

    def select(self):
        # loop between min and max number of components and
        # find out how many states leads to the best solution
        return self.select_best(maximaze=False)
