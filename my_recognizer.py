import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

    :param models: dict of trained models
        {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD':
         GaussianHMM model object, ...}
    :param test_set: SinglesData object
    :return: (list, list) as probabilities, guesses both lists are ordered
        by the test set word_id probabilities is a list of dictionaries
        where each key a word and
        value is Log Liklihood
            [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
             {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... }]
        guesses is a list of the best guess words ordered by the test
        set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []
    
    for test_word in sorted(test_set.get_all_sequences().keys()):
        X, lengths = test_set.get_item_Xlengths(test_word)

        probs = {}

        best_score = float('-inf')
        best_guess = None
        
        for train_word, model in models.items():
            try:
                log_L = model.score(X, lengths)
            except Exception as e:
                # print('Error: {}'.format(e))
                log_L = float('-inf')

            probs[train_word] = log_L
            if log_L > best_score:
                best_score = log_L
                best_guess = train_word
        
        guesses.append(best_guess)
        probabilities.append(probs)

    return probabilities, guesses
