import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    Xlengths = test_set.get_all_Xlengths()
    
    for word_id in Xlengths:
        X, lengths = Xlengths[word_id]
        word_probs = {}
        for word in models:
            try: 
                score = models[word].score(X, lengths)
            except:
                score = float("-Inf")
            word_probs[word] = score

        probabilities.append(word_probs)
        best_guess = max (word_probs, key=word_probs.get)
        guesses.append(best_guess)
    
    return (probabilities, guesses)