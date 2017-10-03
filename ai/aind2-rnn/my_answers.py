import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    series_size = len(series)
    X = [series[start:start + window_size] 
         for start in range(series_size - window_size)]
    y = [series[start] for start in range(window_size, series_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    from string import ascii_lowercase

    text = text.lower()

    punctuation = ['!', ',', '.', ':', ';', '?']
    valid_chars = list(ascii_lowercase) + punctuation
    for c in text: 
        if c not in valid_chars:
            text = text.replace(c, " ")
    return text
    

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    text_size = len(text)
    inputs = [text[start:start+window_size]
             for start in range(0, text_size - window_size, step_size)]
    outputs = [text[start]
               for start in range(window_size, text_size, step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    from keras.layers import Activation
    
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    return model


