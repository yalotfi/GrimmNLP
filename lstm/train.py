import os
import spacy
import pickle
import numpy as np

from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop


# Model Hyperparameters
HYPERPARAMETERS = {
    "MAX_SEQUENCE_LENGTH": 5,
    "STEP_SIZE": 1,
    "BATCH_SIZE": 512,
    "EPOCHS": 10
}

# Dev Text
TRAIN_TEXT = ("Now one day it happened that the princess\'s golden ball did not "
              "fall into her hands, that she held up high, but instead it fell "
              "to the ground and rolled right into the water. The princess "
              "followed it with her eyes, but the ball disappeared, and the "
              "well was so deep that she could not see its bottom. Then she began "
              "to cry. She cried louder and louder, and she could not console "
              "herself.")
TEST_TEXT = ("The fairy was called Fiona. "
             "Fiona was very happy. "
             "One day, she was captured.")


class Preprocessor(object):
    """docstring for Preprocessor"""

    def __init__(self, corpus, hyperparameters):
        super(Preprocessor, self).__init__()
        self.corpus = corpus
        self.max_len = hyperparameters["MAX_SEQUENCE_LENGTH"]
        self.step = hyperparameters["STEP_SIZE"]
        self.tokens = self.__get_tokens
        self.vocab = set(self.tokens)
        self.token2idx, self.idx2token = self.index_tokens()
        self.vocab_size = len(self.token2idx)
        self.eos = ['.', '?', '!']

    @property
    def __get_tokens(self):
        nlp = spacy.load('en')
        doc = nlp(self.corpus)
        return [token.text for token in doc]

    def index_tokens(self):
        # Create dictionaries
        token2idx = dict((t, i) for i, t in enumerate(self.vocab))
        idx2token = dict((i, t) for i, t in enumerate(self.vocab))

        # Add unknown token
        n = len(token2idx.keys())
        token2idx['-UNK-'] = n
        idx2token[n] = '-UNK-'
        return token2idx, idx2token

    def vectorize_tokens(self):
        currtxt, nexttxt = [], []
        for i in range(0, len(self.tokens) - self.max_len, self.step):
            currtxt.append(self.tokens[i: i + self.max_len])
            nexttxt.append(self.tokens[i + self.max_len])
        return currtxt, nexttxt

    def build_train_set(self, currtxt, nexttxt, token2idx):
        y_labels = len(token2idx)
        X = np.zeros((len(currtxt), self.max_len, y_labels), dtype=np.bool)
        y = np.zeros((len(currtxt), y_labels), dtype=np.bool)
        for i, currtxt in enumerate(currtxt):
            for t, word in enumerate(currtxt):
                X[i, t, token2idx[word]] = 1
            y[i, token2idx[nexttxt[i]]] = 1
        return X, y

    def preprocess(self):
        # token2idx, idx2token = self.index_tokens()
        currtxt, nexttxt = self.vectorize_tokens()
        X, y = self.build_train_set(currtxt, nexttxt, self.token2idx)
        return {'X': X, 'y': y}

    def dump_train_set(self, train_set, outfile):
        with open(outfile, 'wb') as f:
            np.savez(f, X=train_set["X"], y=train_set["y"])


def compile_model(X):
    m_examples = X.shape[0]
    max_len = X.shape[1]
    y_labels = X.shape[2]

    model = Sequential()
    model.add(LSTM(256, input_shape=(max_len, y_labels)))
    model.add(Dense(y_labels))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
    return model


def main():
    # FILE I/O #
    infile = os.path.join(os.getcwd(), 'lstm', 'data', 'grimm_corpus.txt')
    corpus = open(infile, encoding='latin-1').read().lower()
    # corpus = TRAIN_TEXT

    # PREPROCESS #
    pp = Preprocessor(corpus, HYPERPARAMETERS)
    train_set = pp.preprocess()
    print(
        "Max Sequence Length: {0}\n"
        "Timestep Size: {1}\n"
        "Vocab Size: {2}\n"
        "Input Tensor Shape: {3}\n".format(
            pp.max_len, pp.step, pp.vocab_size, train_set['X'].shape))

    # BUILD #
    model = compile_model(train_set['X'])
    print(model.summary())

    # TRAIN #
    batch_size = HYPERPARAMETERS['BATCH_SIZE']
    epochs = HYPERPARAMETERS['EPOCHS']
    print("Training RNN for {} epochs...".format(epochs))
    model.fit(train_set['X'], train_set['y'],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)

    # SAVE #
    outdir = os.path.join(os.getcwd(), 'lstm', 'output')
    outtoken = os.path.join(outdir, 'tokens.p')
    outmodel = os.path.join(outdir, 'model.h5')
    model.save(outmodel)
    token_dict = {
        "tokens": pp.tokens,
        "token2idx": pp.token2idx,
        "idx2token": pp.idx2token,
        "vocab": pp.vocab,
        "max_len": pp.max_len
    }
    with open(outtoken, 'wb') as f:
        pickle.dump(token_dict, f)

    # INFERENCE #
    cached_model = load_model(outmodel)
    print(cached_model.summary())

    # TOKENIZE #
    nlp = spacy.load('en')
    doc = nlp(TEST_TEXT)
    test_tokens = [token.text for token in doc]
    for i, token in enumerate(test_tokens):
        if token not in pp.token2idx.keys():
            test_tokens[i] = '-UNK-'
        else:
            continue

    # VECTORIZE #
    input_tokens = test_tokens[-5:]
    input_idx = [pp.token2idx[token] for token in input_tokens]
    input_seq = np.zeros((pp.max_len, pp.vocab_size))
    input_seq[np.arange(5), input_idx] = 1.

    # X_test = train_set['X'][0].reshape((1, pp.max_len, pp.vocab_size))
    X_test = input_seq.reshape((1, pp.max_len, pp.vocab_size))
    print(X_test.shape)

    preds = cached_model.predict(X_test)[0]
    print(preds.shape)
    print(preds)

    pred_score = np.max(preds)
    pred_idx = np.argmax(preds)
    pred_word = pp.idx2token[pred_idx]
    print(pred_score)
    print(pred_idx)
    print(pred_word)

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    sampled_word = pp.idx2token[sample(preds)]
    print(sampled_word)


if __name__ == '__main__':
    main()
