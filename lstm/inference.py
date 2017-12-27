import os
import pickle
import spacy
import numpy as np


from keras.models import load_model


TEST_TEXT = "One day she was captured"


def main():
    # LOAD MODEL #
    outdir = os.path.join(os.getcwd(), 'lstm', 'output')
    outmodel = os.path.join(outdir, 'lstm', 'model.h5')
    model = load_model(outmodel)
    print(model.summary())

    # LOAD TOKENIZER #
    corpus = TEST_TEXT.lower()
    infile = os.path.join(outdir, 'tokens.p')
    with open(infile, 'rb') as f:
        pp = pickle.load(f)

    # TOKENIZE #
    nlp = spacy.load('en')
    doc = nlp(corpus)
    test_tokens = [token.text for token in doc]
    for i, token in enumerate(test_tokens):
        if token not in pp["token2idx"].keys():
            test_tokens[i] = '-UNK-'
        else:
            continue

    input_tokens = test_tokens[-pp["max_len"]:]  # Input Seed
    eos = ['.', '?', '!']  # end of sentence tokens
    pred_word = None
    ending = input_tokens
    while pred_word not in eos or len(ending) < 10:
        # VECTORIZE
        input_idx = [pp["token2idx"][token] for token in input_tokens]
        input_seq = np.zeros((pp["max_len"], len(pp["token2idx"])))
        input_seq[np.arange(pp["max_len"]), input_idx] = 1.

        # INFERENCE #
        X_test = input_seq.reshape((1, pp["max_len"], len(pp["token2idx"])))
        preds = model.predict(X_test)
        pred_idx = np.argmax(preds)
        pred_word = pp["idx2token"][pred_idx]
        ending.append(pred_word)
        input_tokens = ending[-pp["max_len"]:]
        print(' '.join(ending))


if __name__ == '__main__':
    main()
