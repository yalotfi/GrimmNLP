import os
import gensim
import logging

from nltk import sent_tokenize
from nltk import word_tokenize


def main():
    # utilies
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
    )
    directory = 'embeddings'
    fcorpus = 'grimm_corpus.txt'

    # preprocessing
    corpus = open(fcorpus).read().lower()
    sentences = [word_tokenize(sent) for sent in sent_tokenize(corpus)]

    # train word2vec
    windows = [3, 5, 7, 9, 11]
    for w in windows:
        fname = 'vecs_300_window' + str(w)
        if not os.path.exists(directory):
            os.mkdir(directory)
        model_path = os.path.join(directory, fname)
        model = gensim.models.Word2Vec(
            sentences, size=300, window=w, min_count=5, workers=4)
        model.save(model_path)


if __name__ == '__main__':
    main()
