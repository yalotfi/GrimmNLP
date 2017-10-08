import os
import spacy


def main():
    # READ #
    infile = os.path.join('data', 'roc_grimm_corpus.txt')
    text = open(infile).read().lower()

    # PROCESS #
    nlp = spacy.load('en')
    doc = nlp(text)
    print("There are {} unqiue tokens".format(len(doc)))
    lemmas = ' '.join([sent.lemma_ for sent in doc])

    # WRITE #
    outfile = os.path.join('data', 'roc_grimm_lemmas.txt')
    open(outfile, 'w').write(lemmas)
    print("Done writing lemmatized corpus")


if __name__ == '__main__':
    main()
