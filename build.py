from collections import Counter
from collections import namedtuple

from pprint import pprint

from gensim import corpora
from nltk.corpus import stopwords


# List of 127 stop words
STOPS = frozenset(stopwords.words('english'))

TRAIN_ROOT = 'data/train/'

Author = namedtuple('Author', ['name', 'id'])
Text = namedtuple('Text', ['file', 'author'])


def is_stop(word):
    """Is `word` a stopword?"""
    return word.lower() in STOPS


def is_stop_test():
    assert is_stop('the')
    assert is_stop('tHe')
    assert is_stop('aN')
    assert is_stop('A')


def tokens_from_file(fname):
    with open(fname) as f:
        for line in f:
            words = unstop(tokenize(line))
            for word in words:
                yield word


def remove_stops(docs):
    return [[word for word in doc if not is_stop(word)] for doc in docs]


def remove_rare(docs):
    counts = Counter(word for doc in docs for word in doc)
    return [[word for word in doc if counts[word] > 1] for doc in docs]


def tokenize(string):
    """Turn a string into a list of (lowercase) words."""
    return string.lower().split()


def tokenize_test():
    assert tokenize('') == []
    assert tokenize('How are you?') == ['how', 'are', 'you?']
    assert tokenize('how') == ['how']
    assert tokenize('How') == ['how']


def test():
    is_stop_test()
    tokenize_test()


def unstop(lst):
    return [word for word in lst if not is_stop(word)]


if __name__ == '__main__':
    test()

    tolst = Author('Tolstoy', 0)
    dosto = Author('Dostoevsky', 1)
    dickens = Author('Dickens', 2)
    flaubert = Author('Flaubert', 3)
    austen = Author('Austen', 4)

    texts = [
        Text('anna_karenina.txt', tolst),
        Text('brothers_karamazov.txt', dosto),
        Text('little_dorrit.txt', dickens),
        Text('madame_bovary.txt', flaubert),
        Text('pride_and_prejudice.txt', austen),
    ]

    docs = [list(tokens_from_file(TRAIN_ROOT + text.file))
            for text in texts]
    pprint(docs[1][:10])

    docs = remove_rare(docs)
    pprint(docs[1][:10])

    dictionary = corpora.Dictionary(docs)
    dictionary.save('state/big5.dict')

    corpus = [dictionary.doc2bow(text) for text in docs]
    corpora.MmCorpus.serialize('state/big5.mm', corpus)
    print(corpus[1][:10])

    new_doc = 'She arrived a couple of days later'
    new_vec = dictionary.doc2bow(tokenize(new_doc))
    print(new_vec)
