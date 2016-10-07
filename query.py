from pprint import pprint

from gensim import corpora, models, similarities

from build import tokenize


if __name__ == '__main__':
    dictionary = corpora.Dictionary.load('state/big5.dict')
    corpus = corpora.MmCorpus('state/big5.mm')
    print(corpus)

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

    doc = 'She arrived a couple of days later'
    vec_bow = dictionary.doc2bow(tokenize(doc))
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    print(vec_lsi)

    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save('state/big5.index')
    index = similarities.MatrixSimilarity.load('state/big5.index')

    sims = index[vec_lsi]
    pprint(list(enumerate(sims)))

