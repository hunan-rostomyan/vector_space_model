from gensim import corpora, models, similarities

from build import tokenize


if __name__ == '__main__':
    dictionary = corpora.Dictionary.load('state/big5.dict')
    corpus = corpora.MmCorpus('state/big5.mm')
    print(corpus)

    tfidf = models.TfidfModel(corpus)
    print(tfidf)

    # Transform individual vector
    new_doc = 'She arrived a couple of days later'
    new_vec = dictionary.doc2bow(tokenize(new_doc))
    print(tfidf[new_vec])

    # Transform the whole corpus
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    corpus_lsi = lsi[corpus_tfidf]
    lsi.print_topics(2)

    for doc in corpus_lsi:
        print(doc)

    lsi.save('state/big5.lsi')
    lsi = models.LsiModel.load('state/big5.lsi')

