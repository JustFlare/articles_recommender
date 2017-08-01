import datetime
import numpy as np

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def fit_lda_model(corpus, num_topics, iterations, passes, min_prob, from_file='', ):
    if from_file != "":
        return LdaModel.load(from_file)
    # turn our tokenized documents into a id <-> term dictionary
    id2word = Dictionary(corpus)
    # convert tokenized documents into a document-term matrix
    corpus = [id2word.doc2bow(text) for text in corpus]
    # generate LDA model
    lda = LdaModel(corpus, num_topics=num_topics, id2word=id2word, iterations=iterations,
                   passes=passes)

    lda.save('saved/ldamodel_%s_%s.serialized' % (num_topics, datetime.datetime.now().strftime('%Y%m%d')))

    print(lda.print_topics(num_topics=num_topics, num_words=4))

    return lda


def transform_to_topic_space(lda, doc):
    """
    :param LdaModel lda:
    :param list     doc:
    """
    res = np.zeros((lda.num_topics,))

    for item in lda[lda.id2word.doc2bow(doc)]:
        ind, val = item
        res[ind] = val

    return res
