import datetime
import numpy as np

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def get_filename(n_topics):
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    return 'saved/ldamodel_%s_%s.serialized' % (n_topics, dt)


def make_corpus(docs):
    # turn our tokenized documents into a id <-> term dictionary
    id2word = Dictionary(docs)
    # convert tokenized documents into a document-term matrix
    corpus = [id2word.doc2bow(text) for text in docs]
    return id2word, corpus


def fit_model(docs, n_topics, iterations, passes, min_prob, eval_every):
    id2word, corpus = make_corpus(docs)
    # generate LDA model
    lda = LdaModel(corpus, num_topics=n_topics, id2word=id2word, iterations=iterations,
                   passes=passes, minimum_probability=min_prob, eval_every=eval_every)
    lda.save(get_filename(n_topics))
    # print(lda.print_topics(num_topics=n_topics, num_words=4))
    return lda


def update_model(saved_model_path, docs):
    id2word, corpus = make_corpus(docs)
    lda = LdaModel.load(saved_model_path)
    lda.update(corpus)
    lda.save(get_filename(lda.n_topics))


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
