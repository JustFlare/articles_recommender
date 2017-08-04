import datetime
import logging
import numpy as np

from gensim.similarities import Similarity
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def current_date():
    return datetime.datetime.now().strftime('%m%d_%H%M')


def make_corpus(docs):
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = Dictionary(docs)

    # remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    # TODO: сделать задание этих параметров юзером
    # dictionary.filter_extremes(no_below=1, no_above=0.8)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in docs]
    return dictionary, corpus


def fit_model(data, n_topics, iterations, passes, min_prob, eval_every, n_best):
    logging.info("creating corpus...")
    dictionary, corpus = make_corpus(list(data.values()))
    # generate LDA model
    logging.info("training model...")
    lda = LdaModel(corpus, num_topics=n_topics, id2word=dictionary, iterations=iterations,
                   passes=passes, minimum_probability=min_prob, eval_every=eval_every)
    logging.info("saving model...")
    dt = current_date()
    lda.save('saved/lda_%s_%s.serialized' % (n_topics, dt))
    # print(lda.print_topics(num_topics=n_topics, num_words=4))

    # get all-vs-all pairwise similarities
    logging.info("creating index...")
    index = Similarity('./sim_index', lda[corpus], num_features=n_topics, num_best=n_best+1)
    filenames = list(data.keys())
    logging.info("write all similarities to result file")
    with open('result_lda_%s.txt' % current_date(), mode='w') as res_file:
        for i, similarities in enumerate(index):
            top_similar = [(filenames[s[0]], s[1]) for s in similarities if s[0] != i]
            res_file.write('%s: %s\n' % (filenames[i], top_similar))
    logging.info("save index")
    index.save('saved/lda_index_%s.index' % dt)


def update_model(saved_model_path, data):
    logging.info("creating corpus...")
    id2word, corpus = make_corpus(list(data.values()))

    logging.info("loading model and its index")
    lda = LdaModel.load(saved_model_path)
    # TODO: может быть лучше сделать явное указание пользователем индекса в конфиге?
    index = Similarity.load('saved/lda_index_%s.index' % saved_model_path[-20:-11])

    logging.info("updating model and index")
    lda.update(corpus)
    index.add_documents(lda[corpus])

    logging.info("saving...")
    dt = current_date()
    lda.save('saved/lda_%s_%s.serialized' % (lda.n_topics, dt))
    index.save('saved/lda_index_%s.index' % dt)


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
