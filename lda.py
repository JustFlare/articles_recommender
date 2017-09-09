
import numpy as np
import logging
import os

from util import get_title, cur_date, get_filename

from gensim.similarities import Similarity
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def make_corpus(docs, min_df, max_df):
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = Dictionary(docs)

    # remove frequency extremes
    dictionary.filter_extremes(no_below=min_df, no_above=max_df)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in docs]
    return dictionary, corpus


def fit_model(data, n_topics, iterations, passes, min_prob, eval_every, n_best, min_df, max_df):
    dt = cur_date()
    output_folder = "lda_%stopics_%s" % (n_topics, dt)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs("%s/separate" % output_folder, exist_ok=True)

    logging.info("creating corpus...")
    dictionary, corpus = make_corpus(list(data.values()), min_df, max_df)
    # generate LDA model
    logging.info("training model...")
    lda = LdaModel(corpus, num_topics=n_topics, id2word=dictionary, iterations=iterations,
                   passes=passes, minimum_probability=min_prob, eval_every=eval_every)
    logging.info("saving model...")
    lda.save('saved/lda_%s_%s.serialized' % (n_topics, dt))
    # print(lda.print_topics(num_topics=n_topics, num_words=4))

    # save all-vs-all pairwise similarities
    logging.info("creating index...")
    index = Similarity('./sim_index', lda[corpus], num_features=n_topics, num_best=n_best+1)
    paths = list(data.keys())
    logging.info("write all similarities to result file")
    with open('%s/similarities.txt' % output_folder, 'w') as res_file:
        with open('%s/similarities_summary.txt' % output_folder, 'w', encoding='utf-8') as res_file_sum:
            for i, similarities in enumerate(index):
                cur_fname = get_filename(paths[i])
                top_similar = [(paths[s[0]], s[1]) for s in similarities if s[0] != i]
                res_file.write('%s: %s\n' % (cur_fname,
                                             [(get_filename(p), c) for (p, c) in top_similar]))

                res_file_sum.write('%s: %s\n' % (cur_fname, get_title(paths[i])))
                for sim in top_similar:
                    res_file_sum.write('%s: %s' % (get_filename(sim[0]), get_title(sim[0])))
                res_file_sum.write('-' * 100 + '\n')

                # for each doc we make separate file which containts list of similar docs
                with open('%s/separate/%s.txt' % (output_folder, cur_fname.split('.')[0]), 'w') as sep_res:
                    sep_res.write('%s\n\n' % cur_fname)
                    for sim in top_similar:
                        sep_res.write('%s\n' % get_filename(sim[0]))

    logging.info("save index")
    index.save('saved/lda_index_%s.index' % dt)

    # save topic - words matrix
    with open("%s/topic_words.txt" % output_folder, 'w', encoding='utf-8') as f:
        for topic_words in lda.print_topics(lda.num_topics):
            f.write("#%s: %s\n" % (topic_words[0], topic_words[1]))

    # save document - topics matrix
    with open("%s/document_topics.txt" % output_folder, 'w') as f:
        for i, topics in enumerate(lda[corpus]):
            f.write("#%s: %s\n" % (get_filename(paths[i]), topics))


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
