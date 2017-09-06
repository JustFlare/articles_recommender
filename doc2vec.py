import logging

from util import get_title, cur_date, get_filename

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


def fit_model(docs, vector_dim, n_epochs, alpha, window, min_count, n_best):
    '''
    :param docs: dict where key is name of file and value is 
    :param vector_dim: dimensionality of the feature vector
    :param n_epochs: number of training iteration
    :param alpha: initial learning rate
    :param window: maximum distance between the predicted word and context words used for prediction
    :param min_count: ignore all words with total frequency lower than this.
    :return: fitted doc2vec model
    '''
    logging.info("creating tagged docs...")
    tagged_docs = [TaggedDocument(w_list, [index]) for index, w_list in docs.items()]
    doc2vec = Doc2Vec(tagged_docs, dm=0, alpha=alpha, size=vector_dim, window=window,
                      min_count=min_count)
    logging.info("start training")
    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            logging.info('Training offset: %s' % epoch)
        doc2vec.train(tagged_docs, total_examples=doc2vec.corpus_count, epochs=doc2vec.iter)
        doc2vec.alpha -= 0.002  # decrease the learning rate
        doc2vec.min_alpha = doc2vec.alpha  # fix the learning rate, no decay

    logging.info("saving model...")
    dt = cur_date()
    doc2vec.save('saved/doc2vec_%s_%s.serialized' % (vector_dim, dt))
    # doc2vec.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    with open('result_doc2vec_%s_%sdim.txt' % (dt, vector_dim), mode='w') as res_file:
        with open('result_doc2vec_%s_%sdim_summary.txt' % (dt, vector_dim), mode='w', encoding='utf-8') as res_file_sum:
            for doc_index in docs.keys():
                fname = get_filename(doc_index)
                top_similar = doc2vec.docvecs.most_similar(doc_index, topn=n_best)
                res_file.write('%s: %s\n' % (fname,
                                             [(get_filename(p), c) for (p, c) in top_similar]))

                res_file_sum.write('%s: %s\n' % (fname.encode('utf-8'),
                                                 get_title(doc_index).encode('utf-8')))
                for sim in top_similar:
                    res_file_sum.write('%s: %s' % (get_filename(sim[0]).encode('utf-8'),
                                                   get_title(sim[0]).encode('utf-8')))
                res_file_sum.write('-'*100 + '\n')


def update_model(saved_model_path, docs, n_epochs):
    logging.info("loading model")
    doc2vec = Doc2Vec.load(saved_model_path)
    logging.info("creating tagged docs...")
    tagged_docs = [TaggedDocument(w_list, str(index)) for index, w_list in docs.items()]

    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            logging.info('Training offset: %s' % epoch)
            print('Training offset: %s' % epoch)
        doc2vec.train(tagged_docs, total_examples=doc2vec.corpus_count, epochs=doc2vec.iter)
        doc2vec.alpha -= 0.002  # decrease the learning rate
        doc2vec.min_alpha = doc2vec.alpha  # fix the learning rate, no decay

    logging.info("saving model...")
    doc2vec.save('saved/doc2vec_%s_%s.serialized' % (doc2vec.vector_dim, cur_date()))


def transform_to_doc2vec_space(doc2vec, doc):
    """
    :param Doc2Vec doc2vec:
    :param list    doc:
    """
    return doc2vec.infer_vector(doc)
