import logging
import datetime

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


def get_filename(vector_dim):
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    return 'saved/doc2vec_%s_%s.serialized' % (vector_dim, dt)


def fit_model(docs, vector_dim, n_epochs, alpha, window, min_count):
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
    tagged_docs = [TaggedDocument(w_list, str(index)) for index, w_list in docs.items()]
    doc2vec = Doc2Vec(tagged_docs, dm=0, alpha=alpha, size=vector_dim, window=window,
                      min_count=min_count)
    logging.info("start training")
    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            logging.info('Training offset: %s' % epoch)
            print('Training offset: %s' % epoch)
        doc2vec.train(tagged_docs)
        doc2vec.alpha -= 0.002  # decrease the learning rate
        doc2vec.min_alpha = doc2vec.alpha  # fix the learning rate, no decay

    logging.info("saving model...")
    doc2vec.save(get_filename(vector_dim))
    # doc2vec.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return doc2vec


def update_model(saved_model_path, docs, n_epochs):
    logging.info("loading model")
    doc2vec = Doc2Vec.load(saved_model_path)
    logging.info("creating tagged docs...")
    tagged_docs = [TaggedDocument(w_list, str(index)) for index, w_list in docs.items()]

    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            logging.info('Training offset: %s' % epoch)
            print('Training offset: %s' % epoch)
        doc2vec.train(tagged_docs)
        doc2vec.alpha -= 0.002  # decrease the learning rate
        doc2vec.min_alpha = doc2vec.alpha  # fix the learning rate, no decay

    logging.info("saving model...")
    doc2vec.save(get_filename(doc2vec.vector_dim))


def transform_to_doc2vec_space(doc2vec, doc):
    """
    :param Doc2Vec doc2vec:
    :param list    doc:
    """
    return doc2vec.infer_vector(doc)
