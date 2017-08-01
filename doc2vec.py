
import datetime

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


def fit_doc2vec(docs, vector_dim, n_epochs, alpha, min_alpha, window, min_count,
                from_file=''):
    '''
    :param docs: dict where key is name of file and value is 
    :param vector_dim: 
    :param n_epochs: number of training iteration
    :param from_file: 
    :return: 
    '''
    if from_file != "":
        return Doc2Vec.load(from_file)

    tagged_docs = [TaggedDocument(w_list, str(index)) for index, w_list in docs.items()]
    doc2vec = Doc2Vec(tagged_docs, dm=0, alpha=alpha, size=vector_dim, min_alpha=min_alpha,
                      window=window, min_count=min_count)

    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            print('Training offset: %s' % epoch)

        doc2vec.train(tagged_docs)
        doc2vec.alpha -= 0.002  # decrease the learning rate
        doc2vec.min_alpha = doc2vec.alpha  # fix the learning rate, no decay

    doc2vec.save('saved/doc2vec_%s_%s.serialized' % (vector_dim, datetime.datetime.now().strftime('%Y%m%d')))
    doc2vec.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return doc2vec


def transform_to_doc2vec_space(doc2vec, doc):
    """
    :param Doc2Vec doc2vec:
    :param list    doc:
    """
    return doc2vec.infer_vector(doc)
