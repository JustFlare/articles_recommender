# -*- coding: utf-8
import pickle
import time
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


def get_lda_topics(corpus, num_topics):
    # turn our tokenized documents into a id <-> term dictionary
    id2word = Dictionary(corpus)

    # convert tokenized documents into a document-term matrix
    corpus = [id2word.doc2bow(text) for text in corpus]

    # generate LDA model
    lda = LdaModel(corpus, num_topics=num_topics, id2word=id2word, passes=20)
    lda.save('ldamodel_%s.serialized' % num_topics)

    print(lda.print_topics(num_topics=num_topics, num_words=4))

    return lda


def train_doc2vec(n_epochs, dim, docs, load_from_file=False):
    '''
     docs - list of lists with words, one such list represents a document

     If you're finished training a model (=no more updates, only querying), you can do
     model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    '''
    tagged_docs = [TaggedDocument(sw_list, str(index)) for index, sw_list in enumerate(docs)]

    if load_from_file:
        model = Doc2Vec.load('doc2vec_%s.serialized' % dim)
    else:
        model = Doc2Vec(tagged_docs, dm=0, alpha=0.025, size=dim, min_alpha=0.025, min_count=0)

    for epoch in range(n_epochs):
        if epoch % 20 == 0:
            print('Now training epoch %s' % epoch)
        model.train(tagged_docs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    model.save('doc2vec_%s_300_epochs.serialized' % dim)
    print('doc2vec model saved')
    return model


if __name__ == '__main__':
    import sys

    old_stdout = sys.stdout
    log_file = open("grid_search.log", "w")
    sys.stdout = log_file


    lemmatized = False
    n_topics = 50
    n_epochs = 100

    #####################################################################

    print('start')

    if lemmatized:
        fname = 'articles.lemmatized.pkl'
    else:
        fname = 'articles.pkl'

    data = None
    with open(fname, mode='rb') as art_pkl:
        data = pickle.load(art_pkl)

    """print("Start training lda")
    start = time.time()
    get_lda_topics(data[0], n_topics)
    print(time.time() - start)"""

    print("\nStart doc2vec with %s epochs" % n_epochs)
    start = time.time()
    train_doc2vec(n_epochs=n_epochs, dim=n_topics, docs=data[0])
    print(time.time() - start)

    print('finish')

    sys.stdout = old_stdout
    log_file.close()

