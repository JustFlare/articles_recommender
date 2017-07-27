# coding: utf-8
import numpy as np
import re
import os
import pickle
import logging
import traceback
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from pymystem3 import Mystem

from gensim.models.ldamodel import LdaModel
from doc2vec import *
from lda import *


from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA

import conf

logging.basicConfig(filename="vectorization.log",
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

np.random.seed(6778)

html = re.compile(r'</?\w+[^>]*>')
link = re.compile('(https?://)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
quotes_escaped_symbols = re.compile(r'[«»"]|(&\w+;)')
numbers = re.compile(r'^\d+$')
tokenizer = RegexpTokenizer('\w+')
mystem = Mystem()

stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english')))

# TODO: вынести в константы названия папок всякие и проч


def preprocess_text(text, do_lemmatize):
    clean_text = link.sub('', quotes_escaped_symbols.sub('', html.sub('', text.lower())))

    words = [w for w in tokenizer.tokenize(clean_text)]
    if do_lemmatize:
        words = mystem.lemmatize(' '.join(words))

    words = [w for w in words if w.strip() and not numbers.match(w) and w not in stop_words]
    return words


def collect_data(root_dir, do_lemmatize=True, from_file='', encoding='cp1251'):
    data = {}
    if from_file != '':
        with open(from_file, mode='rb') as art_pkl:
            data = pickle.load(art_pkl)
    else:
        for cur_root, dirs, files in os.walk(root_dir):
            for name in files:
                with open(os.path.join(cur_root, name), encoding=encoding) as tf:
                    data[name] = preprocess_text(tf.read(), do_lemmatize)
        with open('articles.%spkl' % ('lemmatized.' if do_lemmatize else ''), mode='wb') as art_pkl:
            pickle.dump(data, art_pkl)
    return data


def evaluate_models(dim, from_file=True, plot=True):
    X, y = collect_data('articles', from_file)

    lda = fit_lda_model(X, dim, from_file)
    doc2vec = fit_doc2vec_model(X, dim, 100, from_file)

    for model in [lda, doc2vec]:
        if isinstance(model, LdaModel):
            fit_X = [transform_to_topic_space(model, x) for x in X]
        else:
            fit_X = [transform_to_doc2vec_space(model, x) for x in X]

        cls = KMeans(n_clusters=len(set(y)))
        cls.fit(fit_X)

        pred_y = cls.labels_
        print('%s\nn_clusters=%s\nnormalized_mutual_info_score: %s\nadjusted_mutual_info_score: %s' % (
            type(model),
            cls.n_clusters,
            normalized_mutual_info_score(y, pred_y),
            adjusted_mutual_info_score(y, pred_y)
        ))

        if plot:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(fit_X)
            # Percentage of variance explained for each components
            print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

            plt.figure()
            for color, i in zip('bgrcmyk', range(cls.n_clusters)):
                plt.scatter(reduced_data[y == i, 0], reduced_data[y == i, 1],
                            color=color, s=10, alpha=.8, lw=2, label=i)
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('True labels\nPCA of %s' % type(model))

            plt.figure()
            for color, i in zip('bgrcmyk', range(cls.n_clusters)):
                plt.scatter(reduced_data[pred_y == i, 0], reduced_data[pred_y == i, 1],
                            color=color, s=10, alpha=.8, lw=2, label=i)
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('Predicted labels\nPCA of %s' % type(model))

            plt.show()


def main():
    print('start process')
    if conf.mode == 'fit':
        pass
    elif conf.mode == 'update':
        pass
    elif conf.mode == 'rank':
        pass
    else:
        input("Invalid mode! Try again")

    print("Process finished.\n" )
    # to save console after executing
    input("Press enter to exit")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception('')
        print(traceback.format_exc())
        input("Press enter to exit")
