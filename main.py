# coding: utf-8
import re
import os
import pickle
import logging
import traceback

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from pymystem3 import Mystem

import doc2vec
import lda

import conf


class UnexpectedArgumentException(Exception):
    """Raise for unexpected kind of arguments from config file"""

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


def main():
    print('start process')
    if conf.mode == 'fit':
        data = collect_data(conf.data_dir, conf.do_lemmatize, conf.lemmatized_data,
                            conf.data_encoding)
        if conf.algorithm == "doc2vec":
            doc2vec.fit_model(data, alpha=conf.alpha, n_epochs=conf.n_epochs,
                              vector_dim=conf.vector_dim, window=conf.window,
                              min_count=conf.min_count)
        elif conf.algorithm == "lda":
            lda.fit_model(data, n_topics=conf.topics, iterations=conf.iterations,
                          min_prob=conf.min_prob, passes=conf.passes,
                          eval_every=conf.eval_every)
        else:
            raise UnexpectedArgumentException("Invalid algorithm!")
    elif conf.mode == 'update':
        pass
    elif conf.mode == 'rank':
        pass
    else:
        raise UnexpectedArgumentException("Invalid mode!")

    print("Process finished.\n")
    # to save console after executing
    input("Press enter to exit")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception('')
        print(traceback.format_exc())
        input("Press enter to exit")
