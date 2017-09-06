# coding: utf-8
import re
import os
import pickle
import logging
import traceback

from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem

from util import get_title, cur_date
import doc2vec
import lda

import conf


class UnexpectedArgumentException(Exception):
    """Raise for unexpected kind of arguments from config file"""


LOG_DIR = "log"
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(filename="%s/vectorization_%s.log" % (LOG_DIR, cur_date()),
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    data = OrderedDict()
    if from_file != '' and do_lemmatize:
        logging.info("loading data from file")
        with open(from_file, mode='rb') as art_pkl:
            data = pickle.load(art_pkl)
    else:
        for cur_root, dirs, files in os.walk(root_dir):
            for name in files:
                with open(os.path.join(cur_root, name), encoding=encoding) as tf:
                    text = get_title(tf.name) if conf.only_title else tf.read()
                    data[tf.name] = preprocess_text(text, do_lemmatize)
        logging.info("saving collected data")
        with open('./saved/articles.%spkl' % ('lemmatized.' if do_lemmatize else ''), mode='wb') as art_pkl:
            pickle.dump(data, art_pkl)
    return data


def main():
    print('Start process')
    logging.info("start\ncollecting data...")
    data = collect_data(conf.data_dir, conf.do_lemmatize, conf.lemmatized_data,
                        conf.data_encoding)

    if conf.mode == 'fit':
        if conf.algorithm == "doc2vec":
            logging.info("fitting doc2vec...")
            doc2vec.fit_model(data, alpha=conf.alpha, n_epochs=conf.n_epochs,
                              vector_dim=conf.vector_dim, window=conf.window,
                              min_count=conf.min_count, n_best=conf.num_best)
        elif conf.algorithm == "lda":
            logging.info("fitting lda...")
            lda.fit_model(data, n_topics=conf.topics, iterations=conf.iterations,
                          min_prob=conf.min_prob, passes=conf.passes,
                          eval_every=conf.eval_every, n_best=conf.num_best)
        else:
            raise UnexpectedArgumentException("Invalid algorithm!")

    elif conf.mode == 'update':
        if conf.algorithm == "doc2vec":
            logging.info("updating doc2vec model {0}".format(conf.saved_model))
            # TODO: mb print warning that updating doc2vec is not recommended?
            doc2vec.update_model(conf.saved_model, data, conf.n_epochs)
        elif conf.algorithm == "lda":
            logging.info("updating lda model {0}".format(conf.saved_model))
            lda.update_model(data, conf.saved_model)
        else:
            raise UnexpectedArgumentException("Invalid algorithm!")

    elif conf.mode == 'rank':
        print("not implemented yet")

    else:
        raise UnexpectedArgumentException("Invalid mode!")

    print("Process finished.\n")
    logging.info("Process finished.")
    # to save console after executing
    input("Press enter to exit")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception('')
        print(traceback.format_exc())
        input("Press enter to exit")
