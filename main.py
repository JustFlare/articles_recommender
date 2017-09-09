# coding: utf-8
import os
import pickle
import logging
import traceback

from collections import OrderedDict

from preprocess import preprocess_text
from util import get_title, cur_date
import doc2vec
import lda
import conf


class UnexpectedArgumentException(Exception):
    """Raise for unexpected kind of arguments from config file"""


SAVED_DIR = "saved"
os.makedirs(SAVED_DIR, exist_ok=True)

LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(filename="%s/vectorization_%s.log" % (LOG_DIR, cur_date()),
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
        with open('./%s/articles.%spkl' % (SAVED_DIR, 'lemmatized.' if do_lemmatize else ''),
                  mode='wb') as art_pkl:
            pickle.dump(data, art_pkl)
    return data


def main():
    print('Start process')

    logging.info("start\ncollecting data...")
    data = collect_data(conf.data_dir, conf.do_lemmatize, conf.lemmatized_data,
                        conf.data_encoding)

    if conf.algorithm == "doc2vec":
        logging.info("fitting doc2vec...")
        doc2vec.fit_model(data, dm=conf.dm, alpha=conf.alpha, n_epochs=conf.n_epochs,
                          vector_dim=conf.vector_dim, window=conf.window,
                          min_count=conf.min_count, n_best=conf.num_best)
    elif conf.algorithm == "lda":
        logging.info("fitting lda...")
        lda.fit_model(data, n_topics=conf.topics, iterations=conf.iterations,
                      min_prob=conf.min_prob, passes=conf.passes,
                      eval_every=conf.eval_every, n_best=conf.num_best,
                      min_df=conf.min_df, max_df=conf.max_df)
    else:
        raise UnexpectedArgumentException("Invalid algorithm!")

    print("Process finished.\n")
    logging.info("Process finished.")
    input("Press enter to exit")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception('')
        print(traceback.format_exc())
        input("Press enter to exit")
