import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem

from util import get_list
import conf

html = re.compile(r'</?\w+[^>]*>')
link = re.compile('(https?://)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
quotes_escaped_symbols = re.compile(r'[«»"]|(&\w+;)')
numbers = re.compile(r'^\d+$')
tokenizer = RegexpTokenizer('\w+')
mystem = Mystem()

stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english')))\
    .union(set(get_list(conf.stopwords_list, encoding=conf.data_encoding)))


def preprocess_text(text, do_lemmatize):
    clean_text = link.sub('', quotes_escaped_symbols.sub('', html.sub('', text.lower())))

    words = [w for w in tokenizer.tokenize(clean_text)]
    if do_lemmatize:
        words = mystem.lemmatize(' '.join(words))

    words = [w for w in words if w.strip() and not numbers.match(w) and w not in stop_words]
    return words

