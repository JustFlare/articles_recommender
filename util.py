import datetime
import conf
import os


def current_date():
    return datetime.datetime.now().strftime('%m%d%H%M')


def get_header(filename):
    with open(filename, 'r', encoding=conf.data_encoding) as f:
        return f.readline().replace('<h1>', '').replace('</h1>', '').replace('&quot;', '"')


def get_filename(p):
    return os.path.split(p)[1]