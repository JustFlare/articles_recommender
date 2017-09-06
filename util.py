import datetime
import conf
import os


def get_list(filename, encoding):
    with open(filename, 'r', encoding=encoding) as f:
        lines = f.read().splitlines()
    return lines


def cur_date():
    return datetime.datetime.now().strftime('%m%d%H%M')


def get_title(filename):
    with open(filename, 'r', encoding=conf.data_encoding) as f:
        return f.readline().replace('<h1>', '').replace('</h1>', '').replace('&quot;', '"')


def get_filename(p):
    return os.path.split(p)[1]

