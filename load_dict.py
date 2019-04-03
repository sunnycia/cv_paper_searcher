import os
import cPickle as pkl

def load_dict(dict_path):
    # dictionary = {}
    if os.path.isfile(dict_path):
        dictionary = pkl.load(open(dict_path, 'rb'))
        return dictionary
    else:
        print dict_path, 'not exists.'
        return {}
