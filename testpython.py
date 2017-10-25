#!/usr/bin/python
import cPickle as pickle

x = {'a': 1,
     'b': 2}
pickle.dump(x, open('./x.pkl', 'w'))
