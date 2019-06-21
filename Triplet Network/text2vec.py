# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
import sys
# import tensorflow.contrib.eager as tfe
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import tensorflow as tf
DATADIR = ''
params = {
    'dim': 300,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    'epochs': 25,
    'batch_size': 20,
    'buffer': 15000,
    'lstm_size': 100,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz'))
}


def fwords(name):
    return str(Path(DATADIR, '{}.words.txt'.format(name)))


def ftags(name):
    return str(Path(DATADIR, '{}.tags.txt'.format(name)))

def get_data(data_name):
    training_data = []
    training_label = []
    with open(Path(fwords(data_name)), 'rb', newline=None) as f_words:
        line_words_list = f_words.readlines()
        for i in range(len(line_words_list)):
            line_words = line_words_list[i]
            words_list=line_words.split()
            training_data.append((words_list,len(words_list)))
    return training_data,training_label

training_data,training_label=get_data('1')

testing_data,testing_label=get_data('2')

raw_train_data=[data[0] for data in training_data]
raw_test_data=[data[0] for data in testing_data]

print(len(training_data))
print(len(testing_data))
#
vocab_words = tf.contrib.lookup.index_table_from_file(
     params['words'], num_oov_buckets=params['num_oov_buckets'])
#
words_placeholder = tf.placeholder(tf.string, shape=[None])
# Word Embeddings
word_ids = vocab_words.lookup(words_placeholder)
# print(word_ids)
glove = np.load(params['glove'])['embeddings']  # np.array
variable = np.vstack([glove, [[0.] * params['dim']]])
variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
embeddings = tf.nn.embedding_lookup(variable, word_ids)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    tf.tables_initializer().run(session=sess)
    embeddings_list = []

    for words in raw_train_data[:1000]:
        embeddings_result=sess.run(embeddings, feed_dict={words_placeholder: words})
        # print(word_ids)
        # print(variable)
        embeddings_list.append(embeddings_result)
    print(embeddings_list)
    with open('1.pkl', 'wb') as f:

        # Pickle dictionary using protocol 0.
        pickle.dump(embeddings_list, f)
    embeddings_list = []
    for words in raw_test_data[:1000]:
        embeddings_result=sess.run(embeddings, feed_dict={words_placeholder: words})
        embeddings_list.append(embeddings_result)
    print(embeddings_list)
    with open('2.pkl', 'wb') as f:

        #Pickle dictionary using protocol 0.
       pickle.dump(embeddings_list, f)
