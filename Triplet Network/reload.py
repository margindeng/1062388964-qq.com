# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
# from tensorflow.models.rnn.ptb import reader
import os
import json
import re
# import gensim
#
# # Load Google's pre-trained Word2Vec model.
# #model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
# print(type(model['love']))
# data1 = np.array(model['love'])
# data2 = np.array(model['Leaflet'])
# print(data2)
# print(data1.shape)


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

all_texts = []
file_list = []

pos_path = 'talk_politics_mideast/'
for file in os.listdir(pos_path):
    file_list.append(pos_path+file)
# for file in os.listdir(file_list):
#     file_list.append(file)
# 将所有文本内容加到all_texts
for file_name in file_list:
    with open(file_name, encoding='utf-8') as f:
        all_texts.append(rm_tags(" ".join(f.readlines())))
#print(type(all_texts))
with open('1.words.txt', "wb") as txt_f:
    txt_f.write(str(all_texts).encode())
txt_f.close()
    #print(all_texts)
# tokenizer = Tokenizer(num_words=2000)  # 建立一个2000个单词的字典
# tokenizer.fit_on_texts(all_texts)
# x_id = tokenizer.texts_to_sequences(all_texts)
# x_train = sequence.pad_sequences(x_id, maxlen=150)
# print(type(x_id))
# #定义输入层
# input_data = tf.placeholder(tf.int32, [20, 35])
# embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
# print(embedding)
# # 将原本单词ID转为单词向量。
# embeddings = tf.nn.embedding_lookup(embedding, x_id[:1000])
#
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     embeddings_result=sess.run(embeddings)
#
# print(embeddings_result)