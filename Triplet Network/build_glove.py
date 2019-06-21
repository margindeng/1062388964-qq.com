# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np


if __name__ == '__main__':
    # Load vocab
    with open('vocab.words.txt', 'rb') as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)
    print(size_vocab)
    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with open('glove.840B.300d.txt','rb') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))
    print(len(embeddings))
    print(embeddings)
    # Save np.array to file
    np.savez_compressed('glove.npz', embeddings=embeddings)


