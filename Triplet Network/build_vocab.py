
from collections import Counter
from pathlib import Path
import os

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

if __name__ == '__main__':
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return '{}.words.txt'.format(name)

    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['1', '2']:
        with open(os.path.join("data",words(n)),'r') as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path('vocab.words.txt').open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))


