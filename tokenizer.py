# tokenizer.py
import re
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def train(self, texts):
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        most_common = word_counts.most_common(self.vocab_size - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
        return text.split()

    def encode(self, text, max_len):
        words = self._tokenize(text)[:max_len]
        ids = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        ids += [0] * (max_len - len(ids))
        return np.array(ids)

    def decode(self, ids):
        words = [self.idx2word.get(idx, '<UNK>') for idx in ids if idx != 0]
        return ' '.join(words)
