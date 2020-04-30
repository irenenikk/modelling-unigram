
class Alphabet:
    PAD_IDX = 0

    def __init__(self):
        self._chars2idx = {
            'PAD': 0,
            'SOS': 1,
            'EOS': 2
        }
        self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
        self._updated = True

    def add_word(self, word):
        for char in word:
            if char not in self._chars2idx:
                self._chars2idx[char] = len(self._chars2idx)
                self._updated = False

    def word2idx(self, word):
        return [self._chars2idx[char] for char in word]

    def char2idx(self, char):
        return self._chars2idx[char]

    def idx2word(self, idx_word):
        if not self._updated:
            self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
            self._updated = True
        return [self._idx2chars[idx.item()] for idx in idx_word if idx != self.char2idx('PAD')]

    def __len__(self):
        return len(self._chars2idx)
