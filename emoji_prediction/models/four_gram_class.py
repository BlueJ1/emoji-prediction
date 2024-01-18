class FourGram:
    def __init__(self, words):
        self.words = words

    def __getitem__(self, item):
        return self.words[item]

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return str(self.words)

    def __repr__(self):
        return str(self.words)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self.words))
