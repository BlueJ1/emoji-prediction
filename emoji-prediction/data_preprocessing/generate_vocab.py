from pathlib import Path

data_path = Path(__file__).parent.parent / 'data'
embedding_path = data_path / 'glove.6B.50d.txt'
vocab_path = data_path / 'vocab.txt'
train_path = data_path / 'train.txt'
emojis_path = data_path / 'emojis.txt'

word_vocab = {}  # {'UNK': 0, '<START>': 1, '<END>': 2, ' ': 3, '🏾': 4}
with open(embedding_path, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        word_vocab[word] = len(word_vocab)

with open(vocab_path, 'w+', encoding='utf-8') as file:
    for word in word_vocab:
        file.write(word + '\n')

emoji_vocab = {}

with open(train_path, 'r') as f:
    for i, line in enumerate(f):
        split = line.split()
        if len(split) == 2:
            if split[0] not in word_vocab:
                continue
            if split[1] not in emoji_vocab:
                emoji_vocab[split[1]] = len(emoji_vocab)

with open(emojis_path, 'w+', encoding='utf-8') as file:
    for emoji in emoji_vocab:
        file.write(emoji + '\n')

print(len(emoji_vocab))
