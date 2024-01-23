from pathlib import Path
import nltk
from nltk.corpus import stopwords

# Downloading the stopwords from nltk
nltk.download('stopwords')

# Defining the paths for the data files
data_path = Path(__file__).parent.parent / 'data'
embedding_path = data_path / 'glove.6B.50d.txt'
vocab_path = data_path / 'vocab.txt'
vocab_without_stopwords_path = data_path / 'vocab_without_stopwords.txt'
train_path = data_path / 'train.txt'
emojis_path = data_path / 'emojis.txt'

# Initializing the word and emoji vocabularies
word_vocab = {}
vocab_without_stopwords = {}

# Reading the embeddings file and populating the word vocabularies
with open(embedding_path, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        # If the word is not a stopword, add it to the vocab_without_stopwords dictionary
        if word not in stopwords.words('english'):
            vocab_without_stopwords[word] = len(vocab_without_stopwords)
        # Add the word to the word_vocab dictionary
        word_vocab[word] = len(word_vocab)

# Writing the word vocabularies to their respective files
with open(vocab_path, 'w+', encoding='utf-8') as file:
    for word in word_vocab:
        file.write(word + '\n')

with open(vocab_without_stopwords_path, 'w+', encoding='utf-8') as file:
    for word in vocab_without_stopwords:
        file.write(word + '\n')

# Initializing the emoji vocabulary
emoji_vocab = {}

# Reading the training data file and populating the emoji vocabulary
with open(train_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        split = line.split()
        # If the line contains a word and an emoji
        if len(split) == 2:
            # If the word is not in the word_vocab dictionary, skip this line
            if split[0] not in word_vocab:
                continue
            # If the emoji is not in the emoji_vocab dictionary, add it
            if split[1] not in emoji_vocab:
                emoji_vocab[split[1]] = len(emoji_vocab)

# Writing the emoji vocabulary to its file
with open(emojis_path, 'w+', encoding='utf-8') as file:
    for emoji in emoji_vocab:
        file.write(emoji + '\n')
