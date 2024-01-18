import pandas as pd
from keras.src.metrics import F1Score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy


# Assuming you've DataFrame 'train': 'prev_word1', 'prev_word2',
# 'future_word1', 'future_word2', 'emoji_index'

# Load your data and preprocess it
train = pd.read_csv('emoji_prediction/data/train.csv')
test = pd.read_csv('emoji_prediction/data/test.csv')

# Use LabelEncoder to convert emoji indexes to sequential
# integers (string to int)
emoji_encoder = LabelEncoder()
train['emoji_index'] = emoji_encoder.fit(train['emoji_index'])

# Split the data into train and test sets
train_data, test_data = train_test_split(train, test_size=0.2, random_state=42)

# Assuming you have pre-trained word embeddings (e.g., Word2Vec, GloVe)
# Load word embeddings and create embedding matrix
embeddings = pd.read_csv('data/glove.6B.50d.txt',
                         sep=' ', header=None, index_col=0)

# Define the input layers
prev_word1_input = Input(shape=(50,), name='prev_word1_input')
prev_word2_input = Input(shape=(50,), name='prev_word2_input')
future_word1_input = Input(shape=(50,), name='future_word1_input')
future_word2_input = Input(shape=(50,), name='future_word2_input')
emoji_index_input = Input(shape=(1,), name='emoji_index_input')

# Embedding layers for emoji_index
emoji_embedding = Embedding(input_dim=len(emoji_encoder.classes_),
                            output_dim=50, input_length=1)(emoji_index_input)
emoji_embedding_flat = Flatten()(emoji_embedding)

# Concatenate all input embeddings
concatenated_inputs = Concatenate()([prev_word1_input, prev_word2_input,
                                     future_word1_input, future_word2_input,
                                     emoji_embedding_flat])

# Define the MLP layers
dense_layer1 = Dense(128, activation='relu')(concatenated_inputs)
dense_layer2 = Dense(64, activation='relu')(dense_layer1)

# Output layer with softmax activation for predicting emojis
output_layer = Dense(len(emoji_encoder.classes_), activation='softmax',
                     name='output')(dense_layer2)

# Build the model
model = Model(inputs=[prev_word1_input, prev_word2_input, future_word1_input,
                      future_word2_input, emoji_index_input],
              outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy(),
                       F1Score(num_classes=len(emoji_encoder.classes_),
                               average='micro')])

# Train the model
model.fit([train_data['prev_word1'], train_data['prev_word2'],
           train_data['future_word1'], train_data['future_word2'],
           train_data['emoji_index']], train_data['emoji'], epochs=10,
          validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([test_data['prev_word1'],
                                           test_data['prev_word2'],
                                           test_data['future_word1'],
                                           test_data['future_word2'],
                                           test_data['emoji_index']],
                                          test_data['emoji'])
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Save the model if needed
model.save('emoji_prediction_mlp_model.h5')
