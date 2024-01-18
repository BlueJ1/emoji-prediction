import numpy as np
import pandas as pd
from pathlib import Path
from keras import Sequential
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, InputLayer
from mlp_sum import expand_array


def build_model(data):
    # Split the DataFrame into features (X) and labels (y)
    x = data.iloc[:, -200:]
    y = data.iloc[:, :48]
    print("Length of X: " + str(len(x.columns)) + " and length of y: " +
          str(len(y.columns)) + " and total: " + str(len(data.columns)))

    # make true/false to 1/0 in y
    y = y.astype(int)
    print(y)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    print(data.shape)
    # make an MLP model
    model = Sequential(
        [
            InputLayer(input_shape=(200,)),
            Dense(128, activation='relu'),
            Dropout(0.5),  # Adding dropout for regularization
            Dense(64, activation='relu'),
            Dropout(0.5),  # Adding dropout for regularization
            Dense(32, activation='relu'),  # fewer units
            Dropout(0.5),  # Adding dropout for regularization
            Dense(48, activation='sigmoid')
        ]
    )

    # Compile the model
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # k-fold cross validation
    k = 5
    num_val_samples = len(x_train) // k
    num_epochs = 10
    all_scores = []
    for i in range(k):
        print('processing fold #', i)
        # Prepare the validation data: data from partition # k
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

        # Prepare the training data: data from all other partitions
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y_train[:i * num_val_samples],
             y_train[(i + 1) * num_val_samples:]],
            axis=0)
        # Train the model (in silent mode, verbose=0)
        model.fit(partial_train_data, partial_train_targets,
                  epochs=num_epochs, batch_size=1, verbose=0)
        # Evaluate the model on the validation data
        val_binary_crossentropy, val_accuracy = model.evaluate(
            val_data, val_targets, verbose=0)
        all_scores.append(val_accuracy)
    print(all_scores)
    print(np.mean(all_scores))

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)
    # get balanced accuracy
    print('\n# Balanced accuracy on test data')
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    balanced_accuracy_score_value = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy score: {balanced_accuracy_score_value}")
    # get F1 score
    print('\n# F1 score on test data')
    f1_score_value = f1_score(y_test, y_pred, average='micro')
    print(f"F1 score: {f1_score_value}")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    file_name = 'word_around_emoji_concatenation_of_embeddings_200.pkl'

    df = pd.read_pickle(data_path / file_name)
    # print column names
    # print(df.columns)
    # print max length of words
    print(df['words'].apply(len).max())
    # make all words in ['words'] a column of its own (there are 200 words)
    expanded_df = df.apply(expand_array, axis=1)
    # Concatenate the expanded columns with the original DataFrame
    result_df = pd.concat([df, expanded_df], axis=1)
    # Drop the original array_column
    result_df = result_df.drop('words', axis=1)
    # print(result_df)
    print("These are the result columns: ", result_df.columns)

    build_model(result_df)
