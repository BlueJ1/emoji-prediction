from pathlib import Path
import pandas as pd
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
import keras


def build_model(dataframe):
    # Split the DataFrame into features (X) and labels (y)
    # X is last 50 columns, y is first 49
    x = result_df.iloc[:, 48:].values
    # print(X)
    y = result_df.iloc[:, :48].values
    # print(y)
    X = pd.DataFrame(x)
    y = pd.DataFrame(y)
    # print("Length of X: " + str(len(X.columns)) + " and length of y: " +
    # str(len(y.columns)) + " and total: " + str(len(result_df.columns)))
    # make true/false to 1/0 in y
    y = y.astype(int)
    print(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the MLP model
    model = keras.Sequential([
        layers.Input(shape=(50,)),  # Assuming 50 features
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Adding dropout for regularization
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Adding dropout for regularization
        layers.Dense(32, activation='relu'),  # fewer units
        layers.Dropout(0.5),  # Adding dropout for regularization
        layers.Dense(48, activation='sigmoid')
        # Assuming 48 emoji classes
    ])

    # Compile the model
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    # model.fit(
    #    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # k-fold cross validation
    k = 5
    num_val_samples = len(X_train) // k
    num_epochs = 100
    all_scores = []
    for i in range(k):
        print('processing fold #', i)
        # Prepare the validation data: data from partition # k
        val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
        # Prepare the training data: data from all other partitions
        partial_train_data = np.concatenate(
            [X_train[:i * num_val_samples],
             X_train[(i + 1) * num_val_samples:]],
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

    # Get F1 score and balanced accuracy score
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    f1_score_value = f1_score(y_test, y_pred, average='micro')
    balanced_accuracy_score_value = balanced_accuracy_score(y_test, y_pred)
    print(f"F1 score: {f1_score_value}")
    print(f"Balanced accuracy score: {balanced_accuracy_score_value}")


def check_for_multiple_emojis(to_check):
    for index, row in to_check.iterrows():
        if len(row['words']) != 50:
            print("The row " + str(index) + " has a word array of length "
                  + str(len(row['words'])))

    classes_df = to_check.drop(columns=['words'])
    # iterate through all rows
    for index, row in classes_df.iterrows():
        # iterate through all columns
        for col in row:
            taken = -1
            if col:
                if taken != -1:
                    print("The row " + str(index) +
                          " has more than one emoji, namely "
                          + str(col) + " and " + str(taken))
                else:
                    taken = col


def expand_array(row):
    return pd.Series(row['words'])


data_path = Path(__file__).parent.parent / 'data'
model_path = Path(__file__).parent.parent / 'models'
file_name = 'word_around_emoji_sum_of_embeddings.pkl'

df = pd.read_pickle(data_path / file_name)
# print column names
print(df.columns)
check_for_multiple_emojis(df)

# Apply the function to the DataFrame
expanded_df = df.apply(expand_array, axis=1)
# Concatenate the expanded columns with the original DataFrame
result_df = pd.concat([df, expanded_df], axis=1)
# Drop the original array_column
result_df = result_df.drop('words', axis=1)
# Rename the new columns if needed
# result_df.columns = [f'value_{i+1}' for i in range(result_df.shape[1])]
print(result_df)
print(result_df.columns)

build_model(result_df)
