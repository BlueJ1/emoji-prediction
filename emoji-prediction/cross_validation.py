from models.one_gram import one_gram
from models.generate_four_gram import four_gram
from models.one_gram import one_gram
from models.four_gram import four_gram

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd

parameters = [
    dict(
        model='one_gram',
        data_file='word_before_emoji_index.pkl'
    ),
    dict(
        model='four_gram',
        data_file='words_around_emoji_index.pkl'
    )]


k = 5

for parameter_dict in parameters:
    model = parameter_dict['model']
    data_file = parameter_dict['data_file']
    df = pd.read_pickle(data_file)

    X = df['X']
    y = df['y']

    if model == 'one_gram':
        generate_model = generate_one_gram
        predict_model = predict_one_gram
    elif model == 'four_gram':
        generate_model = generate_four_gram
        predict_model = predict_four_gram
    else:
        raise ValueError('Model must be one of "one_gram" or "four_gram"')

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(cv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # X_train, y_train = balance(X_train, y_train)
        generate_model(X_train, y_train)
        predictions = predict_model(X_test)

        score = f1_score(y_test, predictions, average='weighted')
