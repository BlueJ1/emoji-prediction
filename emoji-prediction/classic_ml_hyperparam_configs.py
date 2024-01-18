from models.four_gram import four_gram, four_gram_data
from models.one_gram import one_gram, one_gram_data
from models.baseline import baseline, baseline_data
from models.classic_ml_models import (basic_ml_data, train_rf, train_svm, train_k_nbh, train_naive_bayes,
                                      train_log_reg, train_qda)

parameters = [
    dict(
        name='baseline',
        data_preprocessing=baseline_data,
        data_file='word_before_emoji_index.pkl',
        evaluate=baseline,
        hyperparameters=dict(),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='one_gram',
        data_preprocessing=one_gram_data,
        data_file='word_before_emoji_index.pkl',
        evaluate=one_gram,
        hyperparameters=dict(),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='four_gram',
        data_preprocessing=four_gram_data,
        data_file='words_around_emoji_index.pkl',
        evaluate=four_gram,
        hyperparameters=dict(),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='random_forest300GiniLog2',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_rf,
        hyperparameters=dict(n_estimators=300, criterion='gini', max_features='log2'),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='random_forest300EntropySqrt',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_rf,
        hyperparameters=dict(n_estimators=300, criterion='entropy', max_features='sqrt'),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='random_forest100GiniSqrt',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_rf,
        hyperparameters=dict(n_estimators=100, criterion='gini', max_features='sqrt'),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='random_forest100EntropyLog2',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_rf,
        hyperparameters=dict(n_estimators=100, criterion='entropy', max_features='log2'),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='random_forest30EntropySqrt',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_rf,
        hyperparameters=dict(n_estimators=30, criterion='entropy', max_features='sqrt'),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='Quadrant Discriminant Analysis',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_qda,
        hyperparameters=dict(),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='k_neighbors3',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_k_nbh,
        hyperparameters=dict(num_neighbors=3),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='k_neighbors10',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_k_nbh,
        hyperparameters=dict(num_neighbors=10),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='k_neighbors30',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_k_nbh,
        hyperparameters=dict(num_neighbors=30),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='naive_bayes',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_naive_bayes,
        hyperparameters=dict(),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='logistic_regressionC1.0ElasticNet0.5',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_log_reg,
        hyperparameters=dict(C=1.0, penalty='elasticnet', l1_ratio=0.5),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='logistic_regressionC0.1ElasticNet0.5',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_log_reg,
        hyperparameters=dict(C=0.1, penalty='elasticnet', l1_ratio=0.5),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='logistic_regressionC0.01ElasticNet0.5',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_log_reg,
        hyperparameters=dict(C=0.01, penalty='elasticnet', l1_ratio=0.5),
        balance_dataset=False,
        parallel=False
    ),
    dict(
        name='svmRBFC1.0tol1e-3',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="rbf", C=1.0, tol=1e-3),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmPolyC0.5tol1e-3',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="poly", C=0.5, tol=1e-3),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmPolyC0.1tol2e-4',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="poly", C=0.1, tol=2e-4),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmRBFC0.1tol1e-3',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="rbf", C=0.1, tol=1e-3),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmRBFC0.01tol1e-3',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="rbf", C=0.01, tol=1e-3),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmRBFC0.5tol1e-4',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="rbf", C=0.5, tol=1e-4),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmRBFC0.1tol1e-4',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="rbf", C=0.1, tol=1e-4),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmSigmoidC0.01tol1e-3',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="sigmoid", C=0.01, tol=1e-3),
        balance_dataset=False,
        parallel=True
    ),
    dict(
        name='svmSigmoidC0.7tol7e-4',
        data_preprocessing=basic_ml_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_svm,
        hyperparameters=dict(kernel="sigmoid", C=0.7, tol=7e-4),
        balance_dataset=False,
        parallel=True
    ),
]


if __name__ == '__main__':
    print(len(parameters))
