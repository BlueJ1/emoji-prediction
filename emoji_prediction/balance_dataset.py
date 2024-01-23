from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# import tensorflow as tf


def balance_multiclass_dataset(X_train, Y_train):
    # Create a pipeline with random oversampling and random undersampling
    pipeline = Pipeline([
        ('over', RandomOverSampler(sampling_strategy='minority')),
        ('under', RandomUnderSampler(sampling_strategy='majority'))
    ])

    # is_tf = isinstance(X_train, tf.Tensor)
    # if is_tf:
    #     X_train = X_train.numpy()
    #     Y_train = Y_train.numpy()

    # Resample the dataset
    X_resampled, Y_resampled = pipeline.fit_resample(X_train, Y_train)

    # if is_tf:
    #     X_resampled = tf.convert_to_tensor(X_resampled, dtype=tf.float32)
    #     Y_resampled = tf.convert_to_tensor(Y_resampled, dtype=tf.float32)

    return X_resampled, Y_resampled
