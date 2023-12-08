from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def balance_multiclass_dataset(X_train, Y_train):
    # Create a pipeline with random oversampling and random undersampling
    pipeline = Pipeline([
        ('over', RandomOverSampler(sampling_strategy='minority')),
        ('under', RandomUnderSampler(sampling_strategy='majority'))
    ])

    # Resample the dataset
    X_resampled, Y_resampled = pipeline.fit_resample(X_train, Y_train)

    return X_resampled, Y_resampled
