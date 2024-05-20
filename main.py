import os
import sys
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
dependencies_path = os.path.join(script_path, 'dependencies')
output_path = os.path.join(script_path, 'output')
input_path = os.path.join(script_path, 'input')


def vectorize_train(df_not_vec):
    data = df_not_vec[['PULocationID', 'DOLocationID']].astype(str).to_dict(orient='records')
    vectorizer = DictVectorizer(sparse=True)
    feature_matrix = vectorizer.fit_transform(data)
    dimensionality = len(vectorizer.vocabulary_)
    print("Dimensionality of feature matrix (train):", dimensionality)
    return feature_matrix, vectorizer


def vectorize_test(df_not_vec, vectorizer):
    data = df_not_vec[['PULocationID', 'DOLocationID']].astype(str).to_dict(orient='records')
    feature_matrix = vectorizer.transform(data)
    return feature_matrix


def preprocess(df):
    initial_rows = df.shape[0]
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    print("Standard deviation of duration:", df['duration'].std())
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    final_rows = df.shape[0]
    diff_rows_percent = (final_rows / initial_rows)
    print("Fraction of rows retained:", diff_rows_percent)
    return df


if __name__ == '__main__':
    train_path = r'yellow_tripdata_2023-01.parquet'
    test_path = r'yellow_tripdata_2023-02.parquet'

    df_train = pd.read_parquet(os.path.join(input_path, train_path))
    print("Training data shape:", df_train.shape)
    df_train = preprocess(df_train)
    df_train_vec, vectorizer = vectorize_train(df_train)

    # Initialize and fit the linear regression model
    regressor = LinearRegression()
    regressor.fit(df_train_vec, df_train['duration'])

    # Predict durations on training data
    predictions_train = regressor.predict(df_train_vec)
    rmse_train = np.sqrt(mean_squared_error(df_train['duration'], predictions_train))
    print("RMSE on training data:", rmse_train)

    df_test = pd.read_parquet(os.path.join(input_path, test_path))
    df_test = preprocess(df_test)
    df_test_vec = vectorize_test(df_test, vectorizer)

    # Predict durations on test data
    predictions_test = regressor.predict(df_test_vec)
    rmse_test = np.sqrt(mean_squared_error(df_test['duration'], predictions_test))
    print("RMSE on test data:", rmse_test)
