import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import pickle
import datetime
from dnn_with_preprocess_module import dnn_with_preprocess
from IPython.display import display
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
import tensorflow as tf

np.random.seed(42)
random.seed(42)


def get_split(data, ratio=0.8, normalise=True, shuffle=False, print_outputs=False):
    """
    Method for returning training/test split with optional normalisation/shuffling

    Inputs: ratio (proportion of training data)
            normalise (if True, normalises data)
            shuffle (if True, shuffles data)
    Outputs: train and test data
    """
    if shuffle:
        data = data.sample(frac=1)
    data = data.values
    train_idx = int(data.shape[0] * ratio)
    x_train, y_train = data[:train_idx, :-1], data[:train_idx, -1]
    x_test, y_test = data[train_idx:, :-1], data[train_idx:, -1]

    if print_outputs:
        print(
            "\033[1mProportion of 1s in Training Data:\033[0m {}%".format(
                round(np.average(y_train) * 100, 2)
            )
        )
        print(
            "\033[1mProportion of 1s in Test Data:\033[0m {}%".format(
                round(np.average(y_test) * 100, 2)
            )
        )

    return x_train, y_train, x_test, y_test


def load_models(dataset, model_name):
    B_name = model_name

    if os.path.exists("models/{}_{}.pkl".format(dataset, B_name)):
        with open("models/{}_{}.pkl".format(dataset, B_name), "rb") as f:
            B = pickle.load(f)
    else:
        if model_name == "lr":
            B = LogisticRegression()
        elif model_name == "xgb":
            B = XGBClassifier()
        else:
            print(f"load_models function cannot deal with these arguments: {dataset=}, {model_name=}. Returning None.")
            B = None

    return B


def process_compas(data):
    """
    Additional method to process specifically the COMPAS dataset

    Input: data (whole dataset)
    Output: data (whole dataset)
    """
    data = data.to_dict("list")
    for k in data.keys():
        data[k] = np.array(data[k])

    dates_in = data["c_jail_in"]
    dates_out = data["c_jail_out"]
    # this measures time in Jail
    time_served = []
    for i in range(len(dates_in)):
        di = datetime.datetime.strptime(dates_in[i], "%Y-%m-%d %H:%M:%S")
        do = datetime.datetime.strptime(dates_out[i], "%Y-%m-%d %H:%M:%S")
        time_served.append((do - di).days)
    time_served = np.array(time_served)
    time_served[time_served < 0] = 0
    data["time_served"] = time_served

    """ Filtering the data """
    # These filters are as taken by propublica
    # (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days
    # from when the person was arrested, we assume that because of data quality
    # reasons, that we do not have the right offense.
    idx = np.logical_and(
        data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30
    )

    # We coded the recidivist flag -- is_recid -- to be -1
    # if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of
    # 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")
    # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows
    # representing people who had either recidivated in two years, or had at least two
    # years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]
    return pd.DataFrame(data)


def load_models_kfold(dataset, data, model, model_name, cv):
    # model_list_path = f"models/{dataset}_{model_name}_model_list.pkl"
    # affected_list_path = f"models/{dataset}_{model_name}_affected_list.pkl"
    # train_dataset_list_path = f"models/{dataset}_{model_name}_train_dataset_list.pkl"
    model_list = []
    affected_list = []
    unaffected_list = []
    train_dataset_list = []

    # try:
    #     with open(model_list_path, "rb") as f:
    #         model_list = pickle.load(f)
    #     print("Model list loaded successfully.")

    #     with open(affected_list_path, "rb") as f:
    #         affected_list = pickle.load(f)
    #     print("Affected list loaded successfully.")

    #     with open(train_dataset_list_path, "rb") as f:
    #         train_dataset_list = pickle.load(f)
    #     print("Train_dataset list loaded successfully.")
    # except FileNotFoundError:
    #     print(f"Pickle file for {dataset} does not exist. Creating models with kfold")
    # except Exception as e:
    #     print("An error occurred while loading the pickle file:", e)

    if model_list == []:
        print(f"Kfold for dataset {dataset}")
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Creating fold {fold+1}:")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if dataset == "germannumeric_credit":
                X_train = pd.DataFrame(X_train, columns=data.columns[:-1])
                X_test = pd.DataFrame(X_test, columns=data.columns[:-1])
                num_features = X_train._get_numeric_data().columns.to_list()
                cate_features = X_train.columns.difference(num_features)

                if model_name == "dnn":
                    model_ = dnn_with_preprocess(
                        model, "german_credit_numeric", X_train, X_test
                    )
                else:

                    class IdentityTransformer(BaseEstimator, TransformerMixin):
                        def __init__(self):
                            pass

                        def fit(self, input_array, y=None):
                            return self

                        def transform(self, input_array, y=None):
                            return input_array * 1

                    model_ = Pipeline(
                        [("preprocessor", IdentityTransformer()), ("classifier", model)]
                    )

                    model_.fit(X_train, y_train)
                feat_to_vary = list(X_train.columns)
            elif dataset == "german_credit":
                dtype_dict = {
                    "Month-Duration": "int64",
                    "Credit-Amount": "int64",
                    "Age": "int64",
                    "Instalment-Rate": "int64",
                    "Residence": "int64",
                    "Existing-Credits": "int64",
                    "Num-People": "int64",
                }
                cols = data.columns
#                 X_train, y_train, X_test, y_test = get_split(
#                     data, normalise=False, shuffle=False
#                 )
                X_train = pd.DataFrame(X_train, columns=cols[:-1]).astype(dtype_dict)
                X_test = pd.DataFrame(X_test, columns=cols[:-1]).astype(dtype_dict)
                X_test.columns = cols[:-1]

                y_train = pd.Series(y_train, dtype="int32")
                y_test = pd.Series(y_test, dtype="int32")

                num_features = X_train._get_numeric_data().columns.to_list()
                cate_features = X_train.columns.difference(num_features)

                X_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_test.reset_index(drop=True, inplace=True)

                if model_name == "dnn":
                    model_ = Sequential(
                        [
                            Dense(8, activation="relu"),
                            Dense(4, activation="relu"),
                            Dense(1, activation="sigmoid"),
                        ]
                    )
                    model_.compile(
                        optimizer=Adam(learning_rate=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("normalize", StandardScaler()),
                            ("classifier", model_),
                        ]
                    )
                    model_.fit(
                        X_train,
                        y_train,
                        classifier__batch_size=16,
                        classifier__epochs=15,
                        classifier__verbose=0,
                        classifier__validation_split=0.2,
                    )

                    tmp = model_.predict

                    def new_predict(X):
                        X = model_.named_steps["preprocessor"].transform(X)
                        X = model_.named_steps["normalize"].transform(X)
                        preds_proba = model_.named_steps["classifier"](X)
                        preds_proba = preds_proba.numpy()
                        return (preds_proba > 0.5).astype(int).flatten()
                    def scorer_fn(X, y):
                        return accuracy_score(y, new_predict(X))

                    model_.predict = new_predict
                    model_.predict_proba = tmp
                    model_.score = scorer_fn

                else:
                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("classifier", clone(model)),
                        ]
                    )
                    model_.fit(X_train, y_train)
                feat_to_vary = list(X_train.columns)
                feat_to_vary.remove("Sex")
                feat_to_vary.remove("Foreign-Worker")
                target_name = "Status"
            elif dataset == "compas":
                cols = data.columns
                X_train = pd.DataFrame(X_train)
                X_train.columns = cols[:-1]
                X_test = pd.DataFrame(X_test)
                X_test.columns = cols[:-1]
                dtype_dict = {"Priors_Count": "int32", "Time_Served": "int32"}
                X_train = X_train.astype(dtype_dict)
                X_test = X_test.astype(dtype_dict)
                y_train = pd.Series(y_train, dtype="int32")
                y_test = pd.Series(y_test, dtype="int32")
                num_features = X_train._get_numeric_data().columns.to_list()
                cate_features = X_train.columns.difference(num_features)
                X_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_test.reset_index(drop=True, inplace=True)

                if model_name == "dnn":
                    model_ = Sequential(
                        [
                            Dense(64, activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(16, activation="relu"),
                            Dense(1, activation="sigmoid"),
                        ]
                    )
                    model_.compile(
                        optimizer=Adam(learning_rate=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("normalize", StandardScaler()),
                            ("classifier", model_),
                        ]
                    )
                    model_.fit(
                        X_train,
                        y_train,
                        classifier__batch_size=16,
                        classifier__epochs=15,
                        classifier__verbose=0,
                        classifier__validation_split=0.2,
                    )

                    tmp = model_.predict

                    def new_predict(X):
                        X = model_.named_steps["preprocessor"].transform(X)
                        X = model_.named_steps["normalize"].transform(X)
                        preds_proba = model_.named_steps["classifier"](X)
                        preds_proba = preds_proba.numpy()
                        return (preds_proba > 0.5).astype(int).flatten()
                    def scorer_fn(X, y):
                        return accuracy_score(y, new_predict(X))

                    model_.predict = new_predict
                    model_.predict_proba = tmp
                    model_.score = scorer_fn
                else:

                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("classifier", clone(model)),
                        ]
                    )

                    model_.fit(X_train, y_train)
                feat_to_vary = list(X_train.columns)
                feat_to_vary.remove("Sex")
                feat_to_vary.remove("Race")
                target_name = "Status"
            elif dataset == "heloc":
                X_train = pd.DataFrame(X_train)
                X_train.columns = data.columns[:-1]
                X_test = pd.DataFrame(X_test)
                X_test.columns = data.columns[:-1]
                X_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_test.reset_index(drop=True, inplace=True)
                y_train = pd.Series(y_train, dtype="int32")
                y_test = pd.Series(y_test, dtype="int32")
                num_features = X_train._get_numeric_data().columns.to_list()
                cate_features = X_train.columns.difference(num_features)
                if model_name == "dnn":
                    model_ = Sequential(
                        [
                            Dense(64, activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(16, activation="relu"),
                            Dense(8, activation="relu"),
                            Dense(1, activation="sigmoid"),
                        ]
                    )
                    model_.compile(
                        optimizer=Adam(learning_rate=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("normalize", StandardScaler()),
                            ("classifier", model_),
                        ]
                    )
                    model_.fit(
                        X_train,
                        y_train,
                        classifier__batch_size=16,
                        classifier__epochs=15,
                        classifier__verbose=0,
                        classifier__validation_split=0.2,
                    )

                    tmp = model_.predict

                    def new_predict(X):
                        X = model_.named_steps["preprocessor"].transform(X)
                        X = model_.named_steps["normalize"].transform(X)
                        preds_proba = model_.named_steps["classifier"](X)
                        preds_proba = preds_proba.numpy()
                        return (preds_proba > 0.5).astype(int).flatten()
                    def scorer_fn(X, y):
                        return accuracy_score(y, new_predict(X))

                    model_.predict = new_predict
                    model_.predict_proba = tmp
                    model_.score = scorer_fn
                else:

                    class IdentityTransformer(BaseEstimator, TransformerMixin):
                        def __init__(self):
                            pass

                        def fit(self, input_array, y=None):
                            return self

                        def transform(self, input_array, y=None):
                            return input_array * 1

                    model_ = Pipeline(
                        [("preprocessor", IdentityTransformer()), ("classifier", clone(model))]
                    )

                    model_.fit(X_train, y_train)

                feat_to_vary = list(X_train.columns)
                feat_to_vary.remove('PercentTradesWBalance')
                target_name = "RiskPerformance"
            elif dataset == "default_credit":
                df = data
                X_train = pd.DataFrame(X_train)
                X_train.columns = df.columns[:-1]
                X_test = pd.DataFrame(X_test)
                X_test.columns = df.columns[:-1]
                X_train = X_train.astype(dict(df.drop(columns="target").dtypes))

                X_test = X_test.astype(dict(df.drop(columns="target").dtypes))
                y_train = pd.Series(y_train, dtype="int32")
                y_test = pd.Series(y_test, dtype="int32")
                X_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_test.reset_index(drop=True, inplace=True)
                num_features = X_train._get_numeric_data().columns.to_list()
                cate_features = X_train.columns.difference(num_features)
                X_train[cate_features] = X_train[cate_features].astype('category')
                
                if model_name == "dnn":
                    model_ = Sequential(
                        [
                            Dense(64, activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(16, activation="relu"),
                            Dense(8, activation="relu"),
                            Dense(1, activation="sigmoid"),
                        ]
                    )
                    model_.compile(
                        optimizer=Adam(learning_rate=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("normalize", StandardScaler()),
                            ("classifier", model_),
                        ]
                    )
                    model_.fit(
                        X_train,
                        y_train,
                        classifier__batch_size=16,
                        classifier__epochs=15,
                        classifier__verbose=0,
                        classifier__validation_split=0.2,
                    )

                    tmp = model_.predict

                    def new_predict(X):
                        X = model_.named_steps["preprocessor"].transform(X)
                        X = model_.named_steps["normalize"].transform(X)
                        preds_proba = model_.named_steps["classifier"](X)
                        preds_proba = preds_proba.numpy()
                        return (preds_proba > 0.5).astype(int).flatten()
                    def scorer_fn(X, y):
                        return accuracy_score(y, new_predict(X))

                    model_.predict = new_predict
                    model_.predict_proba = tmp
                    model_.score = scorer_fn
                elif model_name == "lr":

                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            ),
                            (
                                "num",
                                StandardScaler(),
                                num_features,
                            ),
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("classifier", clone(model)),
                        ]
                    )

                    model_.fit(X_train, y_train)
                else:

                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("classifier", clone(model)),
                        ]
                    )
                    model_.fit(X_train, y_train)

                feat_to_vary = list(X_train.columns)
                feat_to_vary.remove("SEX")
                target_name = "target"
                
            elif dataset == "adult":
                cols = data.columns
                X_train = pd.DataFrame(X_train)
                X_train.columns = cols[:-1]
                X_test = pd.DataFrame(X_test)
                X_test.columns = cols[:-1]
                y_train = pd.Series(y_train, dtype="int32")
                y_test = pd.Series(y_test, dtype="int32")
                num_features = X_train._get_numeric_data().columns.to_list()
                cate_features = X_train.columns.difference(num_features)
                X_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_test.reset_index(drop=True, inplace=True)
                X_train[cate_features] = X_train[cate_features].astype('category')
                if model_name == "dnn":
                    model_ = Sequential(
                        [
                            Dense(64, activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(16, activation="relu"),
                            Dense(1, activation="sigmoid"),
                        ]
                    )
                    model_.compile(
                        optimizer=Adam(learning_rate=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("normalize", StandardScaler()),
                            ("classifier", model_),
                        ]
                    )
                    model_.fit(
                        X_train,
                        y_train,
                        classifier__batch_size=16,
                        classifier__epochs=15,
                        classifier__verbose=0,
                        classifier__validation_split=0.2,
                    )

                    tmp = model_.predict

                    def new_predict(X):
                        X = model_.named_steps["preprocessor"].transform(X)
                        X = model_.named_steps["normalize"].transform(X)
                        preds_proba = model_.named_steps["classifier"](X)
                        preds_proba = preds_proba.numpy()
                        return (preds_proba > 0.5).astype(int).flatten()
                    def scorer_fn(X, y):
                        return accuracy_score(y, new_predict(X))

                    model_.predict = new_predict
                    model_.predict_proba = tmp
                    model_.score = scorer_fn
                else:

                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("classifier", clone(model)),
                        ]
                    )

                    model_.fit(X_train, y_train)
                feat_to_vary = list(X_train.columns)
                feat_to_vary.remove("Sex")
                target_name = "Status"
                X_train[cate_features] = X_train[cate_features].astype('category')
            elif dataset == "attrition":
                cols = data.columns
                X_train = pd.DataFrame(X_train)
                X_train.columns = cols[:-1]
                X_test = pd.DataFrame(X_test)
                X_test.columns = cols[:-1]
                y_train = pd.Series(y_train, dtype="int32")
                y_test = pd.Series(y_test, dtype="int32")
                num_features = X_train._get_numeric_data().columns.to_list()
                cate_features = X_train.columns.difference(num_features)
                X_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_test.reset_index(drop=True, inplace=True)

                if model_name == "dnn":
                    model_ = Sequential(
                        [
                            Dense(64, activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(16, activation="relu"),
                            Dense(1, activation="sigmoid"),
                        ]
                    )
                    model_.compile(
                        optimizer=Adam(learning_rate=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("normalize", StandardScaler()),
                            ("classifier", model_),
                        ]
                    )
                    model_.fit(
                        X_train,
                        y_train,
                        classifier__batch_size=16,
                        classifier__epochs=15,
                        classifier__verbose=0,
                        classifier__validation_split=0.2,
                    )

                    tmp = model_.predict

                    def new_predict(X):
                        X = model_.named_steps["preprocessor"].transform(X)
                        X = model_.named_steps["normalize"].transform(X)
                        preds_proba = model_.named_steps["classifier"](X)
                        preds_proba = preds_proba.numpy()
                        return (preds_proba > 0.5).astype(int).flatten()
                    def scorer_fn(X, y):
                        return accuracy_score(y, new_predict(X))

                    model_.predict = new_predict
                    model_.predict_proba = tmp
                    model_.score = scorer_fn
                else:

                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                cate_features,
                            )
                        ],
                        remainder="passthrough",
                    )
                    model_ = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("classifier", clone(model)),
                        ]
                    )

                    model_.fit(X_train, y_train)
                feat_to_vary = list(X_train.columns)
                feat_to_vary.remove("Gender")
                target_name = "Attrition"
            
            predictions = model_.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            print(f"\tAccuracy: {accuracy:.2%}")
            affected = X_test[predictions == 0].reset_index(drop=True)
            unaffected = X_test[predictions == 1].reset_index(drop=True)

            train_dataset = X_train.copy()
            for col in num_features:
                train_dataset[col] = train_dataset[col].astype(float)

            train_dataset['target'] = y_train
            model_list.append(model_)
            affected_list.append(affected)
            unaffected_list.append(unaffected)
            train_dataset[cate_features] = train_dataset[cate_features].astype('category')
            train_dataset_list.append(train_dataset)
    return affected_list, unaffected_list, model_list, train_dataset_list, feat_to_vary, target_name,num_features,cate_features


def preprocess_datasets_kfold(
    dataset, model, model_name, cv=5, dataset_folder="datasets/"
):
    if dataset == "germannumeric_credit":
        data = pd.read_csv(
            Path(dataset_folder) / "germannumeric.data",
            header=None,
            delim_whitespace=True,
        )
        data.columns = data.columns.astype(str)
        data[data.columns[-1]] = 2 - data[data.columns[-1]]
    elif dataset == "german_credit":
        data = pd.read_csv(
            Path(dataset_folder) / "german.data", header=None, delim_whitespace=True
        )

        cols = [
            "Existing-Account-Status",
            "Month-Duration",
            "Credit-History",
            "Purpose",
            "Credit-Amount",
            "Savings-Account",
            "Present-Employment",
            "Instalment-Rate",
            "Sex",
            "Guarantors",
            "Residence",
            "Property",
            "Age",
            "Installment",
            "Housing",
            "Existing-Credits",
            "Job",
            "Num-People",
            "Telephone",
            "Foreign-Worker",
            "Status",
        ]

        data.columns = cols
        # Prepocess targets to Bad = 0, Good = 1
        data["Status"] = data["Status"].astype("int32")
        data[data.columns[-1]] = 2 - data[data.columns[-1]]
    elif dataset == "compas":
        data = pd.read_csv(Path(dataset_folder) / "compas.data")
        data = data.dropna(subset=["days_b_screening_arrest"])
        data = data.rename(columns={data.columns[-1]: "status"})
        data = process_compas(data)
        cols = [
            "Sex",
            "Age_Cat",
            "Race",
            "C_Charge_Degree",
            "Priors_Count",
            "Time_Served",
            "Status",
        ]
        data = data[[col.lower() for col in cols]]
        data.columns = cols
        data[data.columns[-1]] = 1 - data[data.columns[-1]]
    elif dataset == "heloc":
        data = pd.read_csv(Path(dataset_folder) / "heloc.data")
        data = data[(data.iloc[:, 1:] >= 0).any(axis=1)]
        data["RiskPerformance"] = data["RiskPerformance"].replace(
            ["Bad", "Good"], [0, 1]
        )
        y = data.pop("RiskPerformance")
        data["RiskPerformance"] = y
        data = data[data >= 0]
        nan_cols = data.isnull().any(axis=0)
        for col in data.columns:
            if nan_cols[col]:
                data[col] = data[col].replace(np.nan, np.nanmedian(data[col]))
    elif dataset == "default_credit":
        data = pd.read_excel(Path(dataset_folder) / "default.data", header=1)
        data["default payment next month"] = data["default payment next month"].replace(
            {0: 1, 1: 0}
        )
        data["SEX"] = data["SEX"].astype(str)
        data["EDUCATION"] = data["EDUCATION"].astype(str)
        data["MARRIAGE"] = data["MARRIAGE"].astype(str)
        df = data.copy()
        df = df.drop(columns=["ID"])
        df = df.reset_index(drop=True)
        df = df.rename(columns={"default payment next month": "target"})

        numerical_columns = [
            col
            for col in df.columns
            if col
            not in [
                "SEX",
                "EDUCATION",
                "MARRIAGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
            ]
        ]

        for col in numerical_columns:
            df[col] = df[col].astype(int)

        for col in df.columns:
            if col not in numerical_columns:
                df[col] = df[col].astype(str)
        cols = [
            "Limit_Bal",
            "Sex",
            "Education",
            "Marriage",
            "Age",
            "Pay_0",
            "Pay_2",
            "Pay_3",
            "Pay_4",
            "Pay_5",
            "Pay_6",
            "Bill_Amt1",
            "Bill_Amt2",
            "Bill_Amt3",
            "Bill_Amt4",
            "Bill_Amt5",
            "Bill_Amt6",
            "Pay_Amt1",
            "Pay_Amt2",
            "Pay_Amt3",
            "Pay_Amt4",
            "Pay_Amt5",
            "Pay_Amt6",
            "Status",
        ]
        data = df
        data = data[data['PAY_5'] != '8']
        data = data[data['PAY_2'] != '8']
        data = data[data['PAY_3'] != '8']
        data = data[data['PAY_3'] != '1']
        data = data[data['PAY_5'] != '8']
        data = data[data['PAY_4'] != '1']
        data = data[data['PAY_4'] != '8']
        data = data[data['PAY_6'] != '8']
    elif dataset == "adult":
        all_columns = ["Age", "Workclass", "Fnlwgt", "Education", "Marital-Status",
                             "Occupation", "Relationship", "Race", "Sex", "Capital-Gain",
                             "Capital-Loss", "Hours-Per-Week", "Native-Country", "Status"]
        cate_columns = ['Workclass', 'Education', 'Marital-Status', 'Occupation',
                                     'Relationship', 'Race', 'Sex', 'Native-Country']
        numerical_columns = [c for c in all_columns if c not in cate_columns + ["Status"]]

        data = pd.read_csv(Path(dataset_folder) / "adult.data", header = None, delim_whitespace = True)
        # remove redundant education num column (education processed in one_hot)
        data = data.drop(4, axis=1)
        # remove rows with missing values: '?,'
        data = data.replace('?,', np.nan); data = data.dropna() 
        data.columns = all_columns
        for col in data.columns[:-1]:
            #print(col)
            if col not in cate_columns:
                data[col] = data[col].apply(lambda x: float(x[:-1]))
            else:
                data[col] = data[col].apply(lambda x: x[:-1])
        # Prepocess Targets to <=50K = 0, >50K = 1
        data[data.columns[-1]] = data[data.columns[-1]].replace(['<=50K', '>50K'],
                                                                [0, 1])

        data = data.reset_index(drop=True)

        for col in numerical_columns:
            data[col] = data[col].astype(int)

        for col in data.columns:
            if col not in numerical_columns and col != data.columns[-1]:
                data[col] = data[col].astype(str)
        data = data[data['Native-Country'] != 'Holand-Netherlands']
    elif dataset == "attrition":
        #target_name = "Attrition"
        cols = ['Age', 'BusinessTravel', 'Department',
           'DistanceFromHome', 'Education', 'EducationField',
           'EnvironmentSatisfaction', 'Gender',
           'JobInvolvement', 'JobLevel', 'JobRole',
           'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
           'PerformanceRating', 'OverTime', 'PercentSalaryHike',
           'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
           'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
           'YearsInCurrentRole', 'YearsSinceLastPromotion',
           'YearsWithCurrManager', 'Attrition']
        cate_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        numerical_columns = [c for c in cols if c not in cate_columns + ['Attrition']]

        data = pd.read_csv(Path(dataset_folder) / "attrition.csv")[cols]
        data['Attrition'] = data['Attrition'].replace(["No", "Yes"], [0, 1])

    affected_list,unaffected_list, model_list, train_dataset_list, feat_to_vary, target_name,num_features,cate_features = (
        load_models_kfold(dataset, data, model, model_name, cv)
    )
    return (
        data,
        affected_list,
        unaffected_list,
        model_list,
        train_dataset_list,
        feat_to_vary,
        target_name,
        num_features, 
        cate_features
    )


def preprocess_datasets(dataset, B, model_name, dataset_folder="datasets/"):
    if dataset == "germannumeric_credit":
        data = pd.read_csv(
            Path(dataset_folder) / "germannumeric.data",
            header=None,
            delim_whitespace=True,
        )
        data.columns = data.columns.astype(str)
        data[data.columns[-1]] = 2 - data[data.columns[-1]]

        X_train, y_train, X_test, y_test = get_split(
            data, normalise=False, shuffle=False
        )
        X_train = pd.DataFrame(X_train, columns=data.columns[:-1])
        X_test = pd.DataFrame(X_test, columns=data.columns[:-1])
        num_features = X_train._get_numeric_data().columns.to_list()
        cate_features = X_train.columns.difference(num_features)
        if model_name == "dnn":
            model = dnn_with_preprocess(B, "german_credit_numeric", X_train, X_test)
        else:

            class IdentityTransformer(BaseEstimator, TransformerMixin):
                def __init__(self):
                    pass

                def fit(self, input_array, y=None):
                    return self

                def transform(self, input_array, y=None):
                    return input_array * 1

            model = Pipeline(
                [("preprocessor", IdentityTransformer()), ("classifier", B)]
            )

            model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", round(accuracy, 2))

        affected = X_test[predictions == 0].reset_index(drop=True)
        unaffected = X_test[predictions == 1].reset_index(drop=True)

        train_dataset = X_train.copy()
        for col in num_features:
            train_dataset[col] = train_dataset[col].astype(float)
        train_dataset["target"] = y_train

        feat_to_vary = list(affected.columns)

    if dataset == "german_credit":
        data = pd.read_csv(
            Path(dataset_folder) / "german.data", header=None, delim_whitespace=True
        )

        cols = [
            "Existing-Account-Status",
            "Month-Duration",
            "Credit-History",
            "Purpose",
            "Credit-Amount",
            "Savings-Account",
            "Present-Employment",
            "Instalment-Rate",
            "Sex",
            "Guarantors",
            "Residence",
            "Property",
            "Age",
            "Installment",
            "Housing",
            "Existing-Credits",
            "Job",
            "Num-People",
            "Telephone",
            "Foreign-Worker",
            "Status",
        ]

        data.columns = cols
        # Prepocess targets to Bad = 0, Good = 1
        data["Status"] = data["Status"].astype("int32")
        data[data.columns[-1]] = 2 - data[data.columns[-1]]

        dtype_dict = {
            "Month-Duration": "int64",
            "Credit-Amount": "int64",
            "Age": "int64",
            "Instalment-Rate": "int64",
            "Residence": "int64",
            "Existing-Credits": "int64",
            "Num-People": "int64",
        }

        X_train, y_train, X_test, y_test = get_split(
            data, normalise=False, shuffle=False
        )
        X_train = pd.DataFrame(X_train, columns=cols[:-1]).astype(dtype_dict)
        X_test = pd.DataFrame(X_test, columns=cols[:-1]).astype(dtype_dict)
        X_test.columns = cols[:-1]

        y_train = pd.Series(y_train, dtype="int32")
        y_test = pd.Series(y_test, dtype="int32")

        num_features = X_train._get_numeric_data().columns.to_list()
        cate_features = X_train.columns.difference(num_features)

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)

        if model_name == "dnn":
            model = dnn_with_preprocess(B, "german", X_train, X_test)

        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        cate_features,
                    )
                ],
                remainder="passthrough",
            )

            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", B),
                ]
            )

            model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", round(accuracy, 2))

        affected = X_test[predictions == 0].reset_index(drop=True)
        unaffected = X_test[predictions == 1].reset_index(drop=True)

        train_dataset = X_train.copy()
        for col in num_features:
            train_dataset[col] = train_dataset[col].astype(float)
        train_dataset["target"] = y_train

        feat_to_vary = list(affected.columns)
        feat_to_vary.remove("Sex")
        feat_to_vary.remove("Foreign-Worker")
        target_name = "Status"
    elif dataset == "compas":
        data = pd.read_csv(Path(dataset_folder) / "compas.data")
        data = data.dropna(subset=["days_b_screening_arrest"])  # drop missing vals
        data = data.rename(columns={data.columns[-1]: "status"})
        data = process_compas(data)
        cols = [
            "Sex",
            "Age_Cat",
            "Race",
            "C_Charge_Degree",
            "Priors_Count",
            "Time_Served",
            "Status",
        ]
        data = data[[col.lower() for col in cols]]
        data.columns = cols
        data[data.columns[-1]] = 1 - data[data.columns[-1]]
        X_train, y_train, X_test, y_test = get_split(
            data, normalise=False, shuffle=False
        )
        X_train = pd.DataFrame(X_train)
        X_train.columns = cols[:-1]
        X_test = pd.DataFrame(X_test)
        X_test.columns = cols[:-1]
        dtype_dict = {"Priors_Count": "int32", "Time_Served": "int32"}
        X_train = X_train.astype(dtype_dict)
        X_test = X_test.astype(dtype_dict)
        y_train = pd.Series(y_train, dtype="int32")
        y_test = pd.Series(y_test, dtype="int32")
        num_features = X_train._get_numeric_data().columns.to_list()
        cate_features = X_train.columns.difference(num_features)
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        if model_name == "dnn":
            model = dnn_with_preprocess(B, "compas", X_train, X_test)
        else:

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        cate_features,
                    )
                ],
                remainder="passthrough",
            )
            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", B),
                ]
            )

            model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", round(accuracy, 2))
        affected = X_test[predictions == 0].reset_index(drop=True)
        unaffected = X_test[predictions == 1].reset_index(drop=True)
        
        train_dataset = X_train.copy()
        for col in num_features:
            train_dataset[col] = train_dataset[col].astype(float)
        train_dataset["target"] = y_train
        feat_to_vary = list(affected.columns)
        feat_to_vary.remove("Sex")
        target_name = "Status"
    elif dataset == "heloc":
        data = pd.read_csv(Path(dataset_folder) / "heloc.data")
        data = data[(data.iloc[:, 1:] >= 0).any(axis=1)]
        # Encode string labels
        data["RiskPerformance"] = data["RiskPerformance"].replace(
            ["Bad", "Good"], [0, 1]
        )
        # Move labels to final column (necessary for self.get_split)
        y = data.pop("RiskPerformance")
        data["RiskPerformance"] = y
        # Convert negative values to NaN
        data = data[data >= 0]
        # Replace NaN values with median
        nan_cols = data.isnull().any(axis=0)
        for col in data.columns:
            if nan_cols[col]:
                data[col] = data[col].replace(np.nan, np.nanmedian(data[col]))
        X_train, y_train, X_test, y_test = get_split(
            data, normalise=False, shuffle=False
        )
        X_train = pd.DataFrame(X_train)
        X_train.columns = data.columns[:-1]
        X_test = pd.DataFrame(X_test)
        X_test.columns = data.columns[:-1]
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train = pd.Series(y_train, dtype="int32")
        y_test = pd.Series(y_test, dtype="int32")
        num_features = X_train._get_numeric_data().columns.to_list()
        cate_features = X_train.columns.difference(num_features)
        if model_name == "dnn":
            model = dnn_with_preprocess(B, "heloc", X_train, X_test)
        else:

            class IdentityTransformer(BaseEstimator, TransformerMixin):
                def __init__(self):
                    pass

                def fit(self, input_array, y=None):
                    return self

                def transform(self, input_array, y=None):
                    return input_array * 1

            model = Pipeline(
                [("preprocessor", IdentityTransformer()), ("classifier", B)]
            )

            model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", round(accuracy, 2))
        affected = X_test[predictions == 0].reset_index(drop=True)
        unaffected = X_test[predictions == 1].reset_index(drop=True)

        train_dataset = X_train.copy()
        for col in num_features:
            train_dataset[col] = train_dataset[col].astype(float)
        train_dataset["target"] = y_train
        feat_to_vary = list(affected.columns)
        target_name = "RiskPerformance"
    elif dataset == "default_credit":
        data = pd.read_excel(Path(dataset_folder) / "default.data", header=1)
        data["default payment next month"] = data["default payment next month"].replace(
            {0: 1, 1: 0}
        )
        data["SEX"] = data["SEX"].astype(str)
        data["EDUCATION"] = data["EDUCATION"].astype(str)
        data["MARRIAGE"] = data["MARRIAGE"].astype(str)
        df = data.copy()
        df = df.drop(columns=["ID"])
        df = df.reset_index(drop=True)
        df = df.rename(columns={"default payment next month": "target"})

        numerical_columns = [
            col
            for col in df.columns
            if col
            not in [
                "SEX",
                "EDUCATION",
                "MARRIAGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
            ]
        ]

        for col in numerical_columns:
            df[col] = df[col].astype(int)

        for col in df.columns:
            if col not in numerical_columns:
                df[col] = df[col].astype(str)
        cols = [
            "Limit_Bal",
            "Sex",
            "Education",
            "Marriage",
            "Age",
            "Pay_0",
            "Pay_2",
            "Pay_3",
            "Pay_4",
            "Pay_5",
            "Pay_6",
            "Bill_Amt1",
            "Bill_Amt2",
            "Bill_Amt3",
            "Bill_Amt4",
            "Bill_Amt5",
            "Bill_Amt6",
            "Pay_Amt1",
            "Pay_Amt2",
            "Pay_Amt3",
            "Pay_Amt4",
            "Pay_Amt5",
            "Pay_Amt6",
            "Status",
        ]
        X_train, y_train, X_test, y_test = get_split(df, normalise=False, shuffle=False)
        X_train = pd.DataFrame(X_train)
        X_train.columns = df.columns[:-1]
        X_test = pd.DataFrame(X_test)
        X_test.columns = df.columns[:-1]
        X_train = X_train.astype(dict(df.drop(columns="target").dtypes))

        X_test = X_test.astype(dict(df.drop(columns="target").dtypes))
        y_train = pd.Series(y_train, dtype="int32")
        y_test = pd.Series(y_test, dtype="int32")
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)

        num_features = X_train._get_numeric_data().columns.to_list()
        cate_features = X_train.columns.difference(num_features)

        if model_name == "dnn":
            model = dnn_with_preprocess(B, "default", X_train, X_test)
        elif model_name == "lr":

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        cate_features,
                    ),
                    (
                        "num",
                        StandardScaler(),
                        num_features,
                    ),
                ],
                remainder="passthrough",
            )
            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", B),
                ]
            )

            model.fit(X_train, y_train)
        else:

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        cate_features,
                    )
                ],
                remainder="passthrough",
            )
            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", B),
                ]
            )

            model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", round(accuracy, 2))
        affected = X_test[predictions == 0].reset_index(drop=True)
        unaffected = X_test[predictions == 1].reset_index(drop=True)

        train_dataset = X_train.copy()
        for col in num_features:
            train_dataset[col] = train_dataset[col].astype(float)
        train_dataset["target"] = y_train
        feat_to_vary = list(affected.columns)
        feat_to_vary.remove("SEX")
        data = df
        target_name = "target"

    return (
        train_dataset,
        data,
        X_train,
        y_train,
        X_test,
        y_test,
        affected,
        unaffected,
        model,
        feat_to_vary,
        target_name,
    )
