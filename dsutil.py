# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import gzip
import os
import struct
import urllib.request
from os import path

import numpy as np
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from feature_engine.encoding import WoEEncoder
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.pipeline import Pipeline
from category_encoders.woe import WOEEncoder
from ucimlrepo import fetch_ucirepo

def set_proxy():
    os.environ["http_proxy"] = "http://192.168.117.199:7890"
    os.environ["https_proxy"] = "http://192.168.117.199:7890"
    
def unset_proxy():
    if "http_proxy" in os.environ:
        del os.environ["http_proxy"]
    if "https_proxy" in os.environ:
        del os.environ["https_proxy"]

def fill_nan(df):
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                # 数值列用中位数填充
                median = df[col].median()
                df[col] = df[col].fillna(median)
            else:
                # 类别列用'unknown'填充
                df[col] = df[col].fillna('unknown')
    return df

def gcd_dataset_woe():
    set_proxy()
    data = fetch_ucirepo(id=144)
    unset_proxy()
    X = data.data.features.copy()
    y = data.data.targets.squeeze()
    
    # Convert target to binary 0/1
    y = y.map({1: 0, 2: 1})

    # Identify categorical and numerical features
    categorical_vars = X.select_dtypes(include='object').columns.tolist()
    numerical_vars = X.select_dtypes(include='number').columns.tolist()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Build pipeline
    pipe = Pipeline([
        ("cat_imputer", CategoricalImputer(imputation_method='missing', variables=categorical_vars)),
        ("num_imputer", MeanMedianImputer(imputation_method='median', variables=numerical_vars)),
        ("discretiser", EqualFrequencyDiscretiser(q=5, variables=numerical_vars, return_object=True)),
        ("woe_encoder", WOEEncoder(cols=categorical_vars + numerical_vars, regularization=0.01))
    ])

    # Fit pipeline
    X_train = pipe.fit_transform(X_train, y_train)
    X_test = pipe.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def gcd_dataset_normal():
    set_proxy()
    data = fetch_ucirepo(id=144)
    unset_proxy()
    X = data.data.features.copy()
    y = data.data.targets.squeeze()
    
    # Convert target to binary 0/1
    y = y.map({1: 0, 2: 1})

    # Separate categorical and numerical columns
    categorical = X.select_dtypes(include='object').columns.tolist()
    numerical = X.select_dtypes(exclude='object').columns.tolist()

    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids dummy trap
    X_cat = encoder.fit_transform(X[categorical])

    # Standardize numerical features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[numerical])

    # Combine features
    X_all = np.hstack([X_num, X_cat])

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def hcdr_dataset_woe():
    df = pd.read_csv("./hcdr_application_train.csv") 
    df_clean = fill_nan(df)
    y = df_clean["TARGET"]
    X = df_clean.drop(columns=["TARGET", "SK_ID_CURR"])
    categorical_vars = X.select_dtypes(include="object").columns.tolist()

    categorical_vars += [c for c in X.select_dtypes(include="int64").columns if X[c].nunique() < 10]
    categorical_vars = list(set(categorical_vars))

    numerical_vars = [c for c in X.columns if c not in categorical_vars]
    X[categorical_vars] = X[categorical_vars].astype("category")

    single_vars = [c for c in categorical_vars if X[c].nunique() <= 1]
    X = X.drop(columns=single_vars)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    pipe_woe = Pipeline([
        ("cat_imputer", CategoricalImputer(imputation_method='missing', variables=categorical_vars)),
        ("num_imputer", MeanMedianImputer(imputation_method='median', variables=numerical_vars)),
        ("discretiser", EqualFrequencyDiscretiser(q=10, variables=numerical_vars, return_object=True)),
        # ("woe_encoder", WoEEncoder(variables=categorical_vars + numerical_vars, ignore_format=True)),
        ("woe_encoder", WOEEncoder(cols=categorical_vars + numerical_vars, regularization=0.1))
    ])
    
    X_train = pipe_woe.fit_transform(X_train, y_train)
    X_test = pipe_woe.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def hcdr_dataset_normal():
    df = pd.read_csv("./hcdr_application_train.csv") 
    df_clean = fill_nan(df)
    y = df_clean["TARGET"]
    X = df_clean.drop(columns=["TARGET", "SK_ID_CURR"])
    categorical_vars = X.select_dtypes(include="object").columns.tolist()

    categorical_vars += [c for c in X.select_dtypes(include="int64").columns if X[c].nunique() < 10]
    categorical_vars = list(set(categorical_vars))

    numerical_vars = [c for c in X.columns if c not in categorical_vars]
    X[categorical_vars] = X[categorical_vars].astype("category")

    single_vars = [c for c in categorical_vars if X[c].nunique() <= 1]
    X = X.drop(columns=single_vars)
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = pd.DataFrame(ohe.fit_transform(X[categorical_vars]), 
                        columns=ohe.get_feature_names_out(categorical_vars),
                        index=X.index)
    X_num = X[numerical_vars]
    
    X_full = pd.concat([X_num, X_cat], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=42, stratify=y)
    
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    
    return X_train, X_test, y_train, y_test


# def hcdr_dataset_naive():
#     df = pd.read_csv("./hcdr_application_train.csv") 
#     df_clean = fill_nan(df)
#     y = df_clean["TARGET"]
#     X = df_clean.drop(columns=["TARGET", "SK_ID_CURR"])
#     categorical_vars = X.select_dtypes(include="object").columns.tolist()

#     categorical_vars += [c for c in X.select_dtypes(include="int64").columns if X[c].nunique() < 10]
#     categorical_vars = list(set(categorical_vars))

#     numerical_vars = [c for c in X.columns if c not in categorical_vars]
#     X[categorical_vars] = X[categorical_vars].astype("category")

#     single_vars = [c for c in categorical_vars if X[c].nunique() <= 1]
#     X = X.drop(columns=single_vars)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
#     print(X_train)
#     print(X_test)
#     print(y_train)
#     print(y_test)
    
#     return X_train, X_test, y_train, y_test