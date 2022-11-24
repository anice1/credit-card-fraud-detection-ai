#!/usr/bin/env python3

import os
import toml
import pickle
import warnings
import pandas as pd
import seaborn as sns
import neptune.new as neptune
import matplotlib.pyplot as plt
from toolkit import train, log_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

# load featurization params
env = toml.load("env.toml")


# Import feature data
X_train = pd.read_csv("data/features/X_train.csv", sep=",")
X_test = pd.read_csv("data/features/X_test.csv", sep=",")

y_train = pd.read_csv("data/features/y_train.csv", sep=",")
y_test = pd.read_csv("data/features/y_test.csv", sep=",")


# initialize neptune credentials
run = neptune.init_run(
    project=env["tracking"]["PROJECT_NAME"],
    api_token=env["tracking"]["NEPTUNE_API_TOKEN"],
    tags=["CCFD (Logistic Regression)"],
    capture_hardware_metrics=False,
    source_files=["model.py, featurize.py, train.py, env.toml"],
)

# set model params
log_model = LogisticRegression()
run["model_params"] = log_model.get_params()

# set model params
# rf_parameters = {
#     "n_estimators": env["tracking"]["N_ESTIMATORS"],
#     "max_features": env["tracking"]["MAX_FEATURES"],
#     "max_samples": env["tracking"]["MAX_SAMPLES"],
#     "max_depth": env["tracking"]["MAX_DEPTH"],
#     "min_samples_leaf": env["tracking"]["MIN_SAMPLES_LEAF"],
#     "min_samples_split": env["tracking"]["MIN_SAMPLES_SPLIT"],
# }
# run["model/rf_parameters"] = rf_parameters
# rforest_model = RandomForestClassifier(**rf_parameters)
# run["model_params"] = rforest_model.get_params()

skfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)

# Train the model
training = train(
    X_train=X_train, y_train=y_train, model=log_model, skfold=skfold, logger=run
)
trained_model = training["model"]

# Log test metrics
log_metrics(
    model=trained_model,
    features=X_test,
    labels=y_test,
    filename="eval_conf_matrix",
    logger=run,
    log_type="eval",
)

# log training metrics
log_metrics(
    model=trained_model,
    features=X_train,
    labels=y_train,
    filename="train_conf_matrix",
    logger=run,
    log_type="train",
)

# Export trained model
os.makedirs("models", exist_ok=True)
pickle.dump(trained_model, open("models/model.pkl", "wb"))
run["models"].upload("models/model.pkl")
run["scaled"] = env["featurize"]["SCALE"]

# Track training data
run["train_dataset/X_train"].track_files("data/features/X_train.csv")
run["train_dataset/y_train"].track_files("data/features/y_train.csv")
run["train_dataset/y_train"].track_files("notebooks/analysis.ipynb")

run.stop()
