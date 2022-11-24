from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def train(X_train, y_train, model, skfold, logger):

    fold_no = 1
    recall_scores = []
    f1_scores = []
    acc_scores = []

    for train_index, val_index in skfold.split(X_train, y_train):

        train_X, val_X = X_train.iloc[train_index], X_train.iloc[val_index]
        train_y, val_y = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(train_X, train_y)
        y_pred = model.predict(val_X)

        re_score = recall_score(val_y, y_pred)
        f1 = f1_score(val_y, y_pred)
        acc_score = accuracy_score(val_y, y_pred)

        recall_scores.append(re_score)
        acc_scores.append(acc_score)
        f1_scores.append(f1)

        # Log metrics (series of values)
        logger["train/recall_score"].log(re_score)
        logger["train/f1_score"].log(f1)
        logger["train/accuracy"].log(acc_score)

        print(
            f"Fold {fold_no}| Recall: {recall_scores[fold_no - 1]} |-| F1: {f1_scores[fold_no - 1]} ACC: {acc_scores[fold_no-1]}",
            end="\n",
        )

        fold_no += 1

    re_score, f1, acc_scores = (
        np.mean(recall_scores),
        np.mean(f1_scores),
        np.mean(acc_scores),
    )

    plt.savefig("resources/plots/train_conf_matrix.png")
    logger["train/conf_matrix"].upload("resources/plots/train_conf_matrix.png")
    plt.figure()

    return {"model": model, "recall": re_score, "f1_score": f1, "accuracy": acc_scores}


def log_metrics(model, features, labels, filename, logger, log_type="eval"):
    y_pred = model.predict(features)

    re_score = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    acc_score = accuracy_score(labels, y_pred)

    # Log metrics (series of values)
    logger[f"{log_type}/recall_score"].log(re_score)
    logger[f"{log_type}/f1_score"].log(f1)
    logger[f"{log_type}/accuracy"].log(acc_score)

    conf_matrix = confusion_matrix(labels, y_pred)
    sns.heatmap(
        conf_matrix, annot=True, cmap="YlGnBu", fmt=".2f",
    )

    # You can upload image from the disc
    plt.savefig(f"resources/plots/{filename}")
    logger[f"{log_type}/conf_matrix"].upload(f"resources/plots/{filename}.png")
    plt.figure()
