import lightgbm as gbm
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from product_classification.training import (evaluate_gbm,
                                             find_best_hyper_param_gbm,
                                             train_gbm)


def test_train_gbm():
    col_names = [f"f{i}" for i in range(10)]
    parameters = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "metric": "multi_logloss",
        "learning_rate": 0.001,
        "max_depth": 7,
        "num_leaves": 17,
        "n_estimators": 200,
    }

    x, y = make_classification(
        n_features=10, n_informative=7, n_redundant=3, n_samples=5000, n_classes=5
    )

    data = pd.DataFrame(x, columns=col_names)
    data["target"] = y
    train, val = train_test_split(data, test_size=0.2)

    classifier = train_gbm(
        train=train,
        val=val,
        feature_names=col_names,
        target_name="target",
        params=parameters,
    )

    assert isinstance(classifier, gbm.LGBMClassifier)


def test_evaluate_gbm():
    col_names = [f"f{i}" for i in range(10)]
    parameters = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "metric": "multi_logloss",
        "learning_rate": 0.01,
        "max_depth": 10,
        "num_leaves": 30,
        "n_estimators": 1000,
    }

    x, y = make_classification(
        n_features=10,
        n_informative=5,
        n_redundant=5,
        n_samples=5000,
        n_classes=5,
        flip_y=0.0,
        n_clusters_per_class=1,
    )

    data = pd.DataFrame(x, columns=col_names)
    data["target"] = y

    train, val = train_test_split(data, test_size=0.2, stratify=y)

    classifier = train_gbm(
        train=train,
        val=val,
        feature_names=col_names,
        target_name="target",
        params=parameters,
    )

    metrics = evaluate_gbm(
        val=val, feature_names=col_names, target_name="target", classifier=classifier
    )

    assert metrics["accuracy"] > 0.8
    assert metrics["f1_macro"] > 0.8
    assert len(metrics["per_class_f1"]) == 5


def test_find_best_hyper_param_gbm():
    col_names = [f"f{i}" for i in range(10)]

    x, y = make_classification(
        n_features=10,
        n_informative=5,
        n_redundant=5,
        n_samples=5000,
        n_classes=5,
        flip_y=0.0,
        n_clusters_per_class=1,
    )

    data = pd.DataFrame(x, columns=col_names)
    data["target"] = y
    train, val = train_test_split(data, test_size=0.2)

    study = find_best_hyper_param_gbm(
        train=train, val=val, feature_names=col_names, target_name="target", n_trials=10
    )

    assert study.best_value > 0.8
