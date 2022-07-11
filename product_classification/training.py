from typing import Dict, List

import lightgbm as gbm
import optuna
import pandas as pd

from product_classification.utils import compute_metrics


def train_gbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_names: List[str],
    target_name: str,
    params: Dict,
    log=False,
):
    """
    Trains gradient boosting model

    :param train:
    :param val:
    :param feature_names:
    :param target_name:
    :param params:
    :param log:
    :return:
    """

    classifier = gbm.LGBMClassifier(**params)
    callbacks = [gbm.early_stopping(stopping_rounds=50, first_metric_only=True)]

    if log:
        callbacks.append(gbm.log_evaluation(period=10))

    classifier.fit(
        train[feature_names],
        train[target_name],
        eval_set=[(val[feature_names], val[target_name])],
        callbacks=callbacks,
    )

    return classifier


def evaluate_gbm(
    val: pd.DataFrame,
    feature_names: List[str],
    target_name: str,
    classifier: gbm.LGBMClassifier,
):
    """
    Evaluate classifier on dataframe

    :param val:
    :param feature_names:
    :param target_name:
    :param classifier:
    :return:
    """

    predictions = classifier.predict(val[feature_names])
    metrics = compute_metrics(val[target_name], predictions)

    return metrics


def find_best_hyper_param_gbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_names: List[str],
    target_name: str,
    n_trials=100,
):
    """
    Finds the best hyper-parameters to maximize f1_macro using Optuna

    :param train:
    :param val:
    :param feature_names:
    :param target_name:
    :param n_trials:
    :return:
    """
    def objective(trial):

        params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "metric": "multi_logloss",
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.5),
            "subsample": trial.suggest_float("subsample", 0.1, 0.4),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.01, log=True),
            "n_estimators": 2000,
        }

        classifier = gbm.LGBMClassifier(**params)
        callbacks = [gbm.early_stopping(stopping_rounds=10, first_metric_only=True)]
        classifier.fit(
            train[feature_names],
            train[target_name],
            eval_set=[(val[feature_names], val[target_name])],
            callbacks=callbacks,
        )
        metrics = evaluate_gbm(
            val=val,
            feature_names=feature_names,
            target_name=target_name,
            classifier=classifier,
        )
        return metrics["f1_macro"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study
