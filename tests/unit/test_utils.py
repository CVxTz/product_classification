from product_classification.utils import compute_metrics


def test_compute_metrics():

    y_true = [0, 1, 1, 2, 2, 2]
    y_predicted = [0, 1, 2, 2, 2, 2]

    metrics = compute_metrics(y_true, y_predicted)

    assert metrics["f1_macro"] == 0.8412698412698413
    assert metrics["f1_weighted"] == 0.8174603174603173
    assert metrics["accuracy"] == 0.8333333333333334
    assert metrics["per_class_f1"] == [1.0, 0.6666666666666666, 0.8571428571428571]
