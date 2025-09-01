import torch
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error, accuracy_score
import numpy as np


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts_np = predicts.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels_np, y_score=predicts_np)
    roc_auc = roc_auc_score(y_true=labels_np, y_score=predicts_np)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts_np = predicts.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels_np, y_score=predicts_np)

    return {'roc_auc': roc_auc}

def get_link_regression_metrics(predicts: torch.Tensor, true_value: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts_np = predicts.cpu().detach().numpy()
    true_value_np = true_value.cpu().numpy()

    mse = mean_squared_error(true_value_np, predicts_np)
    mae = mean_absolute_error(true_value_np, predicts_np)

    predicts_rounded = np.round(predicts_np * 2)
    true_value_rounded = np.round(true_value_np * 2)

    accuracy = (predicts_rounded == true_value_rounded).mean()

    return {'mse': mse, 'mae': mae, 'accuracy': accuracy}
