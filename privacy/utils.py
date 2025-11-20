import os
import random
from itertools import chain, combinations

from sklearn import linear_model
from sklearn.metrics import roc_curve, balanced_accuracy_score
from diffprivlib import models as dp_models
from scipy.interpolate import interp1d

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ---------
# LR Models
#----------

def create_lr_model(l1, C):
    model_class = linear_model.LogisticRegression
    param_dict = {
        'solver': 'saga',
        'penalty': 'elasticnet',
        'max_iter': 100,
        'l1_ratio': l1, # Should be 1 (passed in args)
        'class_weight': 'balanced',
        'C': C, # Should be 0.1 (passed in args)
    }
    return model_class, param_dict


def create_lr_dp_model(epsilon):
    model_class = dp_models.LogisticRegression
    param_dict = {
        'max_iter': 100,
        'epsilon': epsilon,
    }
    return model_class, param_dict


# ---------
# NN Models
#----------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def create_nn_model():
    """
    Standard non-private MLP with fixed architecture.
    """
    model_class = SimpleMLP
    param_dict = {
            'max_iter': 100,
            'hidden_layer_sizes': (19, 19),
            'lr': 1e-3,
            'weight_decay': 1e-05
        }
    return model_class, param_dict


# -----------------
# Utility functions
# -----------------

def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    else:
        return obj


def all_combinations(lst):
    return list(chain.from_iterable(combinations(lst, r)
                                    for r in range(1, len(lst) + 1)))


def load_data(cwd, in_features, out_feature, datasets, datasets_ids):
    data_file = os.path.join(cwd, 'AllData.xlsx')

    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25

    dfs = []
    for ds, dsid in zip(datasets, datasets_ids):
        df = pd.read_excel(data_file, sheet_name=ds, index_col=0)
        df['Dataset'] = ds
        df['DatasetNum'] = dsid
        
        # Data truncation
        df['TMB'] = [c if c < TMB_upper else TMB_upper for c in df['TMB']]
        df['Age'] = [c if c < Age_upper else Age_upper for c in df['Age']]
        df['NLR'] = [c if c < NLR_upper else NLR_upper for c in df['NLR']]
        
        dfs.append(df)

    data_all_raw = pd.concat(dfs, axis=0)
    
    all_features = in_features + [out_feature, 'Dataset', 'DatasetNum']
    data_no_nans = data_all_raw[all_features].dropna(axis=0)
    
    return data_no_nans


def balanced_accuracy(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    best_threshold = thresholds[np.argmax(youden)]
    
    y_pred = (y_proba >= best_threshold).astype(int)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return bacc, y_pred


def discretize(vector, n_bins=10):
    # Bin width
    step = 1.0 / n_bins  

    # Compute bin index
    bin_idx = torch.floor(vector / step)  # 0 ... n_bins-1

    # Lower & upper edges of the bin
    bin_low  = bin_idx * step
    bin_high = bin_low + step

    # Apply your rule:
    #   if prob <= 0.5 → snap to lower edge
    #   if prob > 0.5  → snap to upper edge
    result = torch.where(vector <= 0.5, bin_low, bin_high)

    # Clip to [0,1]
    result = torch.clamp(result, 0.0, 1.0)

    return result


def classify_by_dataset(models, params, datasets, verbose=True):
    # Multilabel: average predictions and apply threshold
    all_preds = np.array([model.predict_proba(params) for model in models])
    avg_preds = np.mean(all_preds, axis=0)
    
    if verbose:
        for i in range(len(datasets)):
            print(f'{datasets[i]:>16}: {avg_preds[0][i]:.4f}')
        
    return avg_preds[0]


@torch.no_grad()
def response_confidence(model, x_train, y_train,
                        bin_size=0.1, bs_number=1000):
    """
    Compute the bootstrapped response probability curve with confidence intervals.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    x_train : torch.Tensor
        Input features.
    y_train : torch.Tensor
        True binary outcomes (0/1).
    bin_size : float
        Width of the bin for grouping predictions.
    bs_number : int
        Number of bootstrap resamples.

    Returns
    -------
    score_list : np.ndarray
        The x-axis (prediction score thresholds).
    ORR_mean : np.ndarray
        Mean response probability for each score bin.
    ORR_05 : np.ndarray
        5% quantile (lower bound of CI).
    ORR_95 : np.ndarray
        95% quantile (upper bound of CI).
    """

    # Predictions
    result = model.predict_proba(x_train)
    y_pred = result[:, 1]#.numpy()
    y_true = y_train.numpy()

    sampleNUM = len(y_true)
    score_list = np.arange(0.0, 1.01, 0.01)
    num_scores = len(score_list)

    # ORR storage
    ORR_list = [[] for _ in range(num_scores)]
    ORR_valid = [False for _ in range(num_scores)]

    # Bootstrapping
    for _ in range(bs_number):
        idx_resampled = random.choices(range(sampleNUM), k=sampleNUM)
        aux_y_true = y_true[idx_resampled]
        aux_y_pred = y_pred[idx_resampled]

        for i, score in enumerate(score_list):
            bin_mask = (aux_y_pred > score - bin_size / 2) & \
                       (aux_y_pred <= score + bin_size / 2)

            if bin_mask.sum() > 0:
                ORR_list[i].append(aux_y_true[bin_mask].mean())
                ORR_valid[i] = True
            else:
                ORR_list[i].append(np.nan)

    # Compute statistics, skipping NaNs
    ORR_mean = np.array([np.nanmean(x) if ORR_valid[i] else np.nan
                         for i, x in enumerate(ORR_list)])
    ORR_05 = np.array([np.nanquantile(x, 0.05) if ORR_valid[i] else np.nan
                       for i, x in enumerate(ORR_list)])
    ORR_95 = np.array([np.nanquantile(x, 0.95) if ORR_valid[i] else np.nan
                       for i, x in enumerate(ORR_list)])

    # Forward-fill missing values
    def forward_fill(arr):
        filled = []
        last_val = np.nan
        for val in arr:
            if not np.isnan(val):
                last_val = val
            filled.append(last_val)
        return np.array(filled)

    ORR_mean = forward_fill(ORR_mean)
    ORR_05 = forward_fill(ORR_05)
    ORR_95 = forward_fill(ORR_95)

    return score_list, ORR_mean, ORR_05, ORR_95


@torch.no_grad()
def response_confidence_inverse(score_list, ORR_mean, mean_value,
                                bin_size=0.1, bs_number=1000):
    """
    Given a target mean response probability, return the corresponding score
    (decision threshold) using the bootstrapped response curve.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    x_train : torch.Tensor
        Input features.
    y_train : torch.Tensor
        True binary outcomes (0/1).
    mean_value : float
        Desired mean response probability (between 0 and 1).
    bin_size : float
        Width of the bin for grouping predictions.
    bs_number : int
        Number of bootstrap resamples.

    Returns
    -------
    score : float
        The score (threshold) corresponding to the given mean response.
    """

    # Remove NaNs for interpolation
    valid_mask = ~np.isnan(ORR_mean)
    valid_scores = score_list[valid_mask]
    valid_means = ORR_mean[valid_mask]

    if len(valid_scores) == 0:
        raise ValueError("No valid bins found for interpolation.")

    # Build inverse interpolation function: mean -> score
    inv_func = interp1d(valid_means, valid_scores,
                        bounds_error=False,
                        fill_value=(valid_scores[0], valid_scores[-1]))

    # Get the score for the desired mean
    score = float(inv_func(mean_value))
    return score


def get_lr_param(data1, data2, response1, response2, score_list, ORR_mean):
    param_diff = (data2 - data1).sum()
    
    response1 = response1 / 100.
    response2 = response2 / 100.
    
    # print(f'{response1:.2f}, {response2:.2f}')
    
    response1 = torch.tensor(
        response_confidence_inverse(score_list, ORR_mean, response1))
    response2 = torch.tensor(
        response_confidence_inverse(score_list, ORR_mean, response2))
    
    # print(f'{response1:.2f}, {response2:.2f}')
    # print((response2 - response1) / param_diff)
    
    logit1 = (response1 / (1 - response1)).log()
    logit2 = (response2 / (1 - response2)).log()
    param = (logit2 - logit1) / param_diff
    
    return param


def get_lr_param_from_model(data1, data2, model):
    param_diff = (data2 - data1).sum()
    
    response1 = model(data1.unsqueeze(0)).squeeze(0)
    response2 = model(data2.unsqueeze(0)).squeeze(0)
    
    if response1 == 1.:
        response1 = torch.tensor(0.99)
    if response2 == 1.:
        response2 = torch.tensor(0.99)
    
    # print(f'{float(response1):.2f}, {float(response2):.2f}')
    # print((response2 - response1) / param_diff)
    
    logit1 = (response1 / (1 - response1)).log()
    logit2 = (response2 / (1 - response2)).log()
    param = (logit2 - logit1) / param_diff
    
    return param
