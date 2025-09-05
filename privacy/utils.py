import os
from itertools import chain, combinations

from sklearn import linear_model
from diffprivlib import models as dp_models

import pandas as pd
import torch


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


def create_lr_model(l1, C):
    model_type = linear_model.LogisticRegression
    param_dict = {
        'solver': 'saga',
        'penalty': 'elasticnet',
        'max_iter': 100,
        'l1_ratio': l1, # Should be 1 (passed in args)
        'class_weight': 'balanced',
        'C': C, # Should be 0.1 (passed in args)
    }
    return model_type, param_dict


def create_lr_dp_model(epsilon):
    model_type = dp_models.LogisticRegression
    param_dict = {
        'max_iter': 100,
        'epsilon': epsilon,
    }
    return model_type, param_dict


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
