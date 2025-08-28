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


def load_sketch_data(cwd, in_features, out_feature, datasets, datasets_ids, n_samples=10):
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

    # Identify cancer type columns
    cancer_types = [f for f in in_features if f.startswith("CancerType")]

    selected_rows = []

    for ctype in cancer_types:
        subset = data_no_nans[data_no_nans[ctype] == 1]

        for resp in [0, 1]:
            candidates = subset[subset[out_feature] == resp]

            if len(candidates) > 0:
                take_n = min(n_samples, len(candidates))
                sampled = candidates.sample(n=take_n, random_state=42)
                selected_rows.append(sampled)

    # Combine everything (raw values)
    sketch_data = pd.concat(selected_rows, axis=0)

    return sketch_data


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


def create_dp_model(epsilon):
    model_type = dp_models.LogisticRegression
    param_dict = {
        'max_iter': 100,
        'epsilon': epsilon,
    }
    return model_type, param_dict
