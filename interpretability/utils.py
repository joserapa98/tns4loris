import os
import copy
import random

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss


cwd = os.getcwd()


#--------------------- Load and scale data --------------------

def load_data(cwd, in_features, out_feature, datasets, scaler_type):
    data_file = os.path.join(cwd, 'loris', '02.Input', 'AllData.xlsx')

    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    
    dfs = []
    for ds in datasets:
        df = pd.read_excel(data_file, sheet_name=ds, index_col=0)
        
        # Data truncation
        df['TMB'] = [c if c < TMB_upper else TMB_upper for c in df['TMB']]
        df['Age'] = [c if c < Age_upper else Age_upper for c in df['Age']]
        df['NLR'] = [c if c < NLR_upper else NLR_upper for c in df['NLR']]
        
        dfs.append(df)

    data = pd.concat(dfs, axis=0)
    
    all_features = in_features + [out_feature]
    data_no_nans = data[all_features].dropna(axis=0)
    
    data_scaled = copy.deepcopy(data_no_nans)

    if scaler_type == 'standard':
        scaler_class = StandardScaler
    elif scaler_type == 'minmax':
        scaler_class = MinMaxScaler
    else:
        raise ValueError(f'Unrecognized scaler type of {scaler_type}. '
                         'Only "standard" and "minmax" are accepted.')
    
    scalers_dict = {}
    for feature in in_features:
        scaler = scaler_class()
        data_scaled[feature] = scaler.fit_transform(data_no_nans[[feature]])
        scalers_dict[feature] = scaler
    
    return data_scaled, scalers_dict


def load_sketch_data(cwd, in_features, out_feature, datasets, scaler_type, n_samples=1):
    """
    Loads dataset, fits scalers on full cleaned data, and returns a balanced sketch 
    with up to `n_samples` per (CancerType, Response) combination, scaled at the end.

    Parameters
    ----------
    cwd : str
        Path to working directory.
    in_features : list of str
        Input features, including cancer type dummies.
    out_feature : str
        Target feature name (Response).
    datasets : list of str
        Excel sheet names to load.
    n_samples : int
        Number of samples per combination (CancerType, Response).
    scaler_type : str
        "Standard" or "MinMax".

    Returns
    -------
    sketch_data_scaled : pd.DataFrame
        Sketch dataset (scaled).
    scalers_dict : dict
        Dict of fitted scalers per feature.
    """
    
    data_file = os.path.join(cwd, 'loris', '02.Input', 'AllData.xlsx')
    
    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    
    # Load and concatenate datasets
    dfs = []
    for ds in datasets:
        df = pd.read_excel(data_file, sheet_name=ds, index_col=0)
        
        # Data truncation
        df['TMB'] = [c if c < TMB_upper else TMB_upper for c in df['TMB']]
        df['Age'] = [c if c < Age_upper else Age_upper for c in df['Age']]
        df['NLR'] = [c if c < NLR_upper else NLR_upper for c in df['NLR']]
        
        dfs.append(df)
    
    data_all_raw = pd.concat(dfs, axis=0)

    # Drop NaNs in relevant features
    all_features = in_features + [out_feature]
    data_no_nans = data_all_raw[all_features].dropna(axis=0)

    # Choose scaler class
    if scaler_type == 'standard':
        scaler_class = StandardScaler
    elif scaler_type == 'minmax':
        scaler_class = MinMaxScaler
    else:
        raise ValueError(f'Unrecognized scaler type: {scaler_type}. Use ' 
                         '"standard" or "minmax".')
    
    # Fit scalers on full clean data (not applied yet)
    scalers_dict = {}
    for feature in in_features:
        scaler = scaler_class()
        scaler.fit(data_no_nans[[feature]])
        scalers_dict[feature] = scaler

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

    # Apply scalers to sketch_data
    sketch_data_scaled = sketch_data.copy()
    for feature in in_features:
        sketch_data_scaled[feature] = scalers_dict[feature].transform(sketch_data[[feature]])

    return sketch_data_scaled, scalers_dict


def scale_input(patient, features, scalers_dict):
    """
    Takes in a list of features and their names and scales each one using
    the corresponding scaler fit to Chowell train data.
    """
    patient_df = pd.DataFrame([patient], columns=features)
    
    for feature in patient_df.columns:
        patient_df[feature] = scalers_dict[feature].transform(patient_df[[feature]])
    
    patient_list = patient_df.iloc[0].tolist()
    return patient_list


def rescale_lr_models(model, scalers_dict, in_features):
    # Save model's parameters
    coefs = torch.from_numpy(model.coef_).flatten()
    intercept = torch.from_numpy(model.intercept_).flatten()
    
    means = [torch.from_numpy(scaler.mean_)
             for scaler in scalers_dict.values()]
    means = torch.cat(means, dim=0)
    
    scales = [torch.from_numpy(scaler.scale_)
              for scaler in scalers_dict.values()]
    scales = torch.cat(scales, dim=0)
    
    new_coefs = coefs / scales  # element-wise division

    # Adjust intercept
    intercept_shift = torch.sum(coefs * means / scales)
    new_intercept = intercept - intercept_shift
    
    model.coef_ = new_coefs.numpy()
    model.intercept_ = new_intercept.numpy()
    
    return model


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


#------------------ Show accuracies -----------------------------------

def total_acc(y_true, y_pred):
    # return (y_pred == y_true).sum() / len(y_true)
    return accuracy_score(y_true, y_pred)


def balanced_acc(y_true, y_pred):
    # acc_0 = (y_pred[y_true == 0] == y_true[y_true == 0]).sum() / \
    #     len(y_true[y_true == 0])
    # acc_1 = (y_pred[y_true == 1] == y_true[y_true == 1]).sum() / \
    #     len(y_true[y_true == 1])
    # return (acc_0 + acc_1) / 2
    return balanced_accuracy_score(y_true, y_pred)


def print_correct_preds(y_true, y_pred):
    correct_0 = (y_pred[y_true == 0] == y_true[y_true == 0]).sum()
    print(f'    Class 0: {correct_0} / {len(y_true[y_true == 0])} '
          f'({correct_0 / len(y_true[y_true == 0]):.4f})')
    
    correct_1 = (y_pred[y_true == 1] == y_true[y_true == 1]).sum()
    print(f'    Class 1: {correct_1} / {len(y_true[y_true == 1])} '
          f'({correct_1 / len(y_true[y_true == 1]):.4f})')


#------------------ Train LR model ------------------------------------

def create_lr_model():
    model_class = linear_model.LogisticRegression
    param_dict = {
        'solver': 'saga',
        'penalty': 'elasticnet',
        'max_iter': 100,
        'l1_ratio': 1,
        'class_weight': 'balanced',
        'C': 0.1
    }
    model = model_class(**param_dict)
    return model


def train_lr_model(model_type, x_train, y_train, x_test, y_test):
    model = create_lr_model()
    
    print('*TRAINING MODEL*\n')
    
    if model_type == 'vanilla':
        model.fit(x_train, y_train)
    
    elif model_type == 'average':
        all_coefs = []
        all_inters = []
        kf = RepeatedStratifiedKFold(n_splits=5,
                                     n_repeats=20)
        
        for train_idx, _ in kf.split(x_train, y_train):
            model.fit(x_train[train_idx], y_train[train_idx])
            all_coefs.append(model.coef_)
            all_inters.append(model.intercept_)
        
        all_coefs = np.vstack(all_coefs)
        all_inters = np.hstack(all_inters)
        
        model.coef_ = np.mean(all_coefs, axis=0)
        model.intercept_ = np.mean(all_inters)
            
    else:
        raise ValueError('`model_type` should be "vanilla" or "average"')
    
    # Accuracy
    y_train_lr = model.predict(x_train)
    y_test_lr = model.predict(x_test)
    
    train_acc = accuracy_score(y_train, y_train_lr)
    test_acc = accuracy_score(y_test, y_test_lr)
    print(f'Accuracy: '
          f'Train: {train_acc:.2f}, Test: {test_acc:.2f}')
    
    train_bal_acc = balanced_accuracy_score(y_train, y_train_lr)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_lr)
    print(f'Balanced accuracy: '
          f'Train: {train_bal_acc:.2f}, Test: {test_bal_acc:.2f}')
    
    print('\n*Correct predicitons*')
    print('Train:')
    print_correct_preds(y_train, y_train_lr)
    print('Test:')
    print_correct_preds(y_test, y_test_lr)
    print()
    
    return model


#------------------ Tensorization ------------------------------------

@torch.no_grad()
def tensorize(fn_model, embedding, x_train, y_train, x_test, y_test,
              x_sketch, y_sketch, sketch_size, phys_dim, domain, bond_dim,
              cum_percentage, batch_size, device, dtype, verbose):
    
    ids = torch.randperm(x_sketch.size(0))
    x_sketch = x_sketch[ids]
    y_sketch = y_sketch[ids]
    
    print('*TENSORIZING MODEL*\n')
    
    cores, info_dict = tt_rss(function=fn_model,
                              embedding=embedding,
                              sketch_samples=x_sketch[:sketch_size],
                              labels=y_sketch[:sketch_size],
                              domain_multiplier=1,
                              domain=domain,
                              rank=bond_dim,
                              cum_percentage=cum_percentage,
                              batch_size=batch_size,
                              device=device,
                              dtype=dtype,
                              verbose=verbose,
                              return_info=True)
    
    print(f'Info: {info_dict}')
    
    mps = tk.models.MPSLayer(tensors=cores)
    mps.trace(torch.zeros(1, x_sketch.size(1), phys_dim))
    
    # Error
    y_sketch_mps = mps(embedding(x_sketch))
    y_train_mps = mps(embedding(x_train))
    y_test_mps = mps(embedding(x_test))
    
    y_sketch_lr = fn_model(x_sketch)
    y_train_lr = fn_model(x_train)
    y_test_lr = fn_model(x_test)
    
    sketch_error = (y_sketch_mps - y_sketch_lr).norm().pow(2) / y_sketch_mps.size(0)
    train_error = (y_train_mps - y_train_lr).norm().pow(2) / y_train_mps.size(0)
    test_error = (y_test_mps - y_test_lr).norm().pow(2) / y_test_mps.size(0)
    
    print(f'\nMSE: '
          f'Sketch: {sketch_error:.2e}, '
          f'Train: {train_error:.2}, '
          f'Test: {test_error:.2e}')
    print(y_sketch_mps[:10])
    print(y_sketch_lr[:10])
    print(y_sketch[:10])
    
    # Sketch accuracy
    _, y_sketch_mps = y_sketch_mps.max(dim=1)
    _, y_sketch_lr = y_sketch_lr.max(dim=1)
    sketch_acc_mps = total_acc(y_sketch, y_sketch_mps)
    sketch_acc_lr = total_acc(y_sketch, y_sketch_lr)
    print(f'\n*Sketch accuracies*\n'
          f'Accuracy: TT: {sketch_acc_mps:.2f}, LR: {sketch_acc_lr:.2f}')
    
    sketch_bal_acc_mps = balanced_acc(y_sketch, y_sketch_mps)
    sketch_bal_acc_lr = balanced_acc(y_sketch, y_sketch_lr)
    print(f'Balanced accuracy: '
          f'TT: {sketch_bal_acc_mps:.2f}, '
          f'LR: {sketch_bal_acc_lr:.2f}')
    
    # Train/test accuracy
    _, y_train_mps = y_train_mps.max(dim=1)
    _, y_test_mps = y_test_mps.max(dim=1)
    
    train_acc = total_acc(y_train, y_train_mps)
    test_acc = total_acc(y_test, y_test_mps)
    print(f'\n*Train/test TT accuracies*\n'
          f'Accuracy: Train: {train_acc:.2f}, Test: {test_acc:.2f}')
    
    train_bal_acc = balanced_acc(y_train, y_train_mps)
    test_bal_acc = balanced_acc(y_test, y_test_mps)
    print(f'Balanced accuracy: '
          f'Train: {train_bal_acc:.2f}, Test: {test_bal_acc:.2f}')
    
    print('\n*Correct predicitons*')
    print('Train:')
    print_correct_preds(y_train, y_train_mps)
    print('Test:')
    print_correct_preds(y_test, y_test_mps)
    print()
    
    mps.reset()
    mps.unset_data_nodes()
    
    return mps


@torch.no_grad()
def renormalize(mps, phys_dim, discr_steps, n_classes, num_features,
                x_train, y_train, x_test, y_test):
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    n_num = len(num_features)
    n_cat = mps.n_features - n_num - 1
    n_features = mps.n_features
    
    # For first 4 continuous variables
    emb_input_cont = []
    for i in range(n_num):
        aux_domain = torch.linspace(x_train[:, i].min(),
                                    x_train[:, i].max(),
                                    discr_steps).unsqueeze(1)
        aux_emb_input_cont = embedding(aux_domain).squeeze(1)
        aux_emb_input_cont = aux_emb_input_cont.sum(dim=0, keepdim=True) / discr_steps
        emb_input_cont.append(aux_emb_input_cont)
    
    # For next 17 discrete variables
    aux_domain = torch.arange(phys_dim).unsqueeze(1)
    emb_input_discr = embedding(aux_domain).squeeze(1)
    emb_input_discr = emb_input_discr.sum(dim=0, keepdim=True)
    
    # For output variable
    emb_input_out = torch.ones(1, n_classes)
    
    # All features
    emb_input = emb_input_cont + [emb_input_discr.clone() for _ in range(n_cat)]
    emb_input = emb_input[:(n_features // 2)] + [emb_input_out] + \
                emb_input[(n_features // 2):]
    
    # Compute norm
    mps.reset()
    mps.unset_data_nodes()
    mps.out_features = []
    
    norm = mps(emb_input)
    mps.reset()
    mps.unset_data_nodes()
    print(f'Norm: {norm.item():.4f}')
    
    for node in mps.mats_env:
        node.tensor = node.tensor / norm.pow(1 / n_features)
    
    mps.out_features = [n_features // 2]
    
    mps.trace(torch.zeros(1, x_test.size(1), phys_dim))
    
    # Accuracy
    y_train_mps = mps(embedding(x_train))
    _, y_train_mps = y_train_mps.max(dim=1)
    
    y_test_mps = mps(embedding(x_test))
    _, y_test_mps = y_test_mps.max(dim=1)
    
    train_acc = total_acc(y_train, y_train_mps)
    test_acc = total_acc(y_test, y_test_mps)
    print(f'Model accuracy: Train: {train_acc:.2f}, Test: {test_acc:.2f}')
    
    train_bal_acc = balanced_acc(y_train, y_train_mps)
    test_bal_acc = balanced_acc(y_test, y_test_mps)
    print(f'Model balanced accuracy: '
          f'Train: {train_bal_acc:.2f}, Test: {test_bal_acc:.2f}')
    
    print('Correct predicitons')
    print('Train:')
    print_correct_preds(y_train, y_train_mps)
    print('Test:')
    print_correct_preds(y_test, y_test_mps)
    
    print()
    
    mps.reset()
    mps.unset_data_nodes()
    
    return mps


def norm(mps, phys_dim, discr_steps, n_classes, num_features, x_train):
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    n_num = len(num_features)
    n_cat = mps.n_features - n_num - 1
    n_features = mps.n_features
    
    # For first 4 continuous variables
    emb_input_cont = []
    for i in range(n_num):
        aux_domain = torch.linspace(x_train[:, i].min(),
                                    x_train[:, i].max(),
                                    discr_steps).unsqueeze(1)
        aux_emb_input_cont = embedding(aux_domain).squeeze(1)
        aux_emb_input_cont = aux_emb_input_cont.sum(dim=0, keepdim=True) / discr_steps
        emb_input_cont.append(aux_emb_input_cont)
    
    # For next 17 discrete variables
    aux_domain = torch.arange(phys_dim).unsqueeze(1)
    emb_input_discr = embedding(aux_domain).squeeze(1)
    emb_input_discr = emb_input_discr.sum(dim=0, keepdim=True)
    
    # For output variable
    emb_input_out = torch.ones(1, n_classes)
    
    # All features
    emb_input = emb_input_cont + [emb_input_discr.clone() for _ in range(n_cat)]
    emb_input = emb_input[:(n_features // 2)] + [emb_input_out] + \
                emb_input[(n_features // 2):]
    
    # Compute norm
    mps.reset()
    mps.unset_data_nodes()
    mps.out_features = []
    
    norm = mps(emb_input)
    mps.reset()
    mps.unset_data_nodes()
    
    return norm


#------------------ Distributions ------------------------------------

@torch.no_grad()
def get_distribution(mps, cond_features, cond_data, marg_features,
                     in_features, out_feature, num_features,
                     n_classes, phys_dim, x_train, discr_steps):
    
    assert set(cond_features) & set(marg_features) == set()
    assert set(cond_features) <= set(in_features) | set([out_feature])
    assert set(marg_features) <= set(in_features) | set([out_feature])
    assert set(num_features) <= set(in_features) | set([out_feature])
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    def basis_embedding(data):
        return tk.embeddings.basis(data.int(), dim=n_classes).float()
    
    n_features = mps.n_features
    
    all_features = in_features[:(n_features // 2)] + [out_feature] + \
                   in_features[(n_features // 2):]
    cat_features = list(set(in_features) - set(num_features))
    marg_out_features = list(set(all_features) - \
        (set(cond_features) | set(marg_features)))
    
    dict_feat_idx = {}
    for i, feat in enumerate(all_features):
        dict_feat_idx[feat] = i
    
    emb_input_dict = {}
    
    # Cond. emb. input
    for i, feat in enumerate(cond_features):
        emb_input_cond = torch.tensor(cond_data[i]).unsqueeze(0)
        
        if feat == out_feature:
            emb_input_cond = basis_embedding(emb_input_cond).squeeze(0)
        else:
            emb_input_cond = embedding(emb_input_cond).squeeze(0)
        
        emb_input_dict[feat] = emb_input_cond
    
    # Marg. out emb. input
    for feat in marg_out_features:
        if feat in cat_features:
            aux_domain = torch.arange(phys_dim).unsqueeze(1)
            emb_input_marg_out = embedding(aux_domain).squeeze(1)
            emb_input_marg_out = emb_input_marg_out.sum(dim=0)
        elif feat == out_feature:
            emb_input_marg_out = torch.ones(n_classes)
        else:
            aux_domain = torch.linspace(x_train[:, dict_feat_idx[feat]].min(),
                                        x_train[:, dict_feat_idx[feat]].max(),
                                        discr_steps).unsqueeze(1)
            emb_input_marg_out = embedding(aux_domain).squeeze(1)
            emb_input_marg_out = emb_input_marg_out.sum(dim=0)
            emb_input_marg_out = emb_input_marg_out / discr_steps
        
        emb_input_dict[feat] = emb_input_marg_out
    
    # Marg. emb. input
    for feat in marg_features:
        if feat in cat_features:
            aux_domain = torch.arange(phys_dim).unsqueeze(1)
            emb_input_marg = embedding(aux_domain).squeeze(1)
        elif feat == out_feature:
            aux_domain = torch.arange(n_classes).unsqueeze(1)
            emb_input_marg = basis_embedding(aux_domain).squeeze(1)
        else:
            aux_domain = torch.linspace(x_train[:, dict_feat_idx[feat]].min(),
                                        x_train[:, dict_feat_idx[feat]].max(),
                                        discr_steps).unsqueeze(1)
            emb_input_marg = embedding(aux_domain).squeeze(1)
        
        emb_input_dict[feat] = emb_input_marg
    
    # All emb. input
    mps.reset()
    mps.unset_data_nodes()
    
    emb_input = []
    data_nodes = []
    for i, feat in enumerate(all_features):
        emb_input.append(emb_input_dict[feat])
        
        if feat in marg_features:
            axes_names = (f'batch_({i})', 'feature')
        else:
            axes_names = ('feature',)
        
        node = tk.Node(tensor=emb_input_dict[feat],
                       axes_names=axes_names,
                       name='data',
                       network=mps,
                       data=True)
        data_nodes.append(node)
    
    # Connect MPS and data nodes
    for mps_node, data_node in zip(mps.mats_env, data_nodes):
        mps_node['input'] ^ data_node['feature']
    
    # Contract
    mats_env = mps.mats_env[:]
    mats_env[0] = mps.left_node @ mats_env[0]
    mats_env[-1] = mats_env[-1] @ mps.right_node
    
    for i in range(len(mats_env)):
        mats_env[i] = mats_env[i] @ data_nodes[i]
    
    result = mats_env[0]
    for node in mats_env[1:]:
        result @= node
    
    distr = result.tensor.pow(2)
    distr = distr / distr.sum()
    
    mps.reset()
    mps.unset_data_nodes()
    
    # Select order of marg. features
    marg_features_order = []
    for feat in all_features:
        if feat in marg_features:
            marg_features_order.append(feat)
    
    return distr, marg_features_order


def marginal_prediction(mps, cond_feature, cond_data, in_features, out_feature,
                        num_features, n_classes, phys_dim, x_train, discr_steps,
                        scalers_dict):
    cond_data_scaled = scale_input(cond_data, [cond_feature], scalers_dict)
    marg_features = ['Response']

    distr, _ = get_distribution(
        mps=mps,
        cond_features=[cond_feature],
        cond_data=cond_data_scaled,
        marg_features=marg_features,
        in_features=in_features,
        out_feature=out_feature,
        num_features=num_features,
        n_classes=n_classes,
        phys_dim=phys_dim,
        x_train=x_train,
        discr_steps=discr_steps
    )
    
    return float(distr[1])


@torch.no_grad()
def patient_prediction(model, patient, features, scalers_dict):
    patient_scaled = scale_input(patient, features, scalers_dict)
    patient_scaled = torch.tensor(patient_scaled).unsqueeze(0)
    
    result = model(patient_scaled)
    result = result / result.sum(dim=1, keepdim=True)
    return result[0, 1]


#------------------ Feature sensitivity ------------------------------------

@torch.no_grad()
def feature_sensitivity_cond(feature, model, x_train, features, scalers_dict):
    """
    This function conditions on all variables, setting them to the mean value
    for each feature. The feature of interest is perturbed in its domain.
    """
    try:
        base_data = [scalers_dict[feat].mean_ for feat in features]
    except:
        base_data = [scalers_dict[feat].data_min_ for feat in features]
    base_data_scaled = scale_input(base_data, features, scalers_dict)
    base_data_scaled = torch.tensor(base_data_scaled).unsqueeze(0)

    yvals = []
    
    feat_idx = features.index(feature)
    for limit in ['min', 'max']:
        cond_data = base_data_scaled.clone()
        if limit == 'min':
            cond_data[0, feat_idx] = x_train[:, feat_idx].min()
        elif limit == 'max':
            cond_data[0, feat_idx] = x_train[:, feat_idx].max()
        
        result = model(cond_data).detach()
        score = result / result.sum(dim=1, keepdim=True)
        score = score[0, 1]
        yvals.append(score)

    return yvals[-1] - yvals[0]


def feature_sensitivity_marg(feature, mps, in_features, out_feature,
                             num_features, n_classes, phys_dim, x_train,
                             discr_steps, scalers_dict):
    """
    This function marginalizes all variables, except the feature of interest.
    The feature of interest is perturbed in its domain. This is only for MPS,
    not LORIS.
    """
    yvals = []
    
    feat_idx = in_features.index(feature)
    for limit in ['min', 'max']:
        if limit == 'min':
            cond_data = float(x_train[:, feat_idx].min())
        elif limit == 'max':
            cond_data = float(x_train[:, feat_idx].max())
        
        score = marginal_prediction(
            mps=mps,
            cond_feature=feature,
            cond_data=cond_data,
            in_features=in_features,
            out_feature=out_feature,
            num_features=num_features,
            n_classes=n_classes,
            phys_dim=phys_dim,
            x_train=x_train,
            discr_steps=discr_steps,
            scalers_dict=scalers_dict)
        
        yvals.append(score)

    return yvals[-1] - yvals[0]


@torch.no_grad()
def feature_sensitivity_coeffs(feature, model, x_train, features, scalers_dict):
    """
    This function tries to recover LORIS coefficients. The feature of interest
    is perturbed in its domain.
    """
    yvals = []
    
    feat_idx = features.index(feature)
    for limit in ['min', 'max']:
        cond_data = torch.zeros(1, len(features))
        if limit == 'max':
            cond_data[0, feat_idx] = 1
        
        result = model(cond_data).detach()
        score = result / result.sum(dim=1, keepdim=True)
        score = score[0, 1]
        logit = (score / (1 - score)).log()
        
        yvals.append(logit)
    
    coeff = (yvals[-1] - yvals[0])
    return coeff


#------------------ Monotonic plots ------------------------------------

# This function is adapted from loris/code/07_1.PanCancer_LORIS_TMB_vs_resProb_curve.py
@torch.no_grad()
def response_curve(model, x_train, y_train, xlabel,
                   bin_size=0.1, bs_number=1000, Plot_type=None):
    result = model(x_train)
    score = result / result.sum(dim=1, keepdim=True)
    y_pred = score[:, 1].numpy()
    y_true = y_train.numpy()
    
    sampleNUM = len(y_true)
    score_list = np.arange(0.0, 1.01, 0.01)
    num_scores = len(score_list)

    # Objective reponse ratio (what percentage of patients in this bin survived)
    ORR_list = [[] for _ in range(num_scores)]
    ORR_valid = [False for _ in range(num_scores)]  # Track if a bin ever had samples

    # Bootstrapping
    for _ in range(bs_number):
        idx_resampled = random.choices(range(sampleNUM), k=sampleNUM)
        aux_y_true = y_true[idx_resampled]
        aux_y_pred = y_pred[idx_resampled]
        
        for i, score in enumerate(score_list):
            # Set the bin size
            bin_mask = (aux_y_pred > score - bin_size / 2) & \
                (aux_y_pred <= score + bin_size / 2)

            # Record ORR only if samples exist in this bin
            if bin_mask.sum() > 0:
                ORR_list[i].append(aux_y_true[bin_mask].mean())
                ORR_valid[i] = True
            
            else:
                ORR_list[i].append(np.nan)  # Use NaN for clarity


    # Compute statistics, skipping NaNs (mean after bootstrapping)
    ORR_mean = [np.nanmean(x) if ORR_valid[i] else np.nan
                for i, x in enumerate(ORR_list)]
    # Add the 95% confidence interval
    ORR_05 = [np.nanquantile(x, 0.05) if ORR_valid[i] else np.nan
              for i, x in enumerate(ORR_list)]
    ORR_95 = [np.nanquantile(x, 0.95) if ORR_valid[i] else np.nan
              for i, x in enumerate(ORR_list)]

    # Forward-fill to smooth the line
    # Instead of having drops when there are no samples, we hold at
    # the previous point
    def forward_fill(arr):
        filled = []
        last_val = np.nan
        for val in arr:
            if not np.isnan(val):
                last_val = val
            filled.append(last_val)
        return filled

    ORR_mean = forward_fill(ORR_mean)
    ORR_05 = forward_fill(ORR_05)
    ORR_95 = forward_fill(ORR_95)


    # Plot
    _, ax = plt.subplots(figsize=(3.5, 3))
    ax.plot(score_list, ORR_mean, '-', color='r', label='Mean')
    ax.fill_between(score_list, ORR_05, ORR_95, color='r', alpha=0.25)

    # Add shading
    ax.axvspan(0, 0.3, facecolor='grey', alpha=0.2)   # Grey left region
    ax.axvspan(0.7, 1.0, facecolor='green', alpha=0.2)  # Green right region

    ax.set_ylabel("Response probability (\%)")
    ax.set_xlabel(xlabel)

    ax.set_ylim([-0.02, 1.02])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot([0, 1], [0, 1], '--',color='red', linewidth=1, label='y = x')

    plt.tight_layout()
    return ax
