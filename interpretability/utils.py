import os
import copy
import joblib
import random

from sklearn import linear_model
from diffprivlib import models as dp_models

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from scipy.interpolate import interp1d

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss


#--------------------- Load and scale data --------------------

def load_data(cwd, in_features, out_feature, datasets, scaler_type):
    data_file = os.path.join(cwd, 'AllData.xlsx')

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
    data_file = os.path.join(cwd, 'AllData.xlsx')
    
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


def accuracy(y_true, y_proba):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.detach().numpy()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    best_threshold = thresholds[np.argmax(youden)]
    
    y_pred = (y_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    return acc


def balanced_accuracy(y_true, y_proba):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.detach().numpy()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    best_threshold = thresholds[np.argmax(youden)]
    
    y_pred = (y_proba >= best_threshold).astype(int)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return bacc


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

def print_correct_preds(y_true, y_pred):
    correct_0 = (y_pred[y_true == 0] == y_true[y_true == 0]).sum()
    print(f'    Class 0: {correct_0} / {len(y_true[y_true == 0])} '
          f'({correct_0 / len(y_true[y_true == 0]):.4f})')
    
    correct_1 = (y_pred[y_true == 1] == y_true[y_true == 1]).sum()
    print(f'    Class 1: {correct_1} / {len(y_true[y_true == 1])} '
          f'({correct_1 / len(y_true[y_true == 1]):.4f})')


def evaluate_by_cancertype(model, x_train, y_train, scalers_dict):
    cond_features = ['CancerType1', 'CancerType2', 'CancerType3', 'CancerType4',
                     'CancerType5', 'CancerType6', 'CancerType7', 'CancerType8',
                     'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12',
                     'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16']
    
    for cancer_idx in range(len(cond_features)):
        cond_data = torch.zeros(len(cond_features))
        cond_data[cancer_idx] = 1
        cond_data_scaled = scale_input(cond_data, cond_features, scalers_dict)
        cond_data_scaled = torch.tensor(cond_data_scaled).unsqueeze(0)
        
        # Evaluate model
        mask = (x_train[:, 5:] == cond_data_scaled).all(dim=1)
        
        aux_x_train = x_train[mask]
        aux_y_train = y_train[mask]
        
        y_pred = model(aux_x_train)
        _, y_pred = y_pred.max(dim=1)
        
        train_acc = accuracy(aux_y_train, y_pred)
        train_bal_acc = balanced_accuracy(aux_y_train, y_pred)
        print(f'{cond_features[cancer_idx]}: '
              f'Train accuracy: {train_acc:.2f}, '
              f'Train balanced accuracy: {train_bal_acc:.2f}')
        
        print('Correct predicitons')
        print('Train:')
        print_correct_preds(aux_y_train, y_pred)
        
        print()


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


def create_lr_dp_model(epsilon):
    model_class = dp_models.LogisticRegression
    param_dict = {
        'max_iter': 100,
        'epsilon': epsilon,
    }
    model = model_class(**param_dict)
    return model


def evaluate_lr(model, x_train, y_train, x_test, y_test):
    # Accuracy
    y_proba_train_lr = model.predict_proba(x_train)[:, 1]
    y_proba_test_lr = model.predict_proba(x_test)[:, 1]
    
    y_train_lr = model.predict(x_train)
    y_test_lr = model.predict(x_test)
    
    train_acc = accuracy(y_train, y_proba_train_lr)
    test_acc = accuracy(y_test, y_proba_test_lr)
    print(f'Accuracy: '
          f'Train: {train_acc:.2f}, Test: {test_acc:.2f}')
    
    train_bal_acc = balanced_accuracy(y_train, y_proba_train_lr)
    test_bal_acc = balanced_accuracy(y_test, y_proba_test_lr)
    print(f'Balanced accuracy: '
          f'Train: {train_bal_acc:.2f}, Test: {test_bal_acc:.2f}')
    
    print('\n*Correct predicitons*')
    print('Train:')
    print_correct_preds(y_train, y_train_lr)
    print('Test:')
    print_correct_preds(y_test, y_test_lr)
    print()


def train_lr_model(model_type, x_train, y_train, x_test, y_test, dp=False, epsilon=None):
    if dp:
        if epsilon is None:
            raise ValueError('`epsilon` should be float')
        model = create_lr_dp_model(epsilon)
    else:
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
    evaluate_lr(model, x_train, y_train, x_test, y_test)
    
    return model


def save_lr(model, filepath):
    coefs = torch.from_numpy(model.coef_).flatten()
    intercepts = torch.from_numpy(model.intercept_).flatten()
    params = torch.cat([coefs, intercepts])
    joblib.dump(params, filepath)


def load_lr(filepath):
    params = joblib.load(filepath)
    model = linear_model.LogisticRegression()
    model.coef_ = params[:-1].unsqueeze(0).numpy()
    model.intercept_ = params[-1:].numpy()
    model.classes_ = np.array([0, 1])  # assumes binary classification
    return model


#------------------ Train NN model ------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1),
            # nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.net(x)

        if not self.training:
            # Eval mode → return probabilities
            return self.sigmoid(logits)
        else:
            # Train mode → return logits
            return logits


def create_nn_model(input_dim):
    model_class = SimpleMLP
    model = model_class(input_dim=input_dim,
                        hidden_sizes=(19, 19))
    return model

    
def evaluate_nn(model, x_train, y_train, x_test, y_test):
    # Accuracy
    y_proba_train_nn = model(x_train).flatten()
    y_proba_test_nn = model(x_test).flatten()
    
    y_train_nn = (model(x_train).flatten() > 0.5).float()
    y_test_nn = (model(x_test).flatten() > 0.5).float()
    
    train_acc = accuracy(y_train, y_proba_train_nn)
    test_acc = accuracy(y_test, y_proba_test_nn)
    print(f'Accuracy: '
          f'Train: {train_acc:.2f}, Test: {test_acc:.2f}')
    
    train_bal_acc = balanced_accuracy(y_train, y_proba_train_nn)
    test_bal_acc = balanced_accuracy(y_test, y_proba_test_nn)
    print(f'Balanced accuracy: '
          f'Train: {train_bal_acc:.2f}, Test: {test_bal_acc:.2f}')
    
    print('\n*Correct predicitons*')
    print('Train:')
    print_correct_preds(y_train, y_train_nn)
    print('Test:')
    print_correct_preds(y_test, y_test_nn)
    print()


def train_nn_model(x_train, y_train, x_test, y_test):
    model = create_nn_model(x_train.shape[1])
    
    print('*TRAINING MODEL*\n')
    
    # Train the model
    n_epochs = 100
    lr = 1e-3
    weight_decay = 1e-05
    
    N_pos = (y_train == 1).sum().item()
    N_neg = (y_train == 0).sum().item()

    pos_weight = torch.tensor([N_neg / N_pos], dtype=torch.float32)
    
    batch_size = 32
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True)
    
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

    for _ in range(n_epochs):
        for aux_x, aux_y in train_loader:
            aux_y = aux_y.float().view(-1, 1)
            
            optimizer.zero_grad()
            
            scores = model(aux_x)
            loss = criterion(scores, aux_y)
            
            loss.backward()
            optimizer.step()
    
    # Accuracy
    evaluate_nn(model, x_train, y_train, x_test, y_test)
    
    return model


def save_nn(model, filepath):
    params = [p.data for p in model.parameters()]
    torch.save(params, filepath)


def load_nn(input_dim, filepath):
    params = torch.load(filepath, weights_only=True)
    model = create_nn_model(input_dim)
    for p, loaded_p in zip(model.parameters(), params):
        p.data.copy_(loaded_p)
    return model


#------------------ Tensorization ------------------------------------

def evaluate_mps(mps, embedding, x_train, y_train, x_test, y_test):
    # Accuracy
    y_proba_train_mps = mps(embedding(x_train))[:, 1]
    y_proba_test_mps = mps(embedding(x_test))[:, 1]
    
    _, y_train_mps = mps(embedding(x_train)).max(dim=1)
    _, y_test_mps = mps(embedding(x_test)).max(dim=1)
    
    train_acc = accuracy(y_train, y_proba_train_mps)
    test_acc = accuracy(y_test, y_proba_test_mps)
    print(f'\n*Train/test TT accuracies*\n'
          f'Accuracy: Train: {train_acc:.2f}, Test: {test_acc:.2f}')
    
    train_bal_acc = balanced_accuracy(y_train, y_proba_train_mps)
    test_bal_acc = balanced_accuracy(y_test, y_proba_test_mps)
    print(f'Balanced accuracy: '
          f'Train: {train_bal_acc:.2f}, Test: {test_bal_acc:.2f}')
    
    print('\n*Correct predicitons*')
    print('Train:')
    print_correct_preds(y_train, y_train_mps)
    print('Test:')
    print_correct_preds(y_test, y_test_mps)
    print()


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
    sketch_acc_mps = accuracy(y_sketch, y_sketch_mps)
    sketch_acc_lr = accuracy(y_sketch, y_sketch_lr)
    print(f'\n*Sketch accuracies*\n'
          f'Accuracy: TT: {sketch_acc_mps:.2f}, LR: {sketch_acc_lr:.2f}')
    
    sketch_bal_acc_mps = balanced_accuracy(y_sketch, y_sketch_mps)
    sketch_bal_acc_lr = balanced_accuracy(y_sketch, y_sketch_lr)
    print(f'Balanced accuracy: '
          f'TT: {sketch_bal_acc_mps:.2f}, '
          f'LR: {sketch_bal_acc_lr:.2f}')
    
    # Train/test accuracy
    evaluate_mps(mps, embedding, x_train, y_train, x_test, y_test)
    
    mps.reset()
    mps.unset_data_nodes()
    
    return mps


@torch.no_grad()
def renormalize(mps, phys_dim, discr_steps, n_classes, num_features,
                x_train, y_train, x_test, y_test):
    
    print('*RENORMALIZING MODEL*\n')
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    n_num = len(num_features)
    n_cat = mps.n_features - n_num - 1
    n_features = mps.n_features
    out_position = mps.out_position
    
    # For first 4 continuous variables
    emb_matrices_cont = []
    for i in range(n_num):
        min_value = x_train[:, i].min()
        min_value = min_value - 0.25 * min_value.abs()
        max_value = x_train[:, i].max()
        max_value = max_value + 0.25 * max_value.abs()
        
        domain = torch.linspace(min_value, max_value, discr_steps).unsqueeze(1)
        step_size = domain[1] - domain[0]
        
        emb_input = embedding(domain).squeeze(1)
        aux_emb_mat = emb_input.T @ (step_size * emb_input)
        emb_matrices_cont.append(aux_emb_mat)
    
    # For next 17 binary variables
    emb_matrices_discr = []
    for i in range(n_cat):
        min_value = x_train[:, i].min()
        max_value = x_train[:, i].max()
        
        domain = torch.tensor([min_value, max_value]).unsqueeze(1)
        emb_input = embedding(domain).squeeze(1)
        aux_emb_mat = emb_input.T @ emb_input
        emb_matrices_discr.append(aux_emb_mat)
    
    # For output variable
    emb_matrix_out = torch.eye(n_classes)
    
    # All features
    emb_matrices = emb_matrices_cont + emb_matrices_discr
    emb_matrices = emb_matrices[:out_position] + [emb_matrix_out] + \
        emb_matrices[out_position:]
    
    # Compute norm
    mps.reset()
    mps.unset_data_nodes()
    mps.in_features = []
    
    sq_norm = mps(marginalize_output=True,
                  embedding_matrices=emb_matrices)
    norm = sq_norm.pow(1/2)
    print(f'Norm: {norm.item():.4f}')
    
    mps.reset()
    mps.unset_data_nodes()
    
    for node in mps.mats_env:
        node.tensor = node.tensor / norm.pow(1 / n_features)
    
    mps.out_features = [out_position]
    
    mps.trace(torch.zeros(1, x_test.size(1), phys_dim))
    
    # Accuracy
    evaluate_mps(mps, embedding, x_train, y_train, x_test, y_test)
    
    mps.reset()
    mps.unset_data_nodes()
    
    return mps


@torch.no_grad()
def norm(mps, phys_dim, discr_steps, n_classes, num_features, x_train):
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    n_num = len(num_features)
    n_cat = mps.n_features - n_num - 1
    out_position = mps.out_position
    
    # For first 4 continuous variables
    emb_matrices_cont = []
    for i in range(n_num):
        min_value = x_train[:, i].min()
        min_value = min_value - 0.25 * min_value.abs()
        max_value = x_train[:, i].max()
        max_value = max_value + 0.25 * max_value.abs()
        
        domain = torch.linspace(min_value, max_value, discr_steps).unsqueeze(1)
        step_size = domain[1] - domain[0]
        
        emb_input = embedding(domain).squeeze(1)
        aux_emb_mat = emb_input.T @ (step_size * emb_input)
        emb_matrices_cont.append(aux_emb_mat)
    
    # For next 17 binary variables
    emb_matrices_discr = []
    for i in range(n_cat):
        min_value = x_train[:, i].min()
        max_value = x_train[:, i].max()
        
        domain = torch.tensor([min_value, max_value]).unsqueeze(1)
        emb_input = embedding(domain).squeeze(1)
        aux_emb_mat = emb_input.T @ emb_input
        emb_matrices_discr.append(aux_emb_mat)
    
    # For output variable
    emb_matrix_out = torch.eye(n_classes)
    
    # All features
    emb_matrices = emb_matrices_cont + emb_matrices_discr
    emb_matrices = emb_matrices[:out_position] + [emb_matrix_out] + \
        emb_matrices[out_position:]
    
    # Compute norm
    mps.reset()
    mps.unset_data_nodes()
    mps.in_features = []
    
    sq_norm = mps(marginalize_output=True,
                  embedding_matrices=emb_matrices)
    norm = sq_norm.pow(1/2)
    print(f'Norm: {norm.item():.4f}')
    
    mps.reset()
    mps.unset_data_nodes()
    
    return norm


@torch.no_grad()
def get_cancertype_mps(mps, phys_dim, cancer_idx, x_train, y_train, scalers_dict):
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    cond_features = ['CancerType1', 'CancerType2', 'CancerType3', 'CancerType4',
                     'CancerType5', 'CancerType6', 'CancerType7', 'CancerType8',
                     'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12',
                     'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16']
    cond_data = torch.zeros(len(cond_features))
    cond_data[cancer_idx] = 1
    cond_data_scaled = scale_input(cond_data, cond_features, scalers_dict)
    cond_data_scaled = torch.tensor(cond_data_scaled).unsqueeze(0)
    emb_data_scaled = embedding(cond_data_scaled).squeeze(0)
    
    n_features = mps.n_features
    out_position = mps.out_position
    
    data_nodes = []
    for i in range(len(cond_features)):
        node = tk.Node(tensor=emb_data_scaled[i, :],
                       axes_names=('feature',),
                       name='data',
                       network=mps,
                       data=True)
        data_nodes.append(node)
    
    # Connect MPS and data nodes
    # out_position is in between the cancer features
    cancer_nodes = mps.mats_env[5:out_position] + mps.mats_env[(out_position + 1):]
    assert len(cancer_nodes) == len(cond_features)
    
    for mps_node, data_node in zip(cancer_nodes, data_nodes):
        mps_node['input'] ^ data_node['feature']
    
    # Contract
    cancer_nodes[-1] = cancer_nodes[-1] @ mps.right_node
    for i in range(len(cond_features)):
        cancer_nodes[i] = cancer_nodes[i] @ data_nodes[i]
    
    all_contracted1 = cancer_nodes[0]
    for node in cancer_nodes[1:]:
        if all_contracted1.is_connected_to(node):
            all_contracted1 @= node
        else:
            break
    
    all_contracted2 = cancer_nodes[-1]
    for node in cancer_nodes[-2:0:-1]:
        if all_contracted2.is_connected_to(node):
            all_contracted2 = node @ all_contracted2
        else:
            break
    
    # Collect nodes
    nodes = []
    nodes.append(mps.left_node @ mps.mats_env[0])
    nodes += mps.mats_env[1:5]
    nodes.append(all_contracted1 @ mps.out_node @ all_contracted2)
    
    # Collect tensors
    tensors = [node.tensor for node in nodes]
    
    mps.reset()
    mps.unset_data_nodes()
    
    # Create new mps
    new_mps = tk.models.MPSLayer(tensors=tensors, out_position=5)
    new_mps.canonicalize(renormalize=True)
    new_mps.trace(torch.zeros(1, 5, phys_dim))
    
    # Evaluate model
    mask = (x_train[:, 5:] == cond_data_scaled).all(dim=1)
    
    x_train = x_train[mask, :5]
    y_train = y_train[mask]
    
    y_pred = new_mps(embedding(x_train))
    _, y_pred = y_pred.max(dim=1)
    
    train_acc = accuracy(y_train, y_pred)
    print(f'Model accuracy: Train: {train_acc:.2f}')
    
    train_bal_acc = balanced_accuracy(y_train, y_pred)
    print(f'Model balanced accuracy: '
          f'Train: {train_bal_acc:.2f}')
    
    print('Correct predicitons')
    print('Train:')
    print_correct_preds(y_train, y_pred)
    
    print()
    
    new_mps.reset()
    new_mps.unset_data_nodes()
    
    return new_mps, x_train, y_train, train_bal_acc


#------------------ Distributions ------------------------------------

def take_all_diagonals(x: torch.Tensor) -> torch.Tensor:
    """
    Takes the diagonal along every consecutive pair of dimensions:
    (d1,d1,d2,d2,...) -> (d1,d2,...).

    Args:
        x (torch.Tensor): input tensor of shape (d1,d1,d2,d2,...,dp,dp)

    Returns:
        torch.Tensor: tensor of shape (d1,d2,...,dp)
    """
    num_pairs = x.ndim // 2
    for i in range(num_pairs):
        # after each diagonal, dimensions shrink, so indices shift by -i
        x = x.diagonal(dim1=i, dim2=i+1)
    return x


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
    
    out_position = mps.out_position
    
    all_features = in_features[:out_position] + [out_feature] + \
                   in_features[out_position:]
    cat_features = list(set(in_features) - set(num_features))
    marg_out_features = list(set(all_features) - \
        (set(cond_features) | set(marg_features)))
    
    dict_feat_idx = {}
    for i, feat in enumerate(all_features):
        dict_feat_idx[feat] = i
    
    dict_in_feat_idx = {}
    for i, feat in enumerate(in_features):
        dict_in_feat_idx[feat] = i
    
    emb_input_dict = {}
    emb_matrices_dict = {}
    
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
            min_value = x_train[:, dict_in_feat_idx[feat]].min()
            max_value = x_train[:, dict_in_feat_idx[feat]].max()
            domain = torch.tensor([min_value, max_value]).unsqueeze(1)
            emb_input = embedding(domain).squeeze(1)
            emb_mat_marg_out = emb_input.T @ emb_input
        elif feat == out_feature:
            emb_mat_marg_out = torch.eye(n_classes)
        else:
            min_value = x_train[:, dict_in_feat_idx[feat]].min()
            min_value = min_value - 0.25 * min_value.abs()
            max_value = x_train[:, dict_in_feat_idx[feat]].max()
            max_value = max_value + 0.25 * max_value.abs()
            
            domain = torch.linspace(min_value, max_value, discr_steps).unsqueeze(1)
            step_size = domain[1] - domain[0]
            
            emb_input = embedding(domain).squeeze(1)
            emb_mat_marg_out = emb_input.T @ (step_size * emb_input)
        
        emb_matrices_dict[feat] = emb_mat_marg_out
    
    # Marg. emb. input
    for feat in marg_features:
        if feat in cat_features:
            min_value = x_train[:, dict_in_feat_idx[feat]].min()
            max_value = x_train[:, dict_in_feat_idx[feat]].max()
            domain = torch.tensor([min_value, max_value]).unsqueeze(1)
            emb_input_marg = embedding(domain).squeeze(1)
        elif feat == out_feature:
            domain = torch.arange(n_classes).unsqueeze(1)
            emb_input_marg = basis_embedding(domain).squeeze(1)
        else:
            min_value = x_train[:, dict_in_feat_idx[feat]].min()
            max_value = x_train[:, dict_in_feat_idx[feat]].max()
            domain = torch.linspace(min_value, max_value, discr_steps).unsqueeze(1)
            emb_input_marg = embedding(domain).squeeze(1)
        
        emb_input_dict[feat] = emb_input_marg
    
    # All emb. input
    mps.reset()
    mps.unset_data_nodes()
    
    mps.out_features = [dict_feat_idx[feat]
                        for feat in marg_out_features]
    
    emb_input = []
    data_nodes = []
    for i, feat in enumerate(all_features):
        if feat in cond_features:
            emb_input.append(emb_input_dict[feat])
            axes_names = ('feature',)
        elif feat in marg_features:
            emb_input.append(emb_input_dict[feat])
            axes_names = (f'batch_({i})', 'feature')
        else:
            continue
        
        node = tk.Node(tensor=emb_input_dict[feat],
                       axes_names=axes_names,
                       name='data',
                       network=mps,
                       data=True)
        data_nodes.append(node)
        
        node['feature'] ^ mps.mats_env[dict_feat_idx[feat]]['input']
    
    emb_matrices = []
    for i, feat in enumerate(all_features):
        if feat in marg_out_features:
            emb_matrices.append(emb_matrices_dict[feat])
    
    distr = mps(emb_input,
                inline_input=True,
                inline_mats=True,
                marginalize_output=True,
                embedding_matrices=emb_matrices)
    distr = take_all_diagonals(distr)
    distr = distr / distr.sum()
    
    mps.reset()
    mps.unset_data_nodes()
    
    mps.out_features = [out_position]
    
    # Select order of marg. features
    marg_features_order = []
    for feat in all_features:
        if feat in marg_features:
            marg_features_order.append(feat)
    
    return distr, marg_features_order


def marginal_prediction(mps, cond_feature, cond_data, in_features, out_feature,
                        num_features, n_classes, phys_dim, x_train, discr_steps,
                        scalers_dict, scale_data=True):
    if scale_data:
        cond_data_scaled = scale_input(cond_data, [cond_feature], scalers_dict)
    else:
        cond_data_scaled = [cond_data]
    
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
    return float(result[0, 1])


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
            cond_data[0, feat_idx] = 0.
        elif limit == 'max':
            cond_data[0, feat_idx] = 1.
        
        result = model(cond_data)
        score = result[0, 1]
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
    
    for limit in ['min', 'max']:
        if limit == 'min':
            cond_data = 0.
        elif limit == 'max':
            cond_data = 1.
        
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
            scalers_dict=scalers_dict,
            scale_data=False
            )
        
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
        
        result = model(cond_data)
        score = result[0, 1]
        logit = (score / (1 - score)).log()
        
        yvals.append(logit)
    
    coeff = (yvals[-1] - yvals[0])
    return coeff


#------------------ Monotonic plots ------------------------------------

# This function is adapted from loris/code/07_1.PanCancer_LORIS_TMB_vs_resProb_curve.py
@torch.no_grad()
def response_curve(model, x_train, y_train, xlabel, ax,
                   bin_size=0.1, bs_number=1000, Plot_type=None):
    result = model(x_train)
    y_pred = result[:, 1].numpy()
    y_true = y_train.numpy()
    
    sampleNUM = len(y_true)
    score_list = np.arange(0.0, 1.01, 0.01)
    num_scores = len(score_list)

    # Objective response ratio (what percentage of patients in this bin survived)
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
    ORR_05 = [np.nanquantile(x, 0.05) if ORR_valid[i] else np.nan
              for i, x in enumerate(ORR_list)]
    ORR_95 = [np.nanquantile(x, 0.95) if ORR_valid[i] else np.nan
              for i, x in enumerate(ORR_list)]

    # Forward-fill to smooth the line
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

    # Compute automatic shading limits based on y_true values
    ORR_mean_array = np.array(ORR_mean)
    valid_mask = ~np.isnan(ORR_mean_array)
    score_valid = np.array(score_list)[valid_mask]
    ORR_valid_vals = ORR_mean_array[valid_mask]

    # Interpolation for inverse mapping
    inv_func = interp1d(ORR_valid_vals, score_valid, bounds_error=False, fill_value=(score_valid[0], score_valid[-1]))
    x_gray = float(inv_func(0.1))  # corresponds to y_true=0.1
    x_green = float(inv_func(0.5))  # corresponds to y_true=0.5

    # Plot
    ax.plot(score_list, ORR_mean, '-', color='r', label='Mean')
    ax.fill_between(score_list, ORR_05, ORR_95, color='r', alpha=0.25)

    # Add automatic shading
    ax.axvspan(0, x_gray, facecolor='grey', alpha=0.2)
    ax.axvspan(x_green, 1.0, facecolor='green', alpha=0.2)
    
    ax.set_xlabel(xlabel)

    ax.set_ylim([-0.02, 1.02])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot([0, 1], [0, 1], '--', color='red', linewidth=1, label='y = x')

    # Return both ax and shading limits
    return ax, x_gray, x_green
