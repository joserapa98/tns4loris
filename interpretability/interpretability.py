import os
import copy
import random

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss


cwd = os.getcwd()


#--------------------- Load and scale data --------------------

def data_scaler(data, features, num_features, scaler_type):
    data_scaled = copy.deepcopy(data)
    if scaler_type == 'Standard':
        scaler_class = StandardScaler
    elif scaler_type == 'MinMax':
        scaler_class = MinMaxScaler
    else:
        raise ValueError(f'Unrecognized scaler type of {scaler_type}. '
                         'Only "Standard" and "MinMax" are accepted.')
    scalers_dict = {}
    for feature in num_features:
        scaler = scaler_class()
        data_scaled[feature] = scaler.fit_transform(data[[feature]])
        scalers_dict[feature] = scaler
    df_data_scaled = pd.DataFrame(data_scaled, columns=features)
    return df_data_scaled, scalers_dict


def scale_input(patient, features, scalers_dict):
    """
    Takes in a list of features and their names and scales each one using
    the corresponding scaler fit to Chowell train data.
    """
    patient_df= pd.DataFrame([patient], columns=features)
    
    for feature in patient_df.columns:
        if feature in scalers_dict:
            patient_df[feature] = scalers_dict[feature].transform(patient_df[[feature]])
    
    patient_list = patient_df.iloc[0].tolist()
    return patient_list


def load_data(cwd, in_features, out_feature, num_features, dataset, scaler_type):
    data_file = os.path.join(cwd, 'loris', '02.Input', 'AllData.xlsx')

    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25

    data = pd.read_excel(data_file, sheet_name=dataset, index_col=0)
    
    # Data truncation
    data['TMB'] = [c if c < TMB_upper else TMB_upper for c in data['TMB']]
    data['Age'] = [c if c < Age_upper else Age_upper for c in data['Age']]
    data['NLR'] = [c if c < NLR_upper else NLR_upper for c in data['NLR']]
    
    all_features = in_features + [out_feature]
    data_no_nans = data[all_features].dropna(axis=0)
    
    all_data, scalers_dict = data_scaler(data_no_nans,
                                         all_features,
                                         num_features,
                                         scaler_type)
    
    return all_data, scalers_dict


#------------------ Train LR model ------------------------------------

def create_model(model_name):
    if model_name == 'nn2':
        model_type = MLPClassifier
        param_dict = {
            'max_iter': 100,
            'hidden_layer_sizes': (19, 19),
            'activation': 'tanh',
            'alpha': 1e-05,
            'early_stopping': False
        }
        
    elif model_name == 'llr6':
        model_type = linear_model.LogisticRegression
        param_dict = {
            'solver': 'saga',
            'penalty': 'elasticnet',
            'max_iter': 100,
            'l1_ratio': 1,
            'class_weight': 'balanced',
            'C': 0.1
        }
    
    model = model_type(**param_dict)
    return model


def train_model(model_name, x_train, y_train, x_test, y_test):
    model = create_model(model_name)
    model.fit(x_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(x_train))
    test_acc = accuracy_score(y_test, model.predict(x_test))
    print(f'Model accuracy: Train: {train_acc:.2f}, Test: {test_acc:.2f}')
    
    train_bal_acc = balanced_accuracy_score(y_train, model.predict(x_train))
    test_bal_acc = balanced_accuracy_score(y_test, model.predict(x_test))
    print(f'Model balanced accuracy: '
          f'Train: {train_bal_acc:.2f}, Test: {test_bal_acc:.2f}')
    print()
    
    return model


#------------------ MPS Accuracies ------------------------------------

def total_acc(y_true, y_pred):
    return (y_pred == y_true).sum() / len(y_true)


def balanced_acc(y_true, y_pred):
    acc_0 = (y_pred[y_true == 0] == y_true[y_true == 0]).sum() / \
        len(y_true[y_true == 0])
    acc_1 = (y_pred[y_true == 1] == y_true[y_true == 1]).sum() / \
        len(y_true[y_true == 1])
    return (acc_0 + acc_1) / 2


#------------------ Tensorization ------------------------------------

@torch.no_grad()
def tensorize(model, x_train, y_train, x_test, y_test,
              sketch_size, phys_dim, domain, bond_dim,
              cum_percentage, batch_size, device, dtype, verbose):
    
    def fn_model(data):
        result = torch.from_numpy(model.predict_proba(data)).float()
        return result
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    cores, info_dict = tt_rss(function=fn_model,
                              embedding=embedding,
                              sketch_samples=x_train[:sketch_size],
                              labels=y_train[:sketch_size],
                              domain_multiplier=1,
                              domain=domain,
                              rank=bond_dim,
                              cum_percentage=cum_percentage,
                              batch_size=batch_size,
                              device=device,
                              dtype=dtype,
                              verbose=verbose,
                              return_info=True)
    
    print('Info:', info_dict)
    
    mps = tk.models.MPSLayer(tensors=cores)
    mps.trace(torch.zeros(1, x_train.size(1), phys_dim))
    
    # Error
    y_train_mps = mps(embedding(x_train))
    y_test_mps = mps(embedding(x_test))
    
    y_train_lr = fn_model(x_train)
    y_test_lr = fn_model(x_test)
    
    train_error = (y_train_mps - y_train_lr).norm().pow(2) / y_train_mps.size(0)
    test_error = (y_test_mps - y_test_lr).norm().pow(2) / y_test_mps.size(0)
    
    print(f'MSE: Train: {train_error:.2}, Test: {test_error:.2e}',)
    print(y_train_mps[:10])
    print(y_train_lr[:10])
    
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
    print()
    
    mps.reset()
    mps.unset_data_nodes()
    
    return mps


@torch.no_grad()
def renormalize(mps, phys_dim, discr_steps, n_classes, num_features,
                x_train, x_test, y_test):
    
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
    
    for node in mps.mats_env:
        node.tensor = node.tensor / norm.pow(1 / n_features)
    
    mps.out_features = [n_features // 2]
    
    mps.trace(torch.zeros(1, x_test.size(1), phys_dim))
    
    y_test_mps = mps(embedding(x_test))
    _, y_test_mps = y_test_mps.max(dim=1)
    
    test_acc = total_acc(y_test, y_test_mps)
    print(f'Model accuracy: Test: {test_acc:.2f}')
    
    test_bal_acc = balanced_acc(y_test, y_test_mps)
    print(f'Model balanced accuracy: Test: {test_bal_acc:.2f}')
    print()
    
    mps.reset()
    mps.unset_data_nodes()
    
    return mps


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
    
    distr = result.tensor
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
        base_data = [scalers_dict['TMB'].mean_,
                     scalers_dict['Albumin'].mean_,
                     scalers_dict['NLR'].mean_,
                     scalers_dict['Age'].mean_, 1] + [0] * (len(features) - 5)
    except:
        base_data = [scalers_dict['TMB'].min_,
                     scalers_dict['Albumin'].min_,
                     scalers_dict['NLR'].min_,
                     scalers_dict['Age'].min_, 1] + [0] * (len(features) - 5)
    base_data_scaled = scale_input(base_data, features, scalers_dict)
    base_data_scaled = torch.tensor(base_data_scaled).unsqueeze(0)

    yvals = []
    
    if feature == 'CancerType':
        for i in [5, len(base_data) - 1]:
            cond_data = base_data_scaled.clone()
            cond_data[0, i] = 1  # set 1 at the current position (others remain 0 from index 5 onward)
            
            result = model(cond_data).detach()
            score = result / result.sum(dim=1, keepdim=True)
            score = score[0, 1]
            yvals.append(score)

    else:
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
    
    if feature == 'CancerType':
        for idx in ['1', '16']:
            score = marginal_prediction(
                mps=mps,
                cond_feature=feature+idx,
                cond_data=1,
                in_features=in_features,
                out_feature=out_feature,
                num_features=num_features,
                n_classes=n_classes,
                phys_dim=phys_dim,
                x_train=x_train,
                discr_steps=discr_steps,
                scalers_dict=scalers_dict)
            
            yvals.append(score)

    else:
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
    try:
        base_data = [scalers_dict['TMB'].mean_,
                     scalers_dict['Albumin'].mean_,
                     scalers_dict['NLR'].mean_,
                     scalers_dict['Age'].mean_, 1] + [0] * (len(features) - 5)
    except:
        base_data = [scalers_dict['TMB'].min_,
                     scalers_dict['Albumin'].min_,
                     scalers_dict['NLR'].min_,
                     scalers_dict['Age'].min_, 1] + [0] * (len(features) - 5)
    base_data_scaled = scale_input(base_data, features, scalers_dict)
    base_data_scaled = torch.tensor(base_data_scaled).unsqueeze(0)
    
    xvals = []
    yvals = []
    
    feat_idx = features.index(feature)
    for limit in ['min', 'max']:
        cond_data = base_data_scaled.clone()
        if limit == 'min':
            cond_data[0, feat_idx] = x_train[:, feat_idx].min()
        elif limit == 'max':
            cond_data[0, feat_idx] = x_train[:, feat_idx].max()
        xvals.append(cond_data[0, feat_idx].item())
        
        result = model(cond_data).detach()
        score = result / result.sum(dim=1, keepdim=True)
        score = score[0, 1]
        logit = (score / (1 - score)).log()
        
        yvals.append(logit)
    
    coeff = (yvals[-1] - yvals[0]) / (xvals[-1] - xvals[0])
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

    ax.set_ylabel("Response probability (%)")
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
    plt.show()
