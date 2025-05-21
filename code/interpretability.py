import os
import copy

from itertools import chain, combinations

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import pandas as pd

import torch
import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss

cwd = os.getcwd()


def all_combinations(lst):
    return list(chain.from_iterable(combinations(lst, r)
                                    for r in range(1, len(lst) + 1)))


def dataScaler(data, featuresNA, numeric_featuresNA, scaler_type):
    data_scaled = copy.deepcopy(data)
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise Exception(
            'Unrecognized scaler type of %s! '
            'Only "sd" and "mM" are accepted.' % scaler_type)
    for feature in numeric_featuresNA:
        data_scaled[feature] = scaler.fit_transform(data[[feature]])
    x = pd.DataFrame(data_scaled, columns=featuresNA)
    return x


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


def load_data(featuresNA, phenoNA, dataset, scaler_type):
    data_dir = os.path.join(cwd, '..', '02.Input')
    data_file = os.path.join(data_dir, 'AllData.xlsx')

    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25

    data = pd.read_excel(data_file, sheet_name=dataset, index_col=0)
    
    # Data truncation
    data['TMB'] = [c if c < TMB_upper else TMB_upper for c in data['TMB']]
    data['Age'] = [c if c < Age_upper else Age_upper for c in data['Age']]
    data['NLR'] = [c if c < NLR_upper else NLR_upper for c in data['NLR']]
    
    all_features = featuresNA + [phenoNA]
    data_no_nans = data[all_features].dropna(axis=0)
    
    all_data = dataScaler(data_no_nans, all_features, featuresNA, scaler_type)
    
    return all_data


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


# NOTE: For MPS
def total_acc(y_true, y_pred):
    return (y_pred == y_true).sum() / len(y_true)


def balanced_acc(y_true, y_pred):
    acc_0 = (y_pred[y_true == 0] == y_true[y_true == 0]).sum() / \
        len(y_true[y_true == 0])
    acc_1 = (y_pred[y_true == 1] == y_true[y_true == 1]).sum() / \
        len(y_true[y_true == 1])
    return (acc_0 + acc_1) / 2


@torch.no_grad()
def tensorize(model, x_train, y_train, x_test, y_test,
              sketch_size, phys_dim, domain, bond_dim,
              cum_percentage, batch_size, device, verbose):
    
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
    # print(y_train_mps[:10])
    # print(y_train_lr[:10])
    
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
                x_test, y_test):
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    n_num = len(num_features)
    n_cat = mps.n_features - n_num - 1
    n_features = mps.n_features
    
    # For first 4 continuous variables
    aux_domain = torch.linspace(0, 1, discr_steps).unsqueeze(1)
    emb_input_cont = embedding(aux_domain).squeeze(1)
    emb_input_cont = emb_input_cont.sum(dim=0, keepdim=True) / discr_steps
    
    # For next 17 discrete variables
    aux_domain = torch.arange(phys_dim).unsqueeze(1)
    emb_input_discr = embedding(aux_domain).squeeze(1)
    emb_input_discr = emb_input_discr.sum(dim=0, keepdim=True)
    
    # For output variable
    emb_input_out = torch.ones(1, n_classes)
    
    # All features
    emb_input = [emb_input_cont.clone() for _ in range(n_num)] + \
                [emb_input_discr.clone() for _ in range(n_cat)]
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


@torch.no_grad()
def get_distribution(mps, cond_features, cond_data, marg_features,
                     in_features, out_feature, num_features,
                     n_classes, phys_dim, discr_steps):
    
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
            aux_domain = torch.linspace(0, 1, discr_steps).unsqueeze(1)
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
            aux_domain = torch.linspace(0, 1, discr_steps).unsqueeze(1)
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


if __name__ == '__main__':
    cwd = os.path.join(cwd, 'code')

    # "model_name" should be 'llr6' or 'nn2'
    model_name = 'llr6'
    n_classes = 2


    # Load data
    # ---------
    featuresNA = ['TMB', 'Albumin', 'NLR', 'Age', 'Systemic_therapy_history',
                  'CancerType1', 'CancerType2', 'CancerType3', 'CancerType4',
                  'CancerType5', 'CancerType6', 'CancerType7', 'CancerType8',
                  'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12',
                  'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16']
    phenoNA = 'Response'

    numeric_featuresNA = ['TMB', 'Albumin', 'NLR', 'Age']

    datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
                'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
    
    scaler_type = 'MinMax'
    # scaler_type = 'StandardScaler'

    dataset = 'Chowell_train'
    data_train = load_data(featuresNA, phenoNA, dataset, scaler_type)

    dataset = 'Chowell_test'
    data_test = load_data(featuresNA, phenoNA, dataset, scaler_type)


    # Train model
    # -----------
    x_train = data_train[featuresNA].values
    y_train = data_train[phenoNA].values

    x_test = data_test[featuresNA].values
    y_test = data_test[phenoNA].values

    print('\n* TRAINING MODEL:')
    model = train_model(model_name, x_train, y_train, x_test, y_test)


    # Tensorize model
    # ---------------
    xt_train = torch.from_numpy(x_train).float()
    yt_train = torch.from_numpy(y_train)

    xt_test = torch.from_numpy(x_test).float()
    yt_test = torch.from_numpy(y_test)

    # sketch_size could be chosen smaller, e.g., sketch_size = 50, and the
    # tensorization will be much faster returning a tn_model with the same accuracy,
    # but sometimes this could lead to having negative values for inputs with low
    # probability. Choosing a bigger sketch_size usually avoids these errors.
    sketch_size    = 200
    phys_dim       = 2
    domain         = torch.linspace(0, 1, phys_dim)
    bond_dim       = 2
    cum_percentage = 1 - 1e-2
    batch_size     = 500
    device         = torch.device('cpu')
    verbose        = False

    print('* TENSORIZING MODEL:')
    tn_model = tensorize(model=model,
                         x_train=xt_train,
                         y_train=yt_train,
                         x_test=xt_test,
                         y_test=yt_test,
                         sketch_size=sketch_size,
                         phys_dim=phys_dim,
                         domain=domain,
                         bond_dim=bond_dim,
                         cum_percentage=cum_percentage,
                         batch_size=batch_size,
                         device=device,
                         verbose=verbose)


    # Renormalize model
    # -----------------

    # We renormalize the tensor network so that it represents a normalized
    # distribution of all the features, instead of only giving normalized
    # conditional distributions for the output feature ("Response")

    # Contiuous variables have to be integrated. "discr_steps" indicates the number
    # of dicretization steps used to discretize the continuous variable and perform
    # numerical integration
    discr_steps = int(1e5)

    print('* RENORMALIZING TN MODEL:')
    tn_model = renormalize(mps=tn_model,
                           phys_dim=phys_dim,
                           discr_steps=discr_steps,
                           n_classes=n_classes,
                           num_features=numeric_featuresNA,
                           x_test=xt_test,
                           y_test=yt_test)


    # Get conditional/marginal distribution
    # -------------------------------------

    # We need to specify the condition and marginal variables. The rest will be
    # marginalized out. The "Response" feature can be included as condition or
    # marginal

    # Select features on which we condition
    cond_features = ['TMB', 'Albumin']

    # Create tensor with values (already scaled to [0, 1])
    # for conditioned features
    cond_data = [0.2, 0.9]  # Just an example

    # Select features which we marginalize
    marg_features = ['Systemic_therapy_history', 'Response']

    # In this example, we will get the distribution 
    # P('Systemic_therapy_history', 'Response' | 'TMB', 'Albumin'),
    # marginalizing out the rest of the features

    # The distribution will be given as a tensor with shape:
    # (dim_marg_1, ..., dim_marg_n), with "dim_marg_{i}" being the dimensions of
    # each marginal feature. These marginal features could be returned in a
    # different order as the one specified in "marg_features". The order will
    # be returned in the list "marg_feat_order"

    # For continuous features, the dimension of the marginal features will be the
    # specified "discr_steps"

    discr_steps = int(1e5)

    distr, marg_feat_order = get_distribution(
        mps=tn_model,
        cond_features=cond_features,
        cond_data=cond_data,
        marg_features=marg_features,
        in_features=featuresNA,
        out_feature=phenoNA,
        num_features=numeric_featuresNA,
        n_classes=n_classes,
        phys_dim=phys_dim,
        discr_steps=discr_steps
    )

    print(marg_feat_order, distr.shape)
    print(distr)
    print(distr.sum())
    print(marg_feat_order[1], distr.sum(dim=0))
    print(marg_feat_order[0], distr.sum(dim=1))
    print()


    # Example 2
    cond_features = ['Response']
    cond_data = [1.]

    marg_features = ['TMB']

    discr_steps = int(1e1)  # int(1e5)

    distr, marg_feat_order = get_distribution(
        mps=tn_model,
        cond_features=cond_features,
        cond_data=cond_data,
        marg_features=marg_features,
        in_features=featuresNA,
        out_feature=phenoNA,
        num_features=numeric_featuresNA,
        n_classes=n_classes,
        phys_dim=phys_dim,
        discr_steps=discr_steps
    )

    print(marg_feat_order, distr.shape)
    print(distr)
    print(distr.sum())
    print()


    # Example 3
    cond_features = ['Systemic_therapy_history']
    cond_data = [0.]

    marg_features = ['Response']

    discr_steps = int(1e5)

    distr, marg_feat_order = get_distribution(
        mps=tn_model,
        cond_features=cond_features,
        cond_data=cond_data,
        marg_features=marg_features,
        in_features=featuresNA,
        out_feature=phenoNA,
        num_features=numeric_featuresNA,
        n_classes=n_classes,
        phys_dim=phys_dim,
        discr_steps=discr_steps
    )

    print(marg_feat_order, distr.shape)
    print(distr)
    print(distr.sum())
    print()
    
    # Example 4
    cond_features = []
    cond_data = []

    marg_features = ['Response', 'Albumin']

    discr_steps = int(1e1)  # int(1e5)

    distr, marg_feat_order = get_distribution(
        mps=tn_model,
        cond_features=cond_features,
        cond_data=cond_data,
        marg_features=marg_features,
        in_features=featuresNA,
        out_feature=phenoNA,
        num_features=numeric_featuresNA,
        n_classes=n_classes,
        phys_dim=phys_dim,
        discr_steps=discr_steps
    )

    print(marg_feat_order, distr.shape)
    print(distr / distr.sum(dim=1, keepdim=True))
    print()
