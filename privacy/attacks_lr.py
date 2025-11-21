import sys
import os
import getopt
import joblib
import json

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold

import numpy as np
import torch
import tensorkrowch as tk

from utils import *


cwd = os.getcwd()

if torch.cuda.is_available():
    print('GPU is available')
    device = torch.device('cuda')
else:
    print('GPU is not available')
    device = torch.device('cpu')


# datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
#             'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2',
            'Shim_NSCLC', 'Kato_panCancer']
datasets_ids = list(range(1, len(datasets) + 1))

featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age',
              'CancerType1', 'CancerType2', 'CancerType3', 'CancerType4',
              'CancerType5', 'CancerType6', 'CancerType7', 'CancerType8',
              'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12',
              'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16']
phenoNA = 'Response'


def create_datasets_lr(model_name, scaler_type, model_type):
    # Load data
    all_data = load_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    test_size = 100
    
    x = all_data[featuresNA].values
    z = all_data['DatasetNum'].values
    
    _, x_test, = train_test_split(x,
                                  test_size=test_size,
                                  shuffle=True,
                                  stratify=z,
                                  random_state=1)
    
    # Create whole models dataset
    all_labels = []
    all_params = []
    all_bal_accs = []
    all_auc_scores = []
    all_out_scores = []
    all_bin_scores = []

    datasets_dir = os.path.join(cwd, 'privacy', 'results', 'datasets',
                                model_name, scaler_type, model_type)
    os.makedirs(datasets_dir, exist_ok=True)
    
    data_dir = os.path.join(datasets_dir, 'params_multilabel.pt')
    if not os.path.exists(data_dir):
        models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                  model_name, scaler_type, model_type)
        
        for comb in os.listdir(models_dir):
            print(comb)
            aux_labels = [int(n) - 1 for n in comb.split('_')]
            one_hot_label = torch.zeros(len(datasets))
            one_hot_label[aux_labels] = 1
            
            comb_dir = os.path.join(models_dir, comb)
            for C in [0.1, 1.0, 10.0]:
                for l1 in [0.0, 0.5, 1.0]:
                    model_class, param_dict = create_lr_model(l1, C)
                    model = model_class(**param_dict)
                    
                    for i in range(100):
                        params_file = f'{C}_{l1}_{i}_resc_params.pkl'
                        bal_accs_file = f'{C}_{l1}_{i}_bal_accs.json'
                        auc_scores_file = f'{C}_{l1}_{i}_auc_scores.json'
                        
                        # params
                        params_dir = os.path.join(comb_dir, params_file)
                        if not os.path.exists(params_dir):
                            continue
                        
                        params = joblib.load(params_dir)
                        
                        model.coef_ = params[:-1].unsqueeze(0).numpy()
                        model.intercept_ = params[-1:].numpy()
                        model.classes_ = np.array([0, 1])
                            
                        all_labels.append(one_hot_label.clone())
                        all_params.append(params.clone())
                        
                        # bal accs
                        bal_accs_dir = os.path.join(comb_dir, bal_accs_file)
                        with open(bal_accs_dir, 'r') as f:
                            bal_accs_dict = json.load(f)
                        
                        bal_accs_vector = torch.zeros(len(datasets) + 1)
                        for dat_id in datasets_ids:
                            bal_accs_vector[dat_id - 1] = bal_accs_dict[str(dat_id)]
                        bal_accs_vector[-1] = bal_accs_dict['all']
                        
                        all_bal_accs.append(bal_accs_vector)
                        
                        # auc scores
                        auc_scores_dir = os.path.join(comb_dir, auc_scores_file)
                        with open(auc_scores_dir, 'r') as f:
                            auc_scores_dict = json.load(f)
                        
                        auc_scores_vector = torch.zeros(len(datasets) + 1)
                        for dat_id in datasets_ids:
                            auc_scores_vector[dat_id - 1] = auc_scores_dict[str(dat_id)]
                        auc_scores_vector[-1] = auc_scores_dict['all']
                        
                        all_auc_scores.append(auc_scores_vector)
                        
                        # out scores
                        out_scores = model.predict_proba(x_test)[:, 1]
                        out_scores = torch.from_numpy(out_scores).float()
                        all_out_scores.append(out_scores)
                        
                        # bin scores
                        bin_scores = discretize(out_scores, n_bins=2)
                        all_bin_scores.append(bin_scores)

        all_labels = torch.stack(all_labels, dim=0).int()
        all_params = torch.stack(all_params, dim=0)
        all_bal_accs = torch.stack(all_bal_accs, dim=0)
        all_auc_scores = torch.stack(all_auc_scores, dim=0)
        all_out_scores = torch.stack(all_out_scores, dim=0)
        all_bin_scores = torch.stack(all_bin_scores, dim=0)
        
        data = {
            'labels': all_labels,
            'params': all_params,
            'bal_accs': all_bal_accs,
            'auc_scores': all_auc_scores,
            'out_scores': all_out_scores,
            'bin_scores': all_bin_scores
        }

        torch.save(data, data_dir)
    
    else:
        data = torch.load(data_dir, weights_only=True)
    
    return data


def create_datasets_dp(model_name, scaler_type, model_type):
    # Load data
    all_data = load_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    test_size = 100
    
    x = all_data[featuresNA].values
    z = all_data['DatasetNum'].values
    
    _, x_test, = train_test_split(x,
                                  test_size=test_size,
                                  shuffle=True,
                                  stratify=z,
                                  random_state=1)
    
    # Create whole models dataset
    all_labels = []
    all_params = []
    all_bal_accs = []
    all_auc_scores = []
    all_out_scores = []
    all_bin_scores = []

    datasets_dir = os.path.join(cwd, 'privacy', 'results', 'datasets',
                                model_name, scaler_type, model_type)
    os.makedirs(datasets_dir, exist_ok=True)
    
    epsilon_list = [0.01, 0.1, 1.0, 10.0, 100.0, float('inf')]
    for epsilon in epsilon_list:
        data_dir_eps = os.path.join(datasets_dir, f'params_multilabel_{epsilon}.pt')
        if not os.path.exists(data_dir_eps):
            models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                      model_name, scaler_type, model_type)
            
            model_class, param_dict = create_lr_dp_model(epsilon)
            model = model_class(**param_dict)
            
            # Create whole models dataset, for each epsilon
            all_labels_eps = []
            all_params_eps = []
            all_bal_accs_eps = []
            all_auc_scores_eps = []
            all_out_scores_eps = []
            all_bin_scores_eps = []
            
            for comb in os.listdir(models_dir):
                print(epsilon, comb)
                aux_labels = [int(n) - 1 for n in comb.split('_')]
                one_hot_label = torch.zeros(len(datasets))
                one_hot_label[aux_labels] = 1
                
                comb_dir = os.path.join(models_dir, comb)
                for i in range(100):
                    params_file = f'{epsilon}_{i}_resc_params.pkl'
                    bal_accs_file = f'{epsilon}_{i}_bal_accs.json'
                    auc_scores_file = f'{epsilon}_{i}_auc_scores.json'
                    
                    # params
                    params_dir = os.path.join(comb_dir, params_file)
                    if not os.path.exists(params_dir):
                        continue
                    
                    params = joblib.load(params_dir)
                    
                    model.coef_ = params[:-1].unsqueeze(0).numpy()
                    model.intercept_ = params[-1:].numpy()
                    model.classes_ = np.array([0, 1])
                        
                    all_labels_eps.append(one_hot_label.clone())
                    all_params_eps.append(params.clone())
                    
                    # bal accs
                    bal_accs_dir = os.path.join(comb_dir, bal_accs_file)
                    with open(bal_accs_dir, 'r') as f:
                        bal_accs_dict = json.load(f)
                    
                    bal_accs_vector = torch.zeros(len(datasets) + 1)
                    for dat_id in datasets_ids:
                        bal_accs_vector[dat_id - 1] = bal_accs_dict[str(dat_id)]
                    bal_accs_vector[-1] = bal_accs_dict['all']
                    
                    all_bal_accs_eps.append(bal_accs_vector)
                    
                    # auc scores
                    auc_scores_dir = os.path.join(comb_dir, auc_scores_file)
                    with open(auc_scores_dir, 'r') as f:
                        auc_scores_dict = json.load(f)
                    
                    auc_scores_vector = torch.zeros(len(datasets) + 1)
                    for dat_id in datasets_ids:
                        auc_scores_vector[dat_id - 1] = auc_scores_dict[str(dat_id)]
                    auc_scores_vector[-1] = auc_scores_dict['all']
                    
                    all_auc_scores_eps.append(auc_scores_vector)
                    
                    # out scores
                    out_scores = model.predict_proba(x_test)[:, 1]
                    out_scores = torch.from_numpy(out_scores).float()
                    all_out_scores_eps.append(out_scores)
                    
                    # bin scores
                    bin_scores = discretize(out_scores, n_bins=2)
                    all_bin_scores_eps.append(bin_scores)

            all_labels_eps = torch.stack(all_labels_eps, dim=0).int()
            all_params_eps = torch.stack(all_params_eps, dim=0)
            all_bal_accs_eps = torch.stack(all_bal_accs_eps, dim=0)
            all_auc_scores_eps = torch.stack(all_auc_scores_eps, dim=0)
            all_out_scores_eps = torch.stack(all_out_scores_eps, dim=0)
            all_bin_scores_eps = torch.stack(all_bin_scores_eps, dim=0)
            
            data_eps = {
                'labels': all_labels_eps,
                'params': all_params_eps,
                'bal_accs': all_bal_accs_eps,
                'auc_scores': all_auc_scores_eps,
                'out_scores': all_out_scores_eps,
                'bin_scores': all_bin_scores_eps
            }

            torch.save(data_eps, data_dir_eps)
    
        else:
            data_eps = torch.load(data_dir_eps, weights_only=True)
        
        all_labels.append(data_eps['labels'])
        all_params.append(data_eps['params'])
        all_bal_accs.append(data_eps['bal_accs'])
        all_auc_scores.append(data_eps['auc_scores'])
        all_out_scores.append(data_eps['out_scores'])
        all_bin_scores.append(data_eps['bin_scores'])
        
    data = {
        'labels': all_labels,
        'params': all_params,
        'bal_accs': all_bal_accs,
        'auc_scores': all_auc_scores,
        'out_scores': all_out_scores,
        'bin_scores': all_bin_scores
    }
    
    return data


def create_datasets_tt(model_name, scaler_type, model_type):
    # Load data
    all_data = load_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    test_size = 100
    
    x = all_data[featuresNA].values
    z = all_data['DatasetNum'].values
    
    _, x_test, = train_test_split(x,
                                  test_size=test_size,
                                  shuffle=True,
                                  stratify=z,
                                  random_state=1)
    
    xt_test = torch.from_numpy(x_test).float()
    
    # Create whole models dataset
    all_labels = []
    all_cores = []
    all_lr_params = []
    all_bal_accs = []
    all_auc_scores = []
    all_out_scores = []
    all_bin_scores = []

    datasets_dir = os.path.join(cwd, 'privacy', 'results', 'datasets',
                                model_name, scaler_type, model_type)
    os.makedirs(datasets_dir, exist_ok=True)
    
    n_bins_list = [2, 4, 6, 10]
    for n_bins in n_bins_list:
        data_dir_bins = os.path.join(datasets_dir, f'params_multilabel_{n_bins}.pt')
        if not os.path.exists(data_dir_bins):
            models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                      model_name, scaler_type, model_type)
            
            # Create whole models dataset, for each n_bins
            all_labels_bins = []
            all_cores_bins = []
            all_lr_params_bins = []
            all_bal_accs_bins = []
            all_auc_scores_bins = []
            all_out_scores_bins = []
            all_bin_scores_bins = []
            
            for comb in os.listdir(models_dir):
                print(comb)
                aux_labels = [int(n) - 1 for n in comb.split('_')]
                one_hot_label = torch.zeros(len(datasets))
                one_hot_label[aux_labels] = 1
                
                comb_dir = os.path.join(models_dir, comb)
                for C in [0.1, 1.0, 10.0]:
                    for l1 in [0.0, 0.5, 1.0]:
                        for i in range(100):
                            cores_file = f'{n_bins}_{C}_{l1}_{i}_resc_cores.pt'
                            bal_accs_file = f'{n_bins}_{C}_{l1}_{i}_bal_accs.json'
                            auc_scores_file = f'{n_bins}_{C}_{l1}_{i}_auc_scores.json'
                            
                            # cores
                            cores_dir = os.path.join(comb_dir, cores_file)
                            if not os.path.exists(cores_dir):
                                continue
                            
                            cores = torch.load(cores_dir, weights_only=True)
                            mps = tk.models.MPSLayer(tensors=cores)
                            
                            def embedding(data):
                                out = tk.embeddings.poly(
                                    data,
                                    degree=mps.phys_dim[0] - 1).float()
                                return out
                            
                            @torch.no_grad()
                            def tt_model(data):
                                return mps(embedding(data)).pow(2)
                            
                            flat_cores = [c.flatten() for c in cores]
                            flat_cores = torch.cat(flat_cores, dim=0)
                                
                            all_labels_bins.append(one_hot_label.clone())
                            all_cores_bins.append(flat_cores.clone())
                            
                            # bal accs
                            bal_accs_dir = os.path.join(comb_dir, bal_accs_file)
                            with open(bal_accs_dir, 'r') as f:
                                bal_accs_dict = json.load(f)
                            
                            bal_accs_vector = torch.zeros(len(datasets) + 1)
                            for dat_id in datasets_ids:
                                bal_accs_vector[dat_id - 1] = bal_accs_dict[str(dat_id)]
                            bal_accs_vector[-1] = bal_accs_dict['all']
                            
                            all_bal_accs_bins.append(bal_accs_vector)
                            
                            # auc scores
                            auc_scores_dir = os.path.join(comb_dir, auc_scores_file)
                            with open(auc_scores_dir, 'r') as f:
                                auc_scores_dict = json.load(f)
                            
                            auc_scores_vector = torch.zeros(len(datasets) + 1)
                            for dat_id in datasets_ids:
                                auc_scores_vector[dat_id - 1] = auc_scores_dict[str(dat_id)]
                            auc_scores_vector[-1] = auc_scores_dict['all']
                            
                            all_auc_scores_bins.append(auc_scores_vector)
                            
                            # out scores
                            out_scores = tt_model(xt_test)[:, 1]
                            all_out_scores_bins.append(out_scores)
                            
                            # bin scores
                            bin_scores = discretize(out_scores, n_bins=2)
                            all_bin_scores_bins.append(bin_scores)
                            
                            # lr coeffs from tt
                            n_features = len(mps.in_features)
                            coeffs = []
                            yvals = []
                            
                            # coefficients
                            for i in range(n_features + 1):
                                cond_data = torch.zeros(1, n_features)
                                if i > 0:
                                    cond_data[0, i - 1] = 1
                                    
                                result = tt_model(cond_data).detach()
                                score = result / result.sum(dim=1, keepdim=True)
                                score = score[0, 1]
                                logit = (score / (1 - score)).log()
                                
                                yvals.append(logit)
                                
                                coeff = yvals[-1] - yvals[0]
                                coeffs.append(coeff)
                            
                            # intercept
                            coeffs.append(yvals[0])
                            
                            all_lr_params_bins.append(torch.stack(coeffs, dim=0))

            all_labels_bins = torch.stack(all_labels_bins, dim=0).int()
            all_cores_bins = torch.stack(all_cores_bins, dim=0).detach()
            all_lr_params_bins = torch.stack(all_lr_params_bins, dim=0)
            all_bal_accs_bins = torch.stack(all_bal_accs_bins, dim=0)
            all_auc_scores_bins = torch.stack(all_auc_scores_bins, dim=0)
            all_out_scores_bins = torch.stack(all_out_scores_bins, dim=0)
            all_bin_scores_bins = torch.stack(all_bin_scores_bins, dim=0)
            
            data_bins = {
                'labels': all_labels_bins,
                'cores': all_cores_bins,
                'lr_params': all_lr_params_bins,
                'bal_accs': all_bal_accs_bins,
                'auc_scores': all_auc_scores_bins,
                'out_scores': all_out_scores_bins,
                'bin_scores': all_bin_scores_bins
            }

            torch.save(data, data_dir_bins)
        
        else:
            data = torch.load(data_dir_bins, weights_only=True)
        
        all_labels.append(data_bins['labels'])
        all_cores.append(data_bins['cores'])
        all_lr_params.append(data_bins['lr_params'])
        all_bal_accs.append(data_bins['bal_accs'])
        all_auc_scores.append(data_bins['auc_scores'])
        all_out_scores.append(data_bins['out_scores'])
        all_bin_scores.append(data_bins['bin_scores'])
        
    data = {
        'labels': all_labels,
        'cores': all_cores,
        'lr_params': all_lr_params,
        'bal_accs': all_bal_accs,
        'auc_scores': all_auc_scores,
        'out_scores': all_out_scores,
        'bin_scores': all_bin_scores
    }
    
    return data


def bb_attack(scores, labels, model_name, scaler_type, model_type, attack_name):
    X, y = scores, labels
    
    attack_model_dir = os.path.join(cwd, 'privacy', 'results', 'attacks', 'bb',
                                    model_name, scaler_type, model_type)
    os.makedirs(attack_model_dir, exist_ok=True)
    
    n_splits = 5
    n_repeats = 5
    repkfold = RepeatedKFold(n_splits=n_splits,
                             n_repeats=n_repeats,
                             random_state=1)
    
    for fold, (train_idx, test_idx) in enumerate(repkfold.split(X, y)):
        print(f'Training repeated fold {fold+1}/{n_splits * n_repeats}...')
        
        model_file = f'mlp_bb_{attack_name}_{fold+1}.pkl'
        if os.path.exists(os.path.join(attack_model_dir, model_file)):
            continue
        
        # Train/Test Split (held-out test set)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Save held-out test set (optional)
        joblib.dump((X_test, y_test),
                    os.path.join(attack_model_dir,
                                 f'test_set_bb_{attack_name}_{fold+1}.pkl'))
        
        # Define and train the MLP model
        mlp_bb = MLPClassifier(hidden_layer_sizes=(32, 16, 8),
                               activation='relu',
                               solver='adam',
                               max_iter=100)
        mlp_bb.fit(X_train, y_train)

        # Save model
        joblib.dump(mlp_bb, os.path.join(attack_model_dir, model_file))


def bb_attack_dp(scores, labels, model_name, scaler_type, model_type, attack_name):
    epsilon_list = [0.01, 0.1, 1.0, 10.0, 100.0, float('inf')]
    for eps_id, epsilon in enumerate(epsilon_list):
        print(epsilon)
        X, y = scores[eps_id], labels[eps_id]
        
        attack_model_dir = os.path.join(cwd, 'privacy', 'results', 'attacks',
                                        'bb', model_name, scaler_type, model_type)
        os.makedirs(attack_model_dir, exist_ok=True)
        
        n_splits = 5
        n_repeats = 5
        repkfold = RepeatedKFold(n_splits=n_splits,
                                 n_repeats=n_repeats,
                                 random_state=1)
        
        for fold, (train_idx, test_idx) in enumerate(repkfold.split(X, y)):
            print(f'Training repeated fold {fold+1}/{n_splits * n_repeats}...')
            
            model_file = f'mlp_bb_{attack_name}_{epsilon}_{fold+1}.pkl'
            if os.path.exists(os.path.join(attack_model_dir, model_file)):
                continue
            
            # Train/Test Split (held-out test set)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Save held-out test set (optional)
            joblib.dump((X_test, y_test),
                        os.path.join(
                            attack_model_dir,
                            f'test_set_bb_{attack_name}_{epsilon}_{fold+1}.pkl'))
            
            # Define and train the MLP model
            mlp_bb = MLPClassifier(hidden_layer_sizes=(32, 16, 8),
                                   activation='relu',
                                   solver='adam',
                                   max_iter=100)
            mlp_bb.fit(X_train, y_train)

            # Save model
            joblib.dump(mlp_bb, os.path.join(attack_model_dir, model_file))


def bb_attack_tt(scores, labels, model_name, scaler_type, model_type, attack_name):
    n_bins_list = [2, 4, 6, 10]
    for bins_id, n_bins in enumerate(n_bins_list):
        print(n_bins)
        X, y = scores[bins_id], labels[bins_id]
        
        attack_model_dir = os.path.join(cwd, 'privacy', 'results', 'attacks',
                                        'bb', model_name, scaler_type, model_type)
        os.makedirs(attack_model_dir, exist_ok=True)
        
        n_splits = 5
        n_repeats = 5
        repkfold = RepeatedKFold(n_splits=n_splits,
                                 n_repeats=n_repeats,
                                 random_state=1)
        
        for fold, (train_idx, test_idx) in enumerate(repkfold.split(X, y)):
            print(f'Training repeated fold {fold+1}/{n_splits * n_repeats}...')
            
            model_file = f'mlp_bb_{attack_name}_{n_bins}_{fold+1}.pkl'
            if os.path.exists(os.path.join(attack_model_dir, model_file)):
                continue
            
            # Train/Test Split (held-out test set)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Save held-out test set (optional)
            joblib.dump((X_test, y_test),
                        os.path.join(
                            attack_model_dir,
                            f'test_set_bb_{attack_name}_{n_bins}_{fold+1}.pkl'))
            
            # Define and train the MLP model
            mlp_bb = MLPClassifier(hidden_layer_sizes=(32, 16, 8),
                                   activation='relu',
                                   solver='adam',
                                   max_iter=100)
            mlp_bb.fit(X_train, y_train)

            # Save model
            joblib.dump(mlp_bb, os.path.join(attack_model_dir, model_file))


def wb_attack(params, labels, model_name, scaler_type, model_type):
    X, y = params, labels
    
    # Avoid NaNs and infs in lr_tt params
    mask = ~X.isnan().any(dim=1)
    mask = mask * ~X.isinf().any(dim=1)
    X, y = X[mask], y[mask]
    
    attack_model_dir = os.path.join(cwd, 'privacy', 'results', 'attacks', 'wb',
                                    model_name, scaler_type, model_type)
    os.makedirs(attack_model_dir, exist_ok=True)
    
    n_splits = 5
    n_repeats = 5
    repkfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                             random_state=1)
    
    for rfold, (rtrain_idx, rtest_idx) in enumerate(repkfold.split(X, y)):
        print(f'Training repeated fold {rfold+1}/{n_splits * n_repeats}...')
        
        # First: Train/Test Split (held-out test set)
        X_train_all, y_train_all = X[rtrain_idx], y[rtrain_idx]
        X_test, y_test = X[rtest_idx], y[rtest_idx]

        # Save held-out test set (optional)
        joblib.dump((X_test, y_test),
                    os.path.join(attack_model_dir,
                                 f'test_set_wb_{rfold+1}.pkl'))

        # K-Fold CV on training set
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, _) in enumerate(kfold.split(X_train_all,
                                                          y_train_all)):
            print(f'\tTraining fold {fold+1}/{n_splits}...')
            
            model_file = f'mlp_wb_fold_{fold+1}_{rfold+1}.pkl'
            if os.path.exists(os.path.join(attack_model_dir, model_file)):
                continue

            X_train = X_train_all[train_idx]
            y_train = y_train_all[train_idx]

            # Define and train the model
            mlp_wb = MLPClassifier(hidden_layer_sizes=(32, 16, 8),
                                   activation='relu',
                                   solver='adam',
                                   max_iter=100)
            mlp_wb.fit(X_train, y_train)

            # Save model for this fold
            joblib.dump(mlp_wb, os.path.join(attack_model_dir, model_file))


def wb_attack_dp(params, labels, model_name, scaler_type, model_type):
    epsilon_list = [0.01, 0.1, 1.0, 10.0, 100.0, float('inf')]
    for eps_id, epsilon in enumerate(epsilon_list):
        print(epsilon)
        X, y = params[eps_id], labels[eps_id]
        
        attack_model_dir = os.path.join(cwd, 'privacy', 'results', 'attacks',
                                        'wb', model_name, scaler_type, model_type)
        os.makedirs(attack_model_dir, exist_ok=True)
        
        n_splits = 5
        n_repeats = 5
        repkfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=1)
        
        for rfold, (rtrain_idx, rtest_idx) in enumerate(repkfold.split(X, y)):
            print(f'Training repeated fold {rfold+1}/{n_splits * n_repeats}...')
            
            # First: Train/Test Split (held-out test set)
            X_train_all, y_train_all = X[rtrain_idx], y[rtrain_idx]
            X_test, y_test = X[rtest_idx], y[rtest_idx]

            # Save held-out test set (optional)
            joblib.dump((X_test, y_test),
                        os.path.join(attack_model_dir,
                                     f'test_set_wb_{epsilon}_{rfold+1}.pkl'))

            # K-Fold CV on training set
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (train_idx, _) in enumerate(kfold.split(X_train_all,
                                                              y_train_all)):
                print(f'\tTraining fold {fold+1}/{n_splits}...')
                
                model_file = f'mlp_wb_fold_{fold+1}_{epsilon}_{rfold+1}.pkl'
                if os.path.exists(os.path.join(attack_model_dir, model_file)):
                    continue

                X_train = X_train_all[train_idx]
                y_train = y_train_all[train_idx]

                # Define and train the model
                mlp_wb = MLPClassifier(hidden_layer_sizes=(32, 16, 8),
                                       activation='relu',
                                       solver='adam',
                                       max_iter=100)
                mlp_wb.fit(X_train, y_train)

                # Save model for this fold
                joblib.dump(mlp_wb, os.path.join(attack_model_dir, model_file))


def wb_attack_tt(params, labels, model_name, scaler_type, model_type):
    n_bins_list = [2, 4, 6, 10]
    for bins_id, n_bins in enumerate(n_bins_list):
        print(n_bins)
        X, y = params[bins_id], labels[bins_id]
        
        attack_model_dir = os.path.join(cwd, 'privacy', 'results', 'attacks',
                                        'wb', model_name, scaler_type, model_type)
        os.makedirs(attack_model_dir, exist_ok=True)
        
        n_splits = 5
        n_repeats = 5
        repkfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=1)
        
        for rfold, (rtrain_idx, rtest_idx) in enumerate(repkfold.split(X, y)):
            print(f'Training repeated fold {rfold+1}/{n_splits * n_repeats}...')
            
            # First: Train/Test Split (held-out test set)
            X_train_all, y_train_all = X[rtrain_idx], y[rtrain_idx]
            X_test, y_test = X[rtest_idx], y[rtest_idx]

            # Save held-out test set (optional)
            joblib.dump((X_test, y_test),
                        os.path.join(attack_model_dir,
                                     f'test_set_wb_{n_bins}_{rfold+1}.pkl'))

            # K-Fold CV on training set
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (train_idx, _) in enumerate(kfold.split(X_train_all,
                                                              y_train_all)):
                print(f'\tTraining fold {fold+1}/{n_splits}...')
                
                model_file = f'mlp_wb_fold_{fold+1}_{n_bins}_{rfold+1}.pkl'
                if os.path.exists(os.path.join(attack_model_dir, model_file)):
                    continue

                X_train = X_train_all[train_idx]
                y_train = y_train_all[train_idx]

                # Define and train the model
                mlp_wb = MLPClassifier(hidden_layer_sizes=(32, 16, 8),
                                       activation='relu',
                                       solver='adam',
                                       max_iter=100)
                mlp_wb.fit(X_train, y_train)

                # Save model for this fold
                joblib.dump(mlp_wb, os.path.join(attack_model_dir, model_file))


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--vanilla\n'
              '\t--average\n'
              '\t--bb\n'
              '\t--wb')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h',
                                   ['help',
                                    'vanilla', 'average',
                                    'bb', 'wb'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--vanilla\n'
              '\t--average\n'
              '\t--bb\n'
              '\t--wb')
        sys.exit(2)
    
    # Save selected options
    options = {'vanilla': False,
               'average': False,
               'bb': False,
               'wb': False}
    
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--help, -h\n'
                  '\t--vanilla\n'
                  '\t--average\n'
                  '\t--bb\n'
                  '\t--wb')
            sys.exit()
        elif opt == '--vanilla':
            options['vanilla'] = True
        elif opt == '--average':
            options['average'] = True
        elif opt == '--bb':
            options['bb'] = True
        elif opt == '--wb':
            options['wb'] = True
    
    # Check if selected options are compatible
    if options['vanilla'] and options['average']:
        print('Options "vanilla" and "average" are incompatible')
        sys.exit()
    elif not (options['vanilla'] or options['average']):
        print('One of the options "vanilla" and "average" should be chosen')
        sys.exit()
    
    if options['bb'] and options['wb']:
        print('Options "bb" and "wb" are incompatible')
        sys.exit()
    elif not (options['bb'] or options['wb']):
        print('One of the options "bb" and "wb" should be chosen')
        sys.exit()
    
    if len(args) == 1:
        model_name = args[0]
    else:
        print('<model_name> argument should be provided. It can be one of: '
              '"lr", "lr_priv", "lr_dp" or "tt_lr"')
        sys.exit()
    
    if model_name not in ['lr', 'lr_priv', 'lr_dp', 'tt_lr']:
        raise ValueError('<model_name> argument should be one of: '
                         '"lr", "lr_priv", "lr_dp" or "tt_lr"')
    
    scaler_type = 'standard'
    model_type = 'vanilla' if options['vanilla'] else 'average'
    attack_type = 'bb' if options['bb'] else 'wb'
    
    print('\n* Creating datasets...')
    if model_name in ['lr', 'lr_priv']:
        data = create_datasets_lr(model_name=model_name,
                                  scaler_type=scaler_type,
                                  model_type=model_type)
    elif model_name == 'lr_dp':
        data = create_datasets_dp(model_name=model_name,
                                  scaler_type=scaler_type,
                                  model_type=model_type)
    elif model_name == 'tt_lr':
        data = create_datasets_tt(model_name=model_name,
                                  scaler_type=scaler_type,
                                  model_type=model_type)
    
    print('\n* Performing attacks...')
    if attack_type == 'bb':
        if model_name in ['lr', 'lr_priv']:
            bb_attack(scores=data['bin_scores'], # data['bal_accs']
                      labels=data['labels'],
                      model_name=model_name,
                      scaler_type=scaler_type,
                      model_type=model_type,
                      attack_name='weak')
            bb_attack(scores=data['out_scores'], # data['auc_scores']
                      labels=data['labels'],
                      model_name=model_name,
                      scaler_type=scaler_type,
                      model_type=model_type,
                      attack_name='strong')
        elif model_name == 'lr_dp':
            bb_attack_dp(scores=data['bin_scores'], # data['bal_accs']
                         labels=data['labels'],
                         model_name=model_name,
                         scaler_type=scaler_type,
                         model_type=model_type,
                         attack_name='weak')
            bb_attack_dp(scores=data['out_scores'], # data['auc_scores']
                         labels=data['labels'],
                         model_name=model_name,
                         scaler_type=scaler_type,
                         model_type=model_type,
                         attack_name='strong')
        elif model_name == 'tt_lr':
            bb_attack_tt(scores=data['bin_scores'], # data['bal_accs']
                         labels=data['labels'],
                         model_name=model_name,
                         scaler_type=scaler_type,
                         model_type=model_type,
                         attack_name='weak')
            bb_attack_tt(scores=data['out_scores'], # data['auc_scores']
                         labels=data['labels'],
                         model_name=model_name,
                         scaler_type=scaler_type,
                         model_type=model_type,
                         attack_name='strong')
    else:
        if model_name in ['lr', 'lr_priv']:
            wb_attack(params=data['params'],
                      labels=data['labels'],
                      model_name=model_name,
                      scaler_type=scaler_type,
                      model_type=model_type)
        elif model_name == 'lr_dp':
            wb_attack_dp(params=data['params'],
                         labels=data['labels'],
                         model_name=model_name,
                         scaler_type=scaler_type,
                         model_type=model_type)
        elif model_name == 'tt_lr':
            wb_attack_tt(params=data['cores'],
                         labels=data['labels'],
                         model_name=model_name,
                         scaler_type=scaler_type,
                         model_type=model_type)
            wb_attack_tt(params=data['lr_params'],
                         labels=data['labels'],
                         model_name='lr_' + model_name,
                         scaler_type=scaler_type,
                         model_type=model_type)
