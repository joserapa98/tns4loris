# Run script from parent folder "tns4loris"

import sys
import os
import getopt
import copy
import joblib
import json

from collections import Counter
from itertools import chain, combinations

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import pandas as pd
import numpy as np
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss
from tensorkrowch.utils import random_unitary


cwd = os.getcwd()

if torch.cuda.is_available():
    print('GPU is available')
    device = torch.device('cuda')
else:
    print('GPU is not available')
    device = torch.device('cpu')


def all_combinations(lst):
    return list(chain.from_iterable(combinations(lst, r)
                                    for r in range(1, len(lst) + 1)))


def create_model(model_name, l1, C):
    if model_name == 'nn2':
        model_class = MLPClassifier
        param_dict = {
            'max_iter': 100,
            'hidden_layer_sizes': (19, 19),
            'activation': 'tanh',
            'alpha': 1e-05,
            'early_stopping': False
        }
        
    elif model_name == 'llr6':
        model_class = linear_model.LogisticRegression
        param_dict = {
            'solver': 'saga',
            'penalty': 'elasticnet',
            'max_iter': 100,
            'l1_ratio': l1, #0.5, #1,
            'class_weight': 'balanced',
            'C': C, #1, #0.1
        }
    
    return model_class, param_dict


def load_data(featuresNA, phenoNA, datasets, datasets_ids):
    data_dir = os.path.join(cwd, 'loris', '02.Input')
    data_file = os.path.join(data_dir, 'AllData.xlsx')

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
    
    all_features = featuresNA + [phenoNA, 'Dataset', 'DatasetNum']
    data_no_nans = data_all_raw[all_features].dropna(axis=0)
    return data_no_nans


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--vanilla\n'
              '\t--average')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help', 'vanilla', 'average'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--vanilla\n'
              '\t--average')
        sys.exit(2)
    
    # Save selected options
    options = {'vanilla': False,
               'average': False}
    
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--help, -h\n'
                  '\t--vanilla\n'
                  '\t--average')
            sys.exit()
        elif opt == '--vanilla':
            options['vanilla'] = True
        elif opt == '--average':
            options['average'] = True
    
    # Check if selected options are compatible
    if options['vanilla'] and options['average']:
        print('Options "vanilla" and "average" are incompatible')
        sys.exit()
    elif not (options['vanilla'] or options['average']):
        print('One of the options "vanilla" and "average" should be chosen')
        sys.exit()
    
    if len(args) == 2:
        l1 = float(args[0])      # [0.0, 0.5, 1.0]  -> 1.0
        C = float(args[1])       # [0.1, 1.0, 10.0] -> 0.1
    else:
        print('The following arguments should be passed:\n'
              '\t1) <l1> => l1 regularization weight\n'
              '\t2) <C> => inverse of total regularization weight\n')
    
    # NOTE: We're going to use llr6 only
    # NOTE: We should use scaler_type = "standard"
    model_name = 'llr6'
    scaler_type = 'standard'
    model_type = 'vanilla' if options['vanilla'] else 'average'
    
    # Load data
    featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age',
                  'CancerType1', 'CancerType2', 'CancerType3', 'CancerType4',
                  'CancerType5', 'CancerType6', 'CancerType7', 'CancerType8',
                  'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12',
                  'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16']
    phenoNA = 'Response'
    
    datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
                'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
    datasets_ids = list(range(1, len(datasets) + 1))
    
    all_data = load_data(featuresNA, phenoNA, datasets, datasets_ids, scaler_type)
    
    # Tensorization hyperparameters
    sketch_size    = 100
    phys_dim       = 2
    domain         = torch.linspace(0, 1, phys_dim) if scaler_type == 'minmax' else None
    bond_dim       = 2
    cum_percentage = 1. # 1 - 1e-2
    batch_size     = 500
    device         = torch.device('cpu')
    verbose        = False
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    # NOTE: We are tensorizing using samples from all datasets, should we only
    # consider samples from the training dataset of each model?
    _, x_sketch, _, y_sketch = train_test_split(x, y,
                                                test_size=sketch_size,
                                                shuffle=True,
                                                stratify=y_z,
                                                random_state=1)
    xt_sketch = torch.from_numpy(x_sketch).float()
    yt_sketch = torch.from_numpy(y_sketch)
    
    # Tensorize
    models_dir = os.path.join(cwd, 'privacy', 'models',
                              model_name, scaler_type, model_type)
    os.makedirs(models_dir, exist_ok=True)
    
    tt_models_dir = os.path.join(cwd, 'privacy', 'tt_models',
                                 model_name, scaler_type, model_type)
    os.makedirs(tt_models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(comb_dir, exist_ok=True)
        
        tt_comb_dir = os.path.join(tt_models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(tt_comb_dir, exist_ok=True)
        
        for i in range(100):
            print(comb, C, l1, i)
            
            # This is to skip previously tensorized models
            cores_dir = os.path.join(tt_comb_dir, f'{C}_{l1}_{i}_cores.pkl')
            if os.path.exists(cores_dir):
                continue
            
            params_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_params.pkl')
            params = joblib.load(params_dir)
            
            # Split into weights and intercept
            weights = params[:-1].numpy().reshape(1, -1)  # shape (1, n_features)
            intercept = np.array([params[-1].item()])     # shape (1,)
            
            model_class, param_dict = create_model(model_name, l1, C)
            
            # Initialize the model
            model = model_class(**param_dict)
            model.fit(np.zeros((2, weights.shape[1])), [0, 1])  # dummy fit

            # Manually set parameters
            model.coef_ = weights
            model.intercept_ = intercept
            
            def fn_model(data):
                result = torch.from_numpy(model.predict_proba(data)).float()
                return result
            
            # def fn_model(data):
            #     y_proba = model.predict_proba(data)
            #     y_pred = (y_proba > 0.5)
            #     result = torch.from_numpy(y_pred).float()
            #     return result

            # Tensorization
            cores = tt_rss(function=fn_model,
                           embedding=embedding,
                           sketch_samples=xt_sketch,
                           labels=yt_sketch,
                           domain_multiplier=1,
                           domain=domain,
                           rank=bond_dim,
                           cum_percentage=cum_percentage,
                           batch_size=batch_size,
                           device=device,
                           verbose=verbose)
            
            # Make all cores equal size
            tt_model = tk.models.MPSLayer(tensors=cores)
            for j in range(len(tt_model.mats_env) - 1):
                tt_model.mats_env[j]['right'].change_size(size=bond_dim)
            
            # Randomize gauge
            for j, node in enumerate(tt_model.mats_env):
                right_size = node.size('right')
                # U = random_unitary(right_size)
                U = torch.randn((right_size, right_size))
                if j < (len(tt_model.mats_env) - 1):
                    node.tensor = torch.einsum('lir,rk->lik',
                                                node.tensor, U)
                if j > 0:
                    node.tensor = torch.einsum('kl,lir->kir',
                                                prev_U, node.tensor)
                # prev_U = U.clone().H
                prev_U = torch.linalg.inv(U)
                # print(U @ prev_U)
            
            # Save model's parameters
            cores = [c.detach() for c in tt_model.tensors]
            cores_dir = os.path.join(tt_comb_dir, f'{C}_{l1}_{i}_cores.pkl')
            torch.save(cores, cores_dir)
            
            tt_model.trace(torch.zeros(1, xt_sketch.size(1), phys_dim))

            # Evaluate final model by dataset
            bal_accs = {}
            auc_scores = {}
            results = {}
            
            with torch.no_grad():
                for dat_id in datasets_ids:
                    idx = z == dat_id
                    x_aux = torch.from_numpy(x[idx]).float()
                    y_proba = tt_model(embedding(x_aux))
                    y_pred = (y_proba[:, 1] > 0.5).int()
                    
                    y_proba = y_proba.numpy()
                    y_pred = y_pred.numpy()
                    
                    bacc = balanced_accuracy_score(y[idx], y_pred)
                    bal_accs[dat_id] = bacc
                    
                    auc = roc_auc_score(y[idx], y_proba[:, 1])
                    auc_scores[dat_id] = auc
                    
                    results[dat_id] = {
                        'y_proba': y_proba,
                        'y_pred': y_pred,
                        'y_test': y[idx]
                    }
                    
                    print(dat_id)
                    counter = Counter(y[idx])
                    print(counter)
                
                x_aux = torch.from_numpy(x).float()
                y_proba = tt_model(embedding(x_aux))
                y_pred = (y_proba[:, 1] > 0.5).int()
                
                y_proba = y_proba.numpy()
                y_pred = y_pred.numpy()
                
                bacc = balanced_accuracy_score(y, y_pred)
                auc = roc_auc_score(y, y_proba[:, 1])
                
                bal_accs['all'] = bacc
                auc_scores['all'] = auc
            
            print(bal_accs)
            print(auc_scores)
            print()
            
            # Save scores
            bal_accs_dir = os.path.join(tt_comb_dir, f'{C}_{l1}_{i}_bal_accs.json')
            with open(bal_accs_dir, 'w') as file:
                json.dump(bal_accs, file, indent=4)
            
            auc_scores_dir = os.path.join(tt_comb_dir, f'{C}_{l1}_{i}_auc_scores.json')
            with open(auc_scores_dir, 'w') as file:
                json.dump(auc_scores, file, indent=4)
            
            # Save results
            results_dir = os.path.join(tt_comb_dir, f'{C}_{l1}_{i}_results.pkl')
            joblib.dump(results, results_dir)
