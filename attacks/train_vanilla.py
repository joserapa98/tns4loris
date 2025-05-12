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

def create_model(model_name, l1, C):
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
            'l1_ratio': l1, #0.5, #1,
            'class_weight': 'balanced',
            'C': C, #1, #0.1
        }
    
    return model_type, param_dict
    
    # model = model_type(**param_dict)
    # return model

def load_data(featuresNA, phenoNA, datasets, datasets_ids, scaler_type):
    data_dir = os.path.join(cwd, '..', '02.Input')
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
    
    # all_data = dataScaler(data_no_nans, all_features, featuresNA, scaler_type)
    all_data = data_no_nans
    
    return all_data


if __name__ == '__main__':
    args = sys.argv[1:]
    
    # NOTE: We're going to use llr6 only
    model_name = 'llr6'
    
    if len(args) == 5:
        n_splits = int(args[0])   # 5
        n_repeats = int(args[1])  # 20
        scaler_type = args[2]
        l1 = float(args[3])       # [0., 0.5, 1.] -> 1.
        C = float(args[4])        # [0.1, 1, 10] -> 0.1
    else:
        print('The following arguments should be passed:\n'
              '\t1) <n_splits> => number of splits\n'
              '\t2) <n_repeats> => number of repeats of K-fold\n'
              '\t3) <scaler_type> => standard / minmax scaler\n'
              '\t4) <l1> => l1 regularization weight\n'
              '\t5) <C> => inverse of total regularization weight\n')
        sys.exit()
    
    if scaler_type not in ['standard', 'minmax']:
        print(print('Scaler should be "standard" or "minmax"'))
        sys.exit()
    
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
    
    # NOTE: We should use Standard for loris and MinMax maybe later for tensorized models
    # NOTE: Nope, tensorization also works well for Standard if we use more
    # sketch samples, so we will use always Standard
    # scaler_type = 'MinMax'
    # scaler_type = 'MinMax' if scaler_type == 'minmax' else 'StandardScaler'
    all_data = load_data(featuresNA, phenoNA, datasets, datasets_ids, scaler_type)
    
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    # Train
    models_dir = os.path.join(cwd, 'models', model_name, scaler_type, 'vanilla')
    os.makedirs(models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(comb_dir, exist_ok=True)
            
        model_type, param_dict = create_model(model_name, l1, C)

        # Define repeated k-fold cross-validation
        kf = RepeatedStratifiedKFold(n_splits=n_splits,
                                     n_repeats=n_repeats)

        # Store results
        for i, (train_idx, _) in enumerate(kf.split(x, y_z)):
            x_train = x[train_idx]
            y_train = y[train_idx]
            z_train = z[train_idx]
            
            train_idx_comb = np.isin(z_train, comb)
            x_train = x_train[train_idx_comb]
            y_train = y_train[train_idx_comb]
            
            # Rescale
            scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
            x_train = scaler.fit_transform(x_train)
            
            # print(comb)
            counter = Counter(y_train)
            # print(counter)
            
            # Train the model
            model = model_type(**param_dict)
            model.fit(x_train, y_train)
            
            # Save model's parameters
            if model_name == 'nn2':
                coefs = [torch.from_numpy(c).flatten()
                            for c in model.coefs_]
                coefs = torch.cat(coefs)
                intercepts = [torch.from_numpy(c).flatten()
                                for c in model.intercepts_]
                intercepts = torch.cat(intercepts)
            else: #if model_name == 'llr6':
                coefs = torch.from_numpy(model.coef_).flatten()
                intercepts = torch.from_numpy(model.intercept_).flatten()
            
            params = torch.cat([coefs, intercepts])
            
            # Weight and range to rescale params
            if model_name == 'llr6':
                if scaler_type == "standard":
                    scaler_info = {
                        'mean': scaler.mean_,
                        'scale': scaler.scale_
                    }
                elif scaler_type == "minmax":
                    scaler_info = {
                        'mean': scaler.data_min_,
                        'scale': scaler.data_range_
                    }
                else:
                    raise ValueError(
                        'scaler_type must be "standard" or "minmax"')
            else:
                raise ValueError(
                    'model_name should be "llr6"')
            
            mean = torch.from_numpy(scaler_info['mean'])
            scale = torch.from_numpy(scaler_info['scale'])
        
            # Split coefficients and intercept
            coefs = params[:-1]
            intercept = params[-1]

            # Rescale coefs
            new_coefs = coefs / scale  # element-wise division

            # Adjust intercept
            intercept_shift = torch.sum(coefs * mean / scale)
            new_intercept = intercept - intercept_shift

            # Concatenate back
            new_coefs_tensor = torch.cat([new_coefs, new_intercept.unsqueeze(0)])
            params = new_coefs_tensor
            
            # Save model's parameters
            params_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_params.pkl')
            joblib.dump(params, params_dir)

            # Evaluate final model by dataset
            if model_name == 'llr6':
                model.coef_ = params[:-1].numpy()
                model.intercept_ = params[-1].numpy()
            else:
                raise ValueError(
                    'model_name should be "llr6"')
            
            bal_accs = {}
            auc_scores = {}
            results = {}
            
            for dat_id in datasets_ids:
                idx = z == dat_id
                y_proba = model.predict_proba(x[idx])
                y_pred = model.predict(x[idx])
                
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
            
            y_proba = model.predict_proba(x)
            y_pred = model.predict(x)
            bacc = balanced_accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_proba[:, 1])
            
            bal_accs['all'] = bacc
            auc_scores['all'] = auc
            
            print(bal_accs)
            print(auc_scores)
            print()
            
            # Save scores
            bal_accs_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_bal_accs.json')
            with open(bal_accs_dir, 'w') as file:
                json.dump(bal_accs, file, indent=4)
            
            auc_scores_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_auc_scores.json')
            with open(auc_scores_dir, 'w') as file:
                json.dump(auc_scores, file, indent=4)
            
            # Save results
            results_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_results.pkl')
            joblib.dump(results, results_dir)
