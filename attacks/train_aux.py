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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import balanced_accuracy_score

import pandas as pd
import numpy as np

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
            'max_iter': 500,
            'l1_ratio': l1, #1,
            'class_weight': 'balanced',
            'C': C, #0.1
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


# cwd = os.path.join(cwd, 'attacks')

# model_name = 'nn2' #'llr6'
# n_splits = 5
# n_models = 10

# # Load data
# featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age',
#               'CancerType1', 'CancerType2', 'CancerType3', 'CancerType4',
#               'CancerType5', 'CancerType6', 'CancerType7', 'CancerType8',
#               'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12',
#               'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16']
# phenoNA = 'Response'

# datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
#             'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
# datasets_ids = list(range(1, len(datasets) + 1))

# scaler_type = 'MinMax'
# all_data = load_data(featuresNA, phenoNA, datasets, datasets_ids, scaler_type)
# print(all_data['Dataset'].value_counts())

# # Train
# models_dir = os.path.join(cwd, 'models', model_name)
# os.makedirs(models_dir, exist_ok=True)

# all_combs = all_combinations(datasets_ids)

# for comb in all_combs:
#     comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
#     os.makedirs(comb_dir, exist_ok=True)
    
#     aux_data = all_data[all_data['DatasetNum'].isin(comb)]
#     x = aux_data[featuresNA].values
#     y = aux_data[phenoNA].values
#     z = aux_data['DatasetNum'].values
#     y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
#     # Define repeated k-fold cross-validation
#     kf = RepeatedStratifiedKFold(n_splits=n_splits,
#                                  n_repeats=n_models)

#     # Store results
#     for i, (train_idx, test_idx) in enumerate(kf.split(x, y_z)):
#         model = create_model(model_name)
        
#         x_train, x_test = x[train_idx], x[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#         z_train, z_test = z[train_idx], z[test_idx]
        
#         print(comb)
#         counter = Counter(y_train)
#         print(counter)
#         counter = Counter(y_test)
#         print(counter)
        
#         # Train the model
#         model.fit(x_train, y_train)

#         # Evaluate by dataset
#         scores = {}
#         results = {}
        
#         for c in comb:
#             idx = z_test == c
#             y_proba = model.predict_proba(x_test[idx])
#             y_pred = model.predict(x_test[idx])
#             bacc = balanced_accuracy_score(y_test[idx], y_pred)
#             scores[c] = {
#                 'bacc': bacc,
#             }
#             results[c] = {
#                 'y_proba': y_proba,
#                 'y_pred': y_pred,
#                 'y_test': y_test[idx]
#             }
            
#             print(c)
#             counter = Counter(y_test[idx])
#             print(counter)
        
#         y_pred = model.predict(x_test)
#         bacc = balanced_accuracy_score(y_test, y_pred)
        
#         scores['bacc'] = bacc


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--nn2\n'
              '\t--llr6\n')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h',
                                   ['help', 'nn2', 'llr6'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--nn2\n'
              '\t--llr6\n')
        sys.exit(2)
        
    # Save selected options
    options = {'nn2': False,
               'llr6': False}
        
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--nn2 => trains NeuralNetwork2\n'
                  '\t--llr6 => trains LLR6\n')
            sys.exit()
        elif opt == '--nn2':
            options['nn2'] = True
        elif opt == '--llr6':
            options['llr6'] = True
        
    # Check if selected options are compatible
    if sum(options.values()) != 1:
        print('One, and only one, of the options "nn2" or '
              '"llr6" should be selected')
        sys.exit()
    
    # NOTE: We're goig to use llr6 only
    model_name = 'nn2' if options['nn2'] else 'llr6'
    
    if len(args) == 5:
        n_splits = int(args[0])  # 5        = K
        n_models = int(args[1])  # 10 * 200 = MN
        scaler_type = args[2]
        l1 = float(args[3])      # [0., 0.5, 1.]
        C = float(args[4])       # [0.1, 1, 10]
    else:
        print('The following arguments should be passed:\n'
              '\t1) <n_splits> => number of splits\n'
              '\t2) <n_models> => number of splits\n'
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
    
    # NOTE: we should use Standard for loris and MinMax only later for tensorized models
    # NOTE: Actually, not, tensorization also works with StandardScaler using
    # scaler_type = 'MinMax'
    # scaler_type = 'MinMax' if scaler_type == 'minmax' else 'StandardScaler'
    all_data = load_data(featuresNA, phenoNA, datasets, datasets_ids, scaler_type)
    
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    # Train
    models_dir = os.path.join(cwd, 'models', model_name, scaler_type)
    os.makedirs(models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(comb_dir, exist_ok=True)
        
        model_type, param_dict = create_model(model_name, l1, C)
        model = model_type(**param_dict)

        # Define repeated k-fold cross-validation
        kf = RepeatedStratifiedKFold(n_splits=n_splits,
                                     n_repeats=n_models)

        # Store results
        for i, (train_idx, test_idx) in enumerate(kf.split(x, y_z)):
            
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            z_train, z_test = z[train_idx], z[test_idx]
            
            train_idx_comb = np.isin(z_train, comb)
            x_train = x_train[train_idx_comb]
            y_train = y_train[train_idx_comb]
            
            # Rescale
            scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            
            print(comb)
            counter = Counter(y_train)
            print(counter)
            counter = Counter(y_test)
            print(counter)
            
            # Train the model
            model.fit(x_train, y_train)

            # Evaluate by dataset
            scores = {}
            results = {}
            
            for dat_id in datasets_ids:
                idx = z_test == dat_id
                y_proba = model.predict_proba(x_test[idx])
                y_pred = model.predict(x_test[idx])
                bacc = balanced_accuracy_score(y_test[idx], y_pred)
                scores[dat_id] = bacc
                results[dat_id] = {
                    'y_proba': y_proba,
                    'y_pred': y_pred,
                    'y_test': y_test[idx]
                }
                
                print(dat_id)
                counter = Counter(y_test[idx])
                print(counter)
            print(scores)
            print()
            
            y_pred = model.predict(x_test)
            bacc = balanced_accuracy_score(y_test, y_pred)
            
            scores['all'] = bacc
            
            # Save scores
            scores_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_scores.json')
            with open(scores_dir, 'w') as file:
                json.dump(scores, file, indent=4)
            
            # Save results
            results_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_results.pkl')
            joblib.dump(results, results_dir)    
            
            # Save model's parameters
            if model_name == 'nn2':
                params = {
                    'coefs_': model.coefs_,
                    'intercepts_': model.intercepts_
                }
            else: #if model_name == 'llr6':   
                params = {
                    'coef_': model.coef_,
                    'intercept_': model.intercept_
                }
            
            params_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_params.pkl')
            joblib.dump(params, params_dir)
            
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
            
            scaler_info_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_scaler_info.pkl')
            joblib.dump(scaler_info, scaler_info_dir)
