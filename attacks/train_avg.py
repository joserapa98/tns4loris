#to run navigate to code then paste: python train_juliette.py 3 20 100 standard 1 0.1

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
from sklearn.metrics import balanced_accuracy_score

import pandas as pd
import numpy as np
import torch

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

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

cwd = os.getcwd()
#Specifcy input path here
input_dir=os.path.join(cwd, 'Tensor_Project/02.Input')
#Specify output directory
output_dir=os.path.join(cwd, 'Tensor_Results')


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
            'max_iter': 100, #Should be 200
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
            'max_iter': 100, # was 500 Should be 100?
            'l1_ratio': l1, #0.5, #1, #Should be 1 (passed in args)
            'class_weight': 'balanced',
            'C': C, #1, #0.1, #Should be 0.1 (passed in args)
        }
    
    return model_type, param_dict
    
    # model = model_type(**param_dict)
    # return model

def load_data(featuresNA, phenoNA, datasets, datasets_ids, scaler_type):
    data_file = os.path.join(input_dir, 'AllData.xlsx')

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
    
    if len(args) == 6:
        n_splits = int(args[0])   # 3
        n_repeats = int(args[1])  # 20
        n_models = int(args[2])   # 100
        scaler_type = args[3]
        l1 = float(args[4])      # [0., 0.5, 1.] -> 1.
        C = float(args[5])       # [0.1, 1, 10] -> 0.1
    else:
        print('The following arguments should be passed:\n'
              '\t1) <n_splits> => number of splits\n'
              '\t2) <n_repeats> => number of repeats of K-fold\n'
              '\t2) <n_models> => number of models\n'
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
    
    # NOTE: This split is to test all the averaged models on unseen data
    train_test_arrays = train_test_split(
        x, y, z, y_z,
        test_size=0.2,
        stratify=y_z,
        random_state=1
    )
    
    # x_train, x_test = train_test_arrays[:2]
    # y_train, y_test = train_test_arrays[2:4]
    # z_train, z_test = train_test_arrays[4:6]
    # y_z_train, y_z_test = train_test_arrays[6:8]

    x_train, x_test = x,x
    y_train, y_test = y,y
    z_train, z_test = z,z
    y_z_train, y_z_test = y_z,y_z
    
    # Train
    models_dir = os.path.join(output_dir, 'models', model_name, scaler_type)
    os.makedirs(models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    #
    for comb in all_combs:
        comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(comb_dir, exist_ok=True)
        
        # NOTE: comment if we want to restart the whole process or delete models
        # models_trained = len(os.listdir(comb_dir)) // 3  # params, results, scores
        models_trained = 0
        
        #This is where the loop begins (10000 iterations)
        # NOTE: For each i, we train "n_splits * n_repeats" LRs and then
        # average to return a single model
        for i in range(models_trained, n_models):
            all_params = []
            all_means = []
            all_scales = []
            
            model_type, param_dict = create_model(model_name, l1, C)

            # Define repeated k-fold cross-validation
            kf = RepeatedStratifiedKFold(n_splits=n_splits,
                                         n_repeats=n_repeats)

            # Store results
            for train_idx, _ in kf.split(x_train, y_z_train):
                
                x_train_aux = x_train[train_idx]
                y_train_aux = y_train[train_idx]
                z_train_aux = z_train[train_idx]
                
                train_idx_comb = np.isin(z_train_aux, comb)
                x_train_aux = x_train_aux[train_idx_comb]
                y_train_aux = y_train_aux[train_idx_comb]
                
                # Rescale
                scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
                x_train_aux = scaler.fit_transform(x_train_aux)
                
                #print(comb)
                counter = Counter(y_train_aux)
                #print(counter)
                
                # Train the model
                model = model_type(**param_dict)
                model.fit(x_train_aux, y_train_aux)
                
                # Save model's parameters
                if model_name == 'nn2':
                    coefs = [torch.from_numpy(c).flatten().to(device)
                             for c in model.coefs_]
                    coefs = torch.cat(coefs).to(device)
                    intercepts = [torch.from_numpy(c).flatten().to(device)
                                  for c in model.intercepts_]
                    intercepts = torch.cat(intercepts).to(device)
                else: #if model_name == 'llr6':
                    coefs = torch.from_numpy(model.coef_).flatten().to(device)
                    intercepts = torch.from_numpy(model.intercept_).flatten().to(device)
                
                all_params.append(torch.cat([coefs, intercepts]))
                
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
                
                all_means.append(torch.from_numpy(scaler_info['mean']))
                all_scales.append(torch.from_numpy(scaler_info['scale']))
            
        #unindent here?
        all_params = torch.stack(all_params, dim=0).to(device)
        all_means = torch.stack(all_means, dim=0).to(device)
        all_scales = torch.stack(all_scales, dim=0).to(device)
        
        # Re-scale and average models
        avg_params = all_params.mean(dim=0)
        avg_means = all_means.mean(dim=0)
        avg_scales = all_scales.mean(dim=0)
        
        # Split coefficients and intercept
        coefs = avg_params[:-1]
        intercept = avg_params[-1]
    

        # Rescale coefs
        new_coefs = coefs / avg_scales  # element-wise division

        # Adjust intercept
        intercept_shift = torch.sum(coefs * avg_means / avg_scales).to(device)
        new_intercept = intercept - intercept_shift

        # Concatenate back
        new_coefs_tensor = torch.cat([new_coefs,
                                        new_intercept.unsqueeze(0)]).to(device)
        avg_params = new_coefs_tensor
        
        #move to cpu before dumping
        params_cpu = move_to_cpu(avg_params)

        # Save model's parameters
        params_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_params.pkl')
        joblib.dump(params_cpu, params_dir)

        # Evaluate final model by dataset
        if model_name == 'llr6':
            model.coef_ = avg_params[:-1].cpu().numpy()
            model.intercept_ = avg_params[-1].cpu().numpy()
        else:
            raise ValueError(
                'model_name should be "llr6"')
        
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
            
            #print(dat_id)
            counter = Counter(y_test[idx])
            #print(counter)
        #print(scores)
        #print()
    
        y_pred = model.predict(x_test)
        bacc = balanced_accuracy_score(y_test, y_pred)
        
        scores['all'] = bacc
        

        scores_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_scores.json')
        with open(scores_dir, 'w') as file:
            json.dump(scores, file, indent=4)
        
        #move to cpu before dumping
        results_cpu = move_to_cpu(results)

        # Save model's parameters
        results_dir = os.path.join(comb_dir, f'{C}_{l1}_{i}_results.pkl')
        joblib.dump(results_cpu, results_dir)
