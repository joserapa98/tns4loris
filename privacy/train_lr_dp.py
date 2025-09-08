# Run script from parent folder "tns4loris"

import sys
import os
import getopt
import joblib
import json

from collections import Counter

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import numpy as np
import torch

from utils import *


cwd = os.getcwd()

if torch.cuda.is_available():
    print('GPU is available')
    device = torch.device('cuda')
else:
    print('GPU is not available')
    device = torch.device('cpu')


def train_vanilla(n_splits, n_repeats, scaler_type, epsilon,
                  all_data, featuresNA, phenoNA, datasets_ids):
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    # Train
    models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                              'lr_dp', scaler_type, 'vanilla')
    os.makedirs(models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(comb_dir, exist_ok=True)
            
        model_type, param_dict = create_lr_dp_model(epsilon)

        # Define repeated k-fold cross-validation
        kf = RepeatedStratifiedKFold(n_splits=n_splits,
                                     n_repeats=n_repeats)

        # Store results
        for i, (train_idx, _) in enumerate(kf.split(x, y_z)):
            print(comb, epsilon, i)
            
            params_dir = os.path.join(comb_dir, f'{epsilon}_{i}_params.pkl')
            resc_params_dir = os.path.join(comb_dir, f'{epsilon}_{i}_resc_params.pkl')
            bal_accs_dir = os.path.join(comb_dir, f'{epsilon}_{i}_bal_accs.json')
            auc_scores_dir = os.path.join(comb_dir, f'{epsilon}_{i}_auc_scores.json')
            results_dir = os.path.join(comb_dir, f'{epsilon}_{i}_results.pkl')
            
            if os.path.exists(params_dir):
                # Check if resc_params, bal_accs, auc_scores and results dirs exist
                if not os.path.exists(resc_params_dir):
                    raise ValueError(f'`resc_params` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                if not os.path.exists(bal_accs_dir):
                    raise ValueError(f'`bal_accs` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                if not os.path.exists(auc_scores_dir):
                    raise ValueError(f'`auc_scores` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                if not os.path.exists(results_dir):
                    raise ValueError(f'`results` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                continue
            
            x_train = x[train_idx]
            y_train = y[train_idx]
            z_train = z[train_idx]
            
            train_idx_comb = np.isin(z_train, comb)
            x_train = x_train[train_idx_comb]
            y_train = y_train[train_idx_comb]
            
            # Scale data
            scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
            x_train = scaler.fit_transform(x_train)
            
            # Train the model
            model = model_type(**param_dict)
            model.fit(x_train, y_train)
            
            # Save model's parameters
            coefs = torch.from_numpy(model.coef_).flatten()
            intercepts = torch.from_numpy(model.intercept_).flatten()
            
            params = torch.cat([coefs, intercepts])
            joblib.dump(params, params_dir)
            
            # Weight and range to rescale parameters
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
            resc_params = new_coefs_tensor
            
            # Save rescaled parameters
            joblib.dump(resc_params, resc_params_dir)

            # Evaluate final model by dataset
            model.coef_ = resc_params[:-1].numpy()
            model.intercept_ = resc_params[-1].numpy()
            
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
            
            # Save accuracies
            with open(bal_accs_dir, 'w') as file:
                json.dump(bal_accs, file, indent=4)
            
            # Save scores
            with open(auc_scores_dir, 'w') as file:
                json.dump(auc_scores, file, indent=4)
            
            # Save results
            joblib.dump(results, results_dir)



def train_average(n_splits, n_repeats, n_models, scaler_type, epsilon,
                  all_data, featuresNA, phenoNA, datasets_ids):
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    # Train
    models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                              'lr_dp', scaler_type, 'average')
    os.makedirs(models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(comb_dir, exist_ok=True)
        
        # This is where the loop begins (10000 iterations)
        # For each i, we train "n_splits * n_repeats" LRs and then
        # average to return a single model
        for i in range(n_models):
            print(comb, epsilon, i)
            
            params_dir = os.path.join(comb_dir, f'{epsilon}_{i}_params.pkl')
            resc_params_dir = os.path.join(comb_dir, f'{epsilon}_{i}_resc_params.pkl')
            bal_accs_dir = os.path.join(comb_dir, f'{epsilon}_{i}_bal_accs.json')
            auc_scores_dir = os.path.join(comb_dir, f'{epsilon}_{i}_auc_scores.json')
            results_dir = os.path.join(comb_dir, f'{epsilon}_{i}_results.pkl')
            
            if os.path.exists(params_dir):
                # Check if resc_params, bal_accs, auc_scores and results dirs exist
                if not os.path.exists(resc_params_dir):
                    raise ValueError(f'`resc_params` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                if not os.path.exists(bal_accs_dir):
                    raise ValueError(f'`bal_accs` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                if not os.path.exists(auc_scores_dir):
                    raise ValueError(f'`auc_scores` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                if not os.path.exists(results_dir):
                    raise ValueError(f'`results` dir doesn\'t exist for '
                                     f'{comb, epsilon, i}')
                
                continue
            
            all_params = []
            all_means = []
            all_scales = []
            
            model_type, param_dict = create_lr_dp_model(epsilon)

            # Define repeated k-fold cross-validation
            kf = RepeatedStratifiedKFold(n_splits=n_splits,
                                         n_repeats=n_repeats)

            # Store results
            for train_idx, _ in kf.split(x, y_z):
                
                x_train = x[train_idx]
                y_train = y[train_idx]
                z_train = z[train_idx]
                
                train_idx_comb = np.isin(z_train, comb)
                x_train = x_train[train_idx_comb]
                y_train = y_train[train_idx_comb]
                
                # Scale data
                scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
                x_train = scaler.fit_transform(x_train)
                
                # Train the model
                model = model_type(**param_dict)
                model.fit(x_train, y_train)
                
                # Save model's parameters
                coefs = torch.from_numpy(model.coef_).flatten()
                intercepts = torch.from_numpy(model.intercept_).flatten()
                
                all_params.append(torch.cat([coefs, intercepts]))
                
                # Weight and range to rescale params
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
                
                all_means.append(torch.from_numpy(scaler_info['mean']))
                all_scales.append(torch.from_numpy(scaler_info['scale']))
            
            all_params = torch.stack(all_params, dim=0)
            all_means = torch.stack(all_means, dim=0)
            all_scales = torch.stack(all_scales, dim=0)
            
            # Average models
            avg_params = all_params.mean(dim=0)
            avg_means = all_means.mean(dim=0)
            avg_scales = all_scales.mean(dim=0)
            
            # Save parameters
            joblib.dump(avg_params, params_dir)
            
            # Split coefficients and intercept
            coefs = avg_params[:-1]
            intercept = avg_params[-1]

            # Rescale coefs
            new_coefs = coefs / avg_scales  # element-wise division

            # Adjust intercept
            intercept_shift = torch.sum(coefs * avg_means / avg_scales)
            new_intercept = intercept - intercept_shift

            # Concatenate back
            new_coefs_tensor = torch.cat([new_coefs,
                                          new_intercept.unsqueeze(0)])
            resc_avg_params = new_coefs_tensor
            
            # Save rescaled parameters
            joblib.dump(resc_avg_params, resc_params_dir)

            # Evaluate final model by dataset
            model.coef_ = resc_avg_params[:-1].numpy()
            model.intercept_ = resc_avg_params[-1].numpy()
            
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
            
            # Save accuracies
            with open(bal_accs_dir, 'w') as file:
                json.dump(bal_accs, file, indent=4)
            
            # Save scores
            with open(auc_scores_dir, 'w') as file:
                json.dump(auc_scores, file, indent=4)
            
            # Save results
            joblib.dump(results, results_dir)


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
    
    # VANILLA
    if options['vanilla']:
        if len(args) == 4:
            n_splits = int(args[0])   # 5
            n_repeats = int(args[1])  # 20
            scaler_type = args[2]     # standard / minmax
            epsilon = float(args[3])  # [0.01, 0.1, 1.0, 10.0, 100.0, inf]
        else:
            print('In "vanilla" mode the following arguments should be passed:\n'
                  '\t1) <n_splits> => number of splits\n'
                  '\t2) <n_repeats> => number of repeats of K-fold\n'
                  '\t3) <scaler_type> => standard / minmax scaler\n'
                  '\t4) <epsilon> => epsilon parameter of DP\n')
            sys.exit()
    
    # AVERAGE
    if options['average']:
        if len(args) == 5:
            n_splits = int(args[0])   # 3
            n_repeats = int(args[1])  # 20
            n_models = int(args[2])   # 100
            scaler_type = args[3]     # standard / minmax
            epsilon = float(args[4])  # [0.01, 0.1, 1.0, 10.0, 100.0]
        else:
            print('In "average" mode the following arguments should be passed:\n'
                  '\t1) <n_splits> => number of splits\n'
                  '\t2) <n_repeats> => number of repeats of K-fold\n'
                  '\t3) <n_models> => number of models\n'
                  '\t4) <scaler_type> => standard / minmax scaler\n'
                  '\t5) <epsilon> => epsilon parameter of DP\n')
            sys.exit()
    
    # We should use scaler_type = "standard"
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
    
    all_data = load_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    # VANILLA
    if options['vanilla']:
        train_vanilla(n_splits=n_splits,
                      n_repeats=n_repeats,
                      scaler_type=scaler_type,
                      epsilon=epsilon,
                      all_data=all_data,
                      featuresNA=featuresNA,
                      phenoNA=phenoNA,
                      datasets_ids=datasets_ids)
    # AVERAGE
    else:
        train_average(n_splits=n_splits,
                      n_repeats=n_repeats,
                      n_models=n_models,
                      scaler_type=scaler_type,
                      epsilon=epsilon,
                      all_data=all_data,
                      featuresNA=featuresNA,
                      phenoNA=phenoNA,
                      datasets_ids=datasets_ids)
