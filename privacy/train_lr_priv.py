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
    
    # We should use scaler_type = "standard"
    scaler_type = 'standard'
    model_type = 'vanilla' if options['vanilla'] else 'average'
    n_splits = 5
    n_repeats = 20
    
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
    
    # Load all data
    all_data = load_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    # Train
    aux_models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                  'lr', scaler_type, model_type)
    os.makedirs(aux_models_dir, exist_ok=True)
    
    priv_models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                   'lr_priv', scaler_type, model_type)
    os.makedirs(priv_models_dir, exist_ok=True)
    
    # Initialize the aux model
    aux_model_class, aux_param_dict = create_lr_model(1.0, 0.1)
    aux_model = aux_model_class(**aux_param_dict)
    aux_model.classes_ = np.array([0, 1])
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        aux_comb_dir = os.path.join(aux_models_dir,
                                    '_'.join([str(c) for c in comb]))
        os.makedirs(aux_comb_dir, exist_ok=True)
        
        priv_comb_dir = os.path.join(priv_models_dir,
                                     '_'.join([str(c) for c in comb]))
        os.makedirs(priv_comb_dir, exist_ok=True)
            
        model_class, param_dict = create_lr_model(l1, C)

        # Define repeated k-fold cross-validation
        kf = RepeatedStratifiedKFold(n_splits=n_splits,
                                     n_repeats=n_repeats)

        # Store results
        for i, (train_idx, _) in enumerate(kf.split(x, y_z)):
            print(comb, C, l1, i)
            
            priv_params_dir = os.path.join(priv_comb_dir,
                                           f'{C}_{l1}_{i}_params.pkl')
            resc_priv_params_dir = os.path.join(priv_comb_dir,
                                                f'{C}_{l1}_{i}_resc_params.pkl')
            bal_accs_dir = os.path.join(priv_comb_dir,
                                        f'{C}_{l1}_{i}_bal_accs.json')
            auc_scores_dir = os.path.join(priv_comb_dir,
                                          f'{C}_{l1}_{i}_auc_scores.json')
            results_dir = os.path.join(priv_comb_dir,
                                       f'{C}_{l1}_{i}_results.pkl')
            
            if os.path.exists(priv_params_dir):
                # Check if resc_params, bal_accs, auc_scores and results dirs exist
                if not os.path.exists(resc_priv_params_dir):
                    raise ValueError(f'`resc_params` dir doesn\'t exist for '
                                     f'{comb, C, l1, i}')
                
                if not os.path.exists(bal_accs_dir):
                    raise ValueError(f'`bal_accs` dir doesn\'t exist for '
                                     f'{comb, C, l1, i}')
                
                if not os.path.exists(auc_scores_dir):
                    raise ValueError(f'`auc_scores` dir doesn\'t exist for '
                                     f'{comb, C, l1, i}')
                
                if not os.path.exists(results_dir):
                    raise ValueError(f'`results` dir doesn\'t exist for '
                                     f'{comb, C, l1, i}')
                
                continue
            
            x_train = x[train_idx]
            z_train = z[train_idx]
            
            train_idx_comb = np.isin(z_train, comb)
            x_train = x_train[train_idx_comb]
            
            # Scale data
            scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
            x_train = scaler.fit_transform(x_train)
            
            # Load pre-trained LR
            aux_params_dir = os.path.join(aux_comb_dir, f'{C}_{l1}_{i}_params.pkl')
            aux_params = joblib.load(aux_params_dir)
            
            # Split into coefs and intercept
            aux_coefs = aux_params[:-1].unsqueeze(0).numpy()  # shape (1, n_features)
            aux_intercept = aux_params[-1:].numpy()           # shape (1,)

            # Manually set parameters of aux model
            aux_model.coef_ = aux_coefs
            aux_model.intercept_ = aux_intercept
            
            # Evaluate aux model to obtain labels
            y_train = aux_model.predict(x_train)
            
            # Solve case when y_train only has one label
            if len(np.unique(y_train)) == 1:
                y_train = y[train_idx][train_idx_comb]
            
            # Train the model
            model = model_class(**param_dict)
            model.fit(x_train, y_train)
            
            # Save model's parameters
            coefs = torch.from_numpy(model.coef_).flatten()
            intercepts = torch.from_numpy(model.intercept_).flatten()
            
            priv_params = torch.cat([coefs, intercepts])
            joblib.dump(priv_params, priv_params_dir)
            
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
            coefs = priv_params[:-1]
            intercept = priv_params[-1]
            
            # Rescale coefs
            new_coefs = coefs / scale  # element-wise division

            # Adjust intercept
            intercept_shift = torch.sum(coefs * mean / scale)
            new_intercept = intercept - intercept_shift

            # Concatenate back
            new_coefs_tensor = torch.cat([new_coefs, new_intercept.unsqueeze(0)])
            resc_priv_params = new_coefs_tensor
            
            # Save rescaled parameters
            joblib.dump(resc_priv_params, resc_priv_params_dir)

            # Evaluate final model by dataset
            model.coef_ = resc_priv_params[:-1].numpy()
            model.intercept_ = resc_priv_params[-1].numpy()
            
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
