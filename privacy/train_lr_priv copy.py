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

import tensorkrowch as tk

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
    
    if len(args) == 3:
        n_bins = int(args[0])    # [2, 4, 6, 10]    -> 10
        l1 = float(args[1])      # [0.0, 0.5, 1.0]  -> 1.0
        C = float(args[2])       # [0.1, 1.0, 10.0] -> 0.1
    else:
        print('The following arguments should be passed:\n'
              '\t1) <l1> => l1 regularization weight\n'
              '\t2) <C> => inverse of total regularization weight\n')
    
    # We should use scaler_type = "standard"
    scaler_type = 'standard'
    model_type = 'vanilla' if options['vanilla'] else 'average'
    
    # Load data
    featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age',
                  'CancerType1', 'CancerType2', 'CancerType3', 'CancerType4',
                  'CancerType5', 'CancerType6', 'CancerType7', 'CancerType8',
                  'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12',
                  'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16']
    phenoNA = 'Response'
    
    # datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
    #             'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
    datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2',
                'Shim_NSCLC', 'Kato_panCancer']
    datasets_ids = list(range(1, len(datasets) + 1))
    
    # Load all data
    all_data = load_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    idx_datasets = np.isin(z, datasets_ids)
    x = x[idx_datasets]
    y = y[idx_datasets]
    z = z[idx_datasets]
    
    
    # Tensorization hyperparameters
    sketch_size    = 50
    phys_dim       = 2
    domain         = torch.linspace(0, 1, phys_dim) if scaler_type == 'minmax' else None
    bond_dim       = 2
    cum_percentage = 1 - 1e-2
    batch_size     = 1000
    device         = torch.device('cpu')
    verbose        = False
    
    def embedding(data):
        return tk.embeddings.poly(data, degree=phys_dim - 1).float()
    
    def tt_model(mps, data):
        y_proba = mps(embedding(data)).pow(2)
        y_proba = y_proba / y_proba.sum(dim=1, keepdim=True)
        return y_proba
    
    # Train
    aux_models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                  'tt', scaler_type, model_type)
    os.makedirs(aux_models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        aux_comb_dir = os.path.join(aux_models_dir,
                                    '_'.join([str(c) for c in comb]))
        os.makedirs(aux_comb_dir, exist_ok=True)

        # Store results
        for i in range(100):
            print(comb, n_bins, C, l1, i)
            bal_accs_dir = os.path.join(aux_comb_dir,
                                        f'{n_bins}_{C}_{l1}_{i}_bal_accs.json')
            auc_scores_dir = os.path.join(aux_comb_dir,
                                          f'{n_bins}_{C}_{l1}_{i}_auc_scores.json')
            results_dir = os.path.join(aux_comb_dir,
                                       f'{n_bins}_{C}_{l1}_{i}_results.pkl')
            
            # Load pre-trained LR
            aux_params_dir = os.path.join(aux_comb_dir, f'{n_bins}_{C}_{l1}_{i}_resc_cores.pt')
            
            cores = torch.load(aux_params_dir, weights_only=True)
            mps = tk.models.MPSLayer(tensors=cores)
            
            bal_accs = {}
            auc_scores = {}
            results = {}
            
            with torch.no_grad():
                for dat_id in datasets_ids:
                    idx = z == dat_id
                    x_aux = torch.from_numpy(x[idx]).float()
                    y_proba = tt_model(mps, x_aux).numpy()
                    
                    bacc, y_pred = balanced_accuracy(y[idx], y_proba[:, 1])
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
                y_proba = tt_model(mps, x_aux).numpy()
                bacc, y_pred = balanced_accuracy(y, y_proba[:, 1])
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
