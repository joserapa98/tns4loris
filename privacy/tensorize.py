# Run script from parent folder "tns4loris"

import sys
import os
import getopt
import joblib
import json

from collections import Counter

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import numpy as np
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss
from tensorkrowch.utils import random_unitary

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
    
    # Load sketch data
    sketch_data = load_sketch_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    # We tensorize using data from all datasets, sampling uniformly from a
    # balanced set that contains the same number of samples for each cancer type,
    # with both Response 0 and 1.
    x_sketch = sketch_data[featuresNA].values
    y_sketch = sketch_data[phenoNA].values
    z_sketch = sketch_data['DatasetNum'].values
    
    xt_sketch = torch.from_numpy(x_sketch).float()
    yt_sketch = torch.from_numpy(y_sketch)
    zt_sketch = torch.from_numpy(z_sketch)
    
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
    
    # Initialize the model
    model_class, param_dict = create_lr_model(1.0, 0.1)
    model = model_class(**param_dict)
    model.classes_ = np.array([0, 1])
    
    def fn_model(data, n_bins=6):
        # Get probabilities (numpy)
        y_proba = model.predict_proba(data)
        y_proba = torch.from_numpy(y_proba).float()

        # Bin width
        step = 1.0 / n_bins

        # Compute bin index
        bin_idx = torch.floor(y_proba / step)  # 0 ... n_bins-1

        # Lower & upper edges of the bin
        bin_low  = bin_idx * step
        bin_high = bin_low + step

        # Apply your rule:
        #   if prob <= 0.5 → snap to lower edge
        #   if prob > 0.5  → snap to upper edge
        result = torch.where(y_proba <= 0.5, bin_low, bin_high)

        # Clip to [0,1]
        result = torch.clamp(result, 0.0, 1.0)

        return result.sqrt()
    
    # Tensorize
    models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                              'lr', scaler_type, model_type)
    os.makedirs(models_dir, exist_ok=True)
    
    tt_models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                 'tt', scaler_type, model_type)
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
            weights = params[:-1].unsqueeze(0).numpy()  # shape (1, n_features)
            intercept = params[-1:].numpy()             # shape (1,)

            # Manually set parameters
            model.coef_ = weights
            model.intercept_ = intercept
            
            # Tensorization
            rand_ids = torch.randperm(xt_sketch.size(0))[:sketch_size]
            cores = tt_rss(function=fn_model,
                           embedding=embedding,
                           sketch_samples=xt_sketch[rand_ids],
                           labels=yt_sketch[rand_ids],
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
                    y_proba = tt_model(embedding(x_aux)).pow(2)
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
