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

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch

from opacus import PrivacyEngine
from opacus.accountants.analysis import rdp as privacy_analysis

from utils import *


cwd = os.getcwd()

if torch.cuda.is_available():
    print('GPU is available')
    device = torch.device('cuda')
else:
    print('GPU is not available')
    device = torch.device('cpu')


def train_model(n_splits, n_repeats, sigma, all_data, featuresNA, phenoNA,
                datasets_ids):
    x = all_data[featuresNA].values
    y = all_data[phenoNA].values
    z = all_data['DatasetNum'].values
    y_z = np.array([f'{a}_{b}' for a, b in zip(y, z)])
    
    # Train
    models_dir = os.path.join(cwd, 'privacy', 'results', 'models', 'nn_dp')
    os.makedirs(models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        comb_dir = os.path.join(models_dir, '_'.join([str(c) for c in comb]))
        os.makedirs(comb_dir, exist_ok=True)
            
        model_type, param_dict = create_nn_model()

        # Define repeated k-fold cross-validation
        kf = RepeatedStratifiedKFold(n_splits=n_splits,
                                     n_repeats=n_repeats)

        # Store results
        for i, (train_idx, _) in enumerate(kf.split(x, y_z)):
            print(comb, i)
            
            params_dir = os.path.join(comb_dir, f'{sigma}_{i}_params.pt')
            epsilon_dir = os.path.join(comb_dir, f'{sigma}_{i}_epsilon.json')
            bal_accs_dir = os.path.join(comb_dir, f'{sigma}_{i}_bal_accs.json')
            auc_scores_dir = os.path.join(comb_dir, f'{sigma}_{i}_auc_scores.json')
            results_dir = os.path.join(comb_dir, f'{sigma}_{i}_results.pkl')
            
            if os.path.exists(params_dir):
                # Check if epsilon, bal_accs, auc_scores and results dirs exist
                if not os.path.exists(epsilon_dir):
                    raise ValueError(f'`epsilon` dir doesn\'t exist for '
                                     f'{comb, sigma, i}')
                    
                if not os.path.exists(bal_accs_dir):
                    raise ValueError(f'`bal_accs` dir doesn\'t exist for '
                                     f'{comb, sigma, i}')
                
                if not os.path.exists(auc_scores_dir):
                    raise ValueError(f'`auc_scores` dir doesn\'t exist for '
                                     f'{comb, sigma, i}')
                
                if not os.path.exists(results_dir):
                    raise ValueError(f'`results` dir doesn\'t exist for '
                                     f'{comb, sigma, i}')
                
                continue
            
            x_train = x[train_idx]
            y_train = y[train_idx]
            z_train = z[train_idx]
            
            train_idx_comb = np.isin(z_train, comb)
            x_train = x_train[train_idx_comb]
            y_train = y_train[train_idx_comb]
            
            xt_train = torch.from_numpy(x_train).float()
            yt_train = torch.from_numpy(y_train).float()
            
            dataset = TensorDataset(xt_train, yt_train)
            
            # Train the model
            model = model_type(input_dim=xt_train.shape[1],
                               hidden_sizes=param_dict['hidden_layer_sizes'])
            model.to(device)
            
            n_epochs = param_dict['max_iter'] // 2  # 50
            lr = param_dict['lr']
            weight_decay = param_dict['weight_decay']
            
            max_grad_norm = 1.0
            delta = 1e-4
            noise_multiplier = sigma  # [20.0, 5.0, 1.0, 0.0]
            
            batch_size = 32
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr,
                                   weight_decay=weight_decay)
            
            # Attach PrivacyEngine AFTER optimizer creation
            privacy_engine = PrivacyEngine()
            
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )

            for _ in range(n_epochs):
                for aux_x, aux_y in train_loader:
                    aux_x = aux_x.to(device)
                    aux_y = aux_y.to(device).float().view(-1, 1)

                    optimizer.zero_grad()
                    
                    scores = model(aux_x)
                    loss = criterion(scores, aux_y)
                    
                    loss.backward()
                    optimizer.step()
            
            # Save model's parameters
            model.cpu()
            torch.save([p.data for p in model.parameters()], params_dir)
            
            if sigma > 0:
                epsilon = privacy_engine.get_epsilon(delta)
            else:
                epsilon = float('inf')
            print('epsilon:', epsilon)
            
            with open(epsilon_dir, 'w') as file:
                json.dump({'epsilon': epsilon}, file, indent=4)

            # Evaluate final model by dataset
            model.eval()
            
            bal_accs = {}
            auc_scores = {}
            results = {}
            
            with torch.no_grad():
                for dat_id in datasets_ids:
                    idx = z == dat_id
                    x_aux = torch.from_numpy(x[idx]).float()
                    y_proba = model(x_aux).numpy()
                    # print(y[idx], y_proba)
                    
                    bacc, y_pred = balanced_accuracy(y[idx], y_proba)
                    bal_accs[dat_id] = bacc
                    
                    auc = roc_auc_score(y[idx], y_proba)
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
                y_proba = model(x_aux).numpy()
                bacc, y_pred = balanced_accuracy(y, y_proba)
                auc = roc_auc_score(y, y_proba)
                
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
              '\t--help, -h')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h')
        sys.exit(2)
    
    if len(args) == 3:
        n_splits = int(args[0])   # 5
        n_repeats = int(args[1])  # 20
        sigma = float(args[2])    # [20.0, 5.0, 1.0, 0.0]
    else:
        print('In "vanilla" mode the following arguments should be passed:\n'
              '\t1) <n_splits> => number of splits\n'
              '\t2) <n_repeats> => number of repeats of K-fold\n'
              '\t3) <sigma> => noise multiplier\n')
        sys.exit()
    
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
    
    all_data = load_data(cwd, featuresNA, phenoNA, datasets, datasets_ids)
    
    train_model(n_splits=n_splits,
                n_repeats=n_repeats,
                sigma=sigma,
                all_data=all_data,
                featuresNA=featuresNA,
                phenoNA=phenoNA,
                datasets_ids=datasets_ids)
