# Run script from parent folder "tns4loris"

import sys
import os
import getopt
import joblib
import json

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold

import torch
import tensorkrowch as tk
from tensorkrowch.utils import random_unitary


cwd = os.getcwd()

if torch.cuda.is_available():
    print('GPU is available')
    device = torch.device('cuda')
else:
    print('GPU is not available')
    device = torch.device('cpu')


def create_datasets_og(model_name, scaler_type, model_type):
    datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
                'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
    datasets_ids = list(range(1, len(datasets) + 1))
    
    # Create whole models dataset
    all_labels = []
    all_params = []
    all_bal_accs = []
    all_auc_scores = []

    datasets_dir = os.path.join(cwd, 'privacy', 'datasets',
                                model_name, scaler_type, model_type)
    os.makedirs(datasets_dir, exist_ok=True)
    
    data_dir = os.path.join(datasets_dir, 'params_multilabel.pt')
    if not os.path.exists(data_dir):
        models_dir = os.path.join(cwd, 'privacy', 'models',
                                  model_name, scaler_type, model_type)
        
        for comb in os.listdir(models_dir):
            print(comb)
            aux_labels = [int(n) - 1 for n in comb.split('_')]
            one_hot_label = torch.zeros(len(datasets))
            one_hot_label[aux_labels] = 1
            
            comb_dir = os.path.join(models_dir, comb)
            for C in [0.1, 1.0, 10.0]:
                for l1 in [0.0, 0.5, 1.0]:
                    for i in range(100):
                        params_file = f'{C}_{l1}_{i}_params.pkl'
                        bal_accs_file = f'{C}_{l1}_{i}_bal_accs.json'
                        auc_scores_file = f'{C}_{l1}_{i}_auc_scores.json'
                        
                        # params
                        params_dir = os.path.join(comb_dir, params_file)
                        if not os.path.exists(params_dir):
                                continue
                        
                        params = joblib.load(params_dir)
                            
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

        all_labels = torch.stack(all_labels, dim=0).int()
        all_params = torch.stack(all_params, dim=0)
        all_bal_accs = torch.stack(all_bal_accs, dim=0)
        all_auc_scores = torch.stack(all_auc_scores, dim=0)

        torch.save((all_labels, all_params, all_bal_accs, all_auc_scores),
                   data_dir)
    
    else:
        all_labels, all_params, all_bal_accs, all_auc_scores = \
            torch.load(data_dir, weights_only=True)
    
    return all_labels, all_params, all_bal_accs, all_auc_scores


def create_datasets_tt(model_name, scaler_type, model_type):
    datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
                'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
    datasets_ids = list(range(1, len(datasets) + 1))
    
    bond_dim = 2
    
    # Create whole models dataset
    all_labels = []
    all_cores = []
    all_bal_accs = []
    all_auc_scores = []

    datasets_dir = os.path.join(cwd, 'privacy', 'tt_datasets',
                                model_name, scaler_type, model_type)
    os.makedirs(datasets_dir, exist_ok=True)
    
    data_dir = os.path.join(datasets_dir, 'params_multilabel.pt')
    if not os.path.exists(data_dir):
        models_dir = os.path.join(cwd, 'privacy', 'tt_models',
                                  model_name, scaler_type, model_type)
        
        for comb in os.listdir(models_dir):
            print(comb)
            aux_labels = [int(n) - 1 for n in comb.split('_')]
            one_hot_label = torch.zeros(len(datasets))
            one_hot_label[aux_labels] = 1
            
            comb_dir = os.path.join(models_dir, comb)
            for C in [0.1, 1.0, 10.0]:
                for l1 in [0.0, 0.5, 1.0]:
                    for i in range(100):
                        cores_file = f'{C}_{l1}_{i}_cores.pkl'
                        bal_accs_file = f'{C}_{l1}_{i}_bal_accs.json'
                        auc_scores_file = f'{C}_{l1}_{i}_auc_scores.json'
                        
                        # cores
                        cores_dir = os.path.join(comb_dir, cores_file)
                        if not os.path.exists(cores_dir):
                            continue
                        
                        cores = torch.load(cores_dir, weights_only=True)
                        
                        
                        # # Make all cores equal size
                        # tt_model = tk.models.MPSLayer(tensors=cores)
                        # for j in range(len(tt_model.mats_env) - 1):
                        #     tt_model.mats_env[j]['right'].change_size(size=bond_dim)
                        
                        # # Randomize gauge
                        # for j, node in enumerate(tt_model.mats_env):
                        #     right_size = node.size('right')
                        #     # U = random_unitary(right_size)
                        #     U = torch.randn((right_size, right_size))
                        #     if j < (len(tt_model.mats_env) - 1):
                        #         node.tensor = torch.einsum('lir,rk->lik',
                        #                                    node.tensor, U)
                        #     if j > 0:
                        #         node.tensor = torch.einsum('kl,lir->kir',
                        #                                    prev_U, node.tensor)
                        #     # prev_U = U.clone().H
                        #     prev_U = torch.linalg.inv(U)
                        #     # print(U @ prev_U)
                        
                        # cores = tt_model.tensors
                        
                        
                        cores = [c.flatten() for c in cores]
                        cores = torch.cat(cores, dim=0)
                            
                        all_labels.append(one_hot_label.clone())
                        all_cores.append(cores.clone())
                        
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

        all_labels = torch.stack(all_labels, dim=0).int()
        all_cores = torch.stack(all_cores, dim=0).detach()
        all_bal_accs = torch.stack(all_bal_accs, dim=0)
        all_auc_scores = torch.stack(all_auc_scores, dim=0)
        
        # print(all_labels, all_cores, all_bal_accs, all_auc_scores)

        torch.save((all_labels, all_cores, all_bal_accs, all_auc_scores),
                   data_dir)
    
    else:
        all_labels, all_cores, all_bal_accs, all_auc_scores = \
            torch.load(data_dir, weights_only=True)
    
    return all_labels, all_cores, all_bal_accs, all_auc_scores


def create_datasets_dp(model_name, scaler_type, model_type):
    datasets = ['Chowell_train', 'Chowell_test', 'MSK1', 'MSK2', 'Shim_NSCLC',
                'Kato_panCancer', 'Vanguri_NSCLC', 'Ravi_NSCLC', 'Pradat_panCancer']
    datasets_ids = list(range(1, len(datasets) + 1))
    
    # Create whole models dataset
    all_labels = []
    all_params = []
    all_bal_accs = []
    all_auc_scores = []

    datasets_dir = os.path.join(cwd, 'privacy', 'dp_datasets',
                                model_name, scaler_type, model_type)
    os.makedirs(datasets_dir, exist_ok=True)
    
    data_dir = os.path.join(datasets_dir, 'params_multilabel.pt')
    if not os.path.exists(data_dir):
        models_dir = os.path.join(cwd, 'privacy', 'dp_models',
                                  model_name, scaler_type, model_type)
        
        for comb in os.listdir(models_dir):
            print(comb)
            aux_labels = [int(n) - 1 for n in comb.split('_')]
            one_hot_label = torch.zeros(len(datasets))
            one_hot_label[aux_labels] = 1
            
            comb_dir = os.path.join(models_dir, comb)
            for epsilon in [0.001, 0.1, 1.0, 10.0, 100.0]:
                for i in range(100):
                    params_file = f'{epsilon}_{i}_params.pkl'
                    bal_accs_file = f'{epsilon}_{i}_bal_accs.json'
                    auc_scores_file = f'{epsilon}_{i}_auc_scores.json'
                    
                    # params
                    params_dir = os.path.join(comb_dir, params_file)
                    if not os.path.exists(params_dir):
                            continue
                    
                    params = joblib.load(params_dir)
                        
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

        all_labels = torch.stack(all_labels, dim=0).int()
        all_params = torch.stack(all_params, dim=0)
        all_bal_accs = torch.stack(all_bal_accs, dim=0)
        all_auc_scores = torch.stack(all_auc_scores, dim=0)

        torch.save((all_labels, all_params, all_bal_accs, all_auc_scores),
                   data_dir)
    
    else:
        all_labels, all_params, all_bal_accs, all_auc_scores = \
            torch.load(data_dir, weights_only=True)
    
    return all_labels, all_params, all_bal_accs, all_auc_scores


def bb_attack(og_model, all_scores, all_labels, model_name, scaler_type,
              model_type, attack_name):
    X, y = all_scores, all_labels
    
    aux_dir = f'attack{og_model}_models'
    attack_model_dir = os.path.join(cwd, 'privacy', aux_dir,
                                    model_name, scaler_type, model_type)
    os.makedirs(attack_model_dir, exist_ok=True)

    # Train/Test Split (held-out test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Save held-out test set (optional)
    joblib.dump((X_test, y_test),
                os.path.join(attack_model_dir,
                             f'heldout_test_set_bb_{attack_name}.pkl'))
    
    # Define and train the MLP model
    # mlp_bb = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
    #                        activation='relu',
    #                        solver='adam',
    #                        max_iter=1000)
    mlp_bb = MLPClassifier(hidden_layer_sizes=(128, 32),
                           activation='relu',
                           solver='adam',
                           max_iter=100)
    mlp_bb.fit(X_train, y_train)

    # Save model
    joblib.dump(mlp_bb,
                os.path.join(attack_model_dir,
                             f'mlp_attacker_multilabel_bb_{attack_name}.pkl'))


def wb_attack(og_model, all_params, all_labels, model_name, scaler_type,
              model_type):
    X, y = all_params, all_labels
    
    aux_dir = f'attack{og_model}_models'
    attack_model_dir = os.path.join(cwd, 'privacy', aux_dir,
                                    model_name, scaler_type, model_type)
    os.makedirs(attack_model_dir, exist_ok=True)

    # First: Train/Test Split (held-out test set)
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Save held-out test set (optional)
    joblib.dump((X_test, y_test),
                os.path.join(attack_model_dir, 'heldout_test_set_wb.pkl'))

    # K-Fold CV on training set
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, _) in enumerate(kfold.split(X_train_all,
                                                        y_train_all)):
        print(f'Training fold {fold+1}/{n_splits}...')

        X_train = X_train_all[train_index]
        y_train = y_train_all[train_index]

        # Define and train the model
        # mlp_wb = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
        #                        activation='relu',
        #                        solver='adam',
        #                        max_iter=500)
        mlp_wb = MLPClassifier(hidden_layer_sizes=(128, 32),
                               activation='relu',
                               solver='adam',
                               max_iter=100)
        mlp_wb.fit(X_train, y_train)

        # Save model for this fold
        joblib.dump(mlp_wb,
                    os.path.join(attack_model_dir,
                                 f'mlp_attacker_multilabel_wb_fold_{fold+1}.pkl'))


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--og_model\n'
              '\t--tt_model\n'
              '\t--dp_model\n'
              '\t--vanilla\n'
              '\t--average\n'
              '\t--bb\n'
              '\t--wb')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h',
                                   ['help',
                                    'og_model', 'tt_model', 'dp_model',
                                    'vanilla', 'average',
                                    'bb', 'wb'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--og_model\n'
              '\t--tt_model\n'
              '\t--dp_model\n'
              '\t--vanilla\n'
              '\t--average\n'
              '\t--bb\n'
              '\t--wb')
        sys.exit(2)
    
    # Save selected options
    options = {'og_model': False,
               'tt_model': False,
               'dp_model': False,
               'vanilla': False,
               'average': False,
               'bb': False,
               'wb': False}
    
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--help, -h\n'
                  '\t--og_model\n'
                  '\t--tt_model\n'
                  '\t--dp_model\n'
                  '\t--vanilla\n'
                  '\t--average\n'
                  '\t--bb\n'
                  '\t--wb')
            sys.exit()
        elif opt == '--og_model':
            options['og_model'] = True
        elif opt == '--tt_model':
            options['tt_model'] = True
        elif opt == '--dp_model':
            options['dp_model'] = True
        elif opt == '--vanilla':
            options['vanilla'] = True
        elif opt == '--average':
            options['average'] = True
        elif opt == '--bb':
            options['bb'] = True
        elif opt == '--wb':
            options['wb'] = True
    
    # Check if selected options are compatible
    if options['og_model'] and options['tt_model'] and options['dp_model']:
        print('Options "og_model", and "tt_model" and "dp_model" are incompatible')
        sys.exit()
    elif not (options['og_model'] or options['tt_model'] or options['dp_model']):
        print('One of the options "og_model", "tt_model" and "dp_model" should be chosen')
        sys.exit()
    
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
    
    if options['og_model']:
        og_model = ''
    elif options['tt_model']:
        og_model = '_tt'
    elif options['dp_model']:
        og_model = '_dp'
    
    model_name = 'llr6' # 'llr6' or 'nn2'
    scaler_type = 'standard'
    model_type = 'vanilla' if options['vanilla'] else 'average'
    attack_type = 'bb' if options['bb'] else 'wb'
    
    print('\n* Creating datasets...')
    if options['og_model']:
        all_labels, all_params, all_bal_accs, all_auc_scores = \
            create_datasets_og(model_name=model_name,
                               scaler_type=scaler_type,
                               model_type=model_type)
    elif options['tt_model']:
        all_labels, all_params, all_bal_accs, all_auc_scores = \
            create_datasets_tt(model_name=model_name,
                               scaler_type=scaler_type,
                               model_type=model_type)
    elif options['dp_model']:
        all_labels, all_params, all_bal_accs, all_auc_scores = \
            create_datasets_dp(model_name=model_name,
                               scaler_type=scaler_type,
                               model_type=model_type)
    
    # Remove models with low accuracies
    lower_bound = 0.64

    # idx = (all_bal_accs >= lower_bound).all(dim=1)
    idx = (all_bal_accs[:, -1] >= lower_bound)

    all_labels = all_labels[idx]
    all_params = all_params[idx]
    all_bal_accs = all_bal_accs[idx]
    all_auc_scores = all_auc_scores[idx]
    
    print('\n* Performing attacks...')
    if attack_type == 'bb':
        bb_attack(og_model=og_model,
                  all_scores=all_bal_accs,
                  all_labels=all_labels,
                  model_name=model_name,
                  scaler_type=scaler_type,
                  model_type=model_type,
                  attack_name='weak')
        bb_attack(og_model=og_model,
                  all_scores=all_auc_scores,
                  all_labels=all_labels,
                  model_name=model_name,
                  scaler_type=scaler_type,
                  model_type=model_type,
                  attack_name='strong')
    else:
        wb_attack(og_model=og_model,
                  all_params=all_params,
                  all_labels=all_labels,
                  model_name=model_name,
                  scaler_type=scaler_type,
                  model_type=model_type)
