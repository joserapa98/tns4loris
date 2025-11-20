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

import shutil
    


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
        
    scaler_type = 'standard'
    model_type = 'vanilla' if options['vanilla'] else 'average'
    
    datasets_ids = list(range(1, 7))
    
    
    aux_models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                  'tt', scaler_type, model_type)
    os.makedirs(aux_models_dir, exist_ok=True)
    
    new_models_dir = os.path.join(cwd, 'privacy', 'results', 'models',
                                  'new_tt', scaler_type, model_type)
    os.makedirs(new_models_dir, exist_ok=True)
    
    all_combs = all_combinations(datasets_ids)
    
    for comb in all_combs:
        print(comb)
        aux_comb_dir = os.path.join(aux_models_dir,
                                    '_'.join([str(c) for c in comb]))
        new_comb_dir = os.path.join(new_models_dir,
                                     '_'.join([str(c) for c in comb]))
        
        shutil.copytree(aux_comb_dir, new_comb_dir)
        