# Guide to reproduce experiments:


1. Train vanilla/average LR models:
    ```
    python privacy/train_lr.py --vanilla <n_splits> <n_repeats> <scaler_type> <l1> <C>
    ```
    - ``n_splits = 5``
    - ``n_repeats = 20``
    - ``scaler_type = "standard"``
    - ``l1 = 0.0, 0.5, 1.0``
    - ``C = 0.1, 1.0, 10.0``

    ```
    python privacy/train_lr.py --average <n_splits> <n_repeats> <n_models> <scaler_type> <l1> <C>
    ```
    - ``n_splits = 3``
    - ``n_repeats = 20``
    - ``n_models = 100``
    - ``scaler_type = "standard"``
    - ``l1 = 0.0, 0.5, 1.0``
    - ``C = 0.1, 1.0, 10.0``


2. Train vanilla LR models with DP:
    ```
    python privacy/train_lr_dp.py --vanilla <n_splits> <n_repeats> <scaler_type> <epsilon>
    ```
    - ``n_splits = 5``
    - ``n_repeats = 20``
    - ``scaler_type = "standard"``
    - ``epsilon = 0.1, 1.0, 10.0, 100.0, inf``


3. Tensorize vanilla LR models:
    ```
    python privacy/tensorize_lr.py --vanilla <n_bins> <l1> <C>
    ```
    - ``n_bins = 2, 6, 10``
    - ``l1 = 0.0, 0.5, 1.0``
    - ``C = 0.1, 1.0, 10.0``


4. Train NN models:
    ```
    python privacy/train_nn.py <n_splits> <n_repeats>
    ```
    - ``n_splits = 5``
    - ``n_repeats = 20``


5. Train NN models with DP:
    ```
    python privacy/train_nn_dp.py <n_splits> <n_repeats> <sigma>
    ```
    - ``n_splits = 5``
    - ``n_repeats = 20``
    - ``sigma = 20.0, 5.0, 1.0, 0.0``


6. Tensorize NN models:
    ```
    python privacy/tensorize_nn.py <n_bins>
    ```
    - ``n_bins = 2, 6, 10``


7. Train attacker models:
    ```
    python privacy/attacks.py --<model_type> --<attack_type> <model_name>
    ```

    - ``model_type = "vanilla", "average"``
    - ``attack_type = "bb", "wb"``
    - ``model_name = "lr", "lr_priv", "lr_dp", "tt"``
