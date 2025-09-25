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


2. Train vanilla/average private LR models:
    ```
    python privacy/train_lr_priv.py --<model_type> <l1> <C>
    ```
    - ``model_type = "vanilla", "average"``
    - ``l1 = 0.0, 0.5, 1.0``
    - ``C = 0.1, 1.0, 10.0``


3. Tensorize vanilla/average LR models:
    ```
    python privacy/tensorize.py --<model_type> <l1> <C>
    ```
    - ``model_type = "vanilla", "average"``
    - ``l1 = 0.0, 0.5, 1.0``
    - ``C = 0.1, 1.0, 10.0``


4. Train vanilla/average LR models with DP:
    ```
    python privacy/train_lr_dp.py --vanilla <n_splits> <n_repeats> <scaler_type> <epsilon>
    ```
    - ``n_splits = 5``
    - ``n_repeats = 20``
    - ``scaler_type = "standard"``
    - ``epsilon = 0.01, 0.1, 1.0, 10.0, 100.0, inf``


    ```
    python privacy/train_lr_dp.py --average <n_splits> <n_repeats> <n_models> <scaler_type> <epsilon>
    ```
    - ``n_splits = 3``
    - ``n_repeats = 20``
    - ``n_models = 100``
    - ``scaler_type = "standard"``
    - ``epsilon = 0.01, 0.1, 1.0, 10.0, 100.0, inf``


5. Train attacker models:
    ```
    python privacy/attacks.py --<model_type> --<attack_type> <model_name>
    ```
    - ``model_type = "vanilla", "average"``
    - ``attack_type = "bb", "wb"``
    - ``model_name = "lr", "lr_priv", "lr_dp", "tt"``
