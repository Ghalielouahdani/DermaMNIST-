import argparse
import torch
import numpy as np
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes
import time
import itertools

def main(args):
    # Original Step 1: Load data (this is common to both HPO and normal run)
    xtrain_orig, xtest_orig, ytrain_orig, y_test_orig = load_data()

    # --- ADDED: MLP Hyperparameter Optimization Block ---
    if args.nn_type == "mlp" and args.tune_mlp_hyperparams:
        print("Starting MLP hyperparameter tuning...")
        print("NOTE: MLP dropout rate is fixed at 0.5 as per its implementation and cannot be tuned by this script without modifying MLP class.")
        
        # Use a fixed seed for reproducibility of the validation split during HPO.
        np.random.seed(42) 

        # Create a fixed validation split from the original training data for HPO.
        # This split is used for all HPO trials.
        # args.test determines if the "validation" set for HPO is the actual test set.
        if not args.test: # Standard HPO: use a validation split from training data
            hpo_idx = np.random.permutation(len(xtrain_orig))
            hpo_split_point = int(0.8 * len(xtrain_orig))
            hpo_train_idx, hpo_val_idx = hpo_idx[:hpo_split_point], hpo_idx[hpo_split_point:]
            x_tr_hpo_base, y_tr_hpo_base = xtrain_orig[hpo_train_idx], ytrain_orig[hpo_train_idx]
            x_val_hpo_base, y_val_hpo_base = xtrain_orig[hpo_val_idx], ytrain_orig[hpo_val_idx]
            hpo_eval_set_name = "Validation"
        else: # HPO, but evaluating performance on the test set (as per args.test)
            x_tr_hpo_base, y_tr_hpo_base = xtrain_orig, ytrain_orig # Train on full training data
            x_val_hpo_base, y_val_hpo_base = xtest_orig, y_test_orig # Evaluate on test data
            hpo_eval_set_name = "Test (during HPO)"

        # Preprocess MLP data for HPO (based on the fixed x_tr_hpo_base)
        x_tr_hpo_reshaped = x_tr_hpo_base.reshape(x_tr_hpo_base.shape[0], -1)
        x_val_hpo_reshaped = x_val_hpo_base.reshape(x_val_hpo_base.shape[0], -1)
        
        mean_hpo_tr = x_tr_hpo_reshaped.mean(axis=0)
        std_hpo_tr = x_tr_hpo_reshaped.std(axis=0) + 1e-8
        
        x_tr_hpo_processed = (x_tr_hpo_reshaped - mean_hpo_tr) / std_hpo_tr
        x_val_hpo_processed = (x_val_hpo_reshaped - mean_hpo_tr) / std_hpo_tr

        # Define hyperparameter search space for MLP
        learning_rates_to_try = args.hpo_lr_range if args.hpo_lr_range else [1e-3, 1e-4, 1e-5]
        weight_decays_to_try = args.hpo_wd_range if args.hpo_wd_range else [1e-3, 1e-4, 0.0]
        max_iters_to_try = args.hpo_epochs_range if args.hpo_epochs_range else [30, 50, 75]

        best_f1_hpo = -1.0
        best_params_hpo = {}
        
        original_args_lr = args.lr
        original_args_max_iters = args.max_iters
        original_args_weight_decay = args.weight_decay

        param_combinations = list(itertools.product(learning_rates_to_try, weight_decays_to_try, max_iters_to_try))
        print(f"MLP HPO: Total combinations to try: {len(param_combinations)}")

        for i, (lr_trial, wd_trial, mi_trial) in enumerate(param_combinations):
            print(f"\n--- HPO Trial {i+1}/{len(param_combinations)} for MLP ---")
            print(f"Params: LR={lr_trial}, WeightDecay={wd_trial}, MaxIters={mi_trial}")
            
            args.lr = lr_trial
            args.max_iters = mi_trial
            args.weight_decay = wd_trial

            # --- Replicated Model Init, Training, Prediction, F1 Calc (Steps 4-7 from original) for MLP ---
            n_classes_hpo = get_n_classes(y_tr_hpo_base) 
            model_hpo = MLP(x_tr_hpo_processed.shape[1], n_classes_hpo) 
            device_hpo = torch.device(args.device if (args.device!="cpu" and torch.cuda.is_available()) or args.device=="mps" else "cpu")
            model_hpo.to(device_hpo)

            trainer_hpo = Trainer(model=model_hpo,
                                  lr=args.lr, 
                                  epochs=args.max_iters, 
                                  batch_size=args.nn_batch_size, 
                                  device=device_hpo,
                                  weight_decay=args.weight_decay)
            
            print(f"Training MLP for HPO trial {i+1} with {args.max_iters} epochs...")
            trainer_hpo.fit(x_tr_hpo_processed, y_tr_hpo_base)
            
            print(f"Predicting on HPO {hpo_eval_set_name} set for trial {i+1}...")
            preds_val_hpo_trial = trainer_hpo.predict(x_val_hpo_processed)

            f1_val_hpo_trial = macrof1_fn(preds_val_hpo_trial, y_val_hpo_base)
            print(f"HPO Trial {i+1} - {hpo_eval_set_name} F1 Score: {f1_val_hpo_trial:.6f}")

            if f1_val_hpo_trial > best_f1_hpo:
                best_f1_hpo = f1_val_hpo_trial
                best_params_hpo = {'lr': args.lr, 'weight_decay': args.weight_decay, 'max_iters': args.max_iters}
                print(f"*** New best F1 found for MLP HPO: {best_f1_hpo:.6f} with params: {best_params_hpo} ***")
            # --- End of Replicated Block for HPO Trial ---

        args.lr = original_args_lr
        args.max_iters = original_args_max_iters
        args.weight_decay = original_args_weight_decay

        print("\n--- MLP Hyperparameter Tuning Concluded ---")
        if best_params_hpo:
            print(f"Best {hpo_eval_set_name} F1 Score for MLP: {best_f1_hpo:.6f}")
            print(f"Best Parameters found: {best_params_hpo}")
            print("Note: MLP dropout_rate is fixed at 0.5 as per its implementation.")
            print(f"To run with these best parameters for MLP (and default dropout 0.5):")
            print(f"python main.py --nn_type mlp --lr {best_params_hpo['lr']} --weight_decay {best_params_hpo['weight_decay']} --max_iters {best_params_hpo['max_iters']}")
        else:
            print("No successful HPO trials completed or no improvement found.")
        
        return # Exit main after HPO block is finished
    # --- END OF ADDED: MLP Hyperparameter Optimization Block ---

    # --- Original main.py logic (Steps 2-7) continues below if not in HPO mode ---
    # Ensure original xtrain, xtest, ytrain, y_test are used
    xtrain, xtest, ytrain, y_test = xtrain_orig, xtest_orig, ytrain_orig, y_test_orig

    # 2. Validation split
    if not args.test:
        idx = np.random.permutation(len(xtrain))
        split = int(0.8 * len(xtrain))
        train_idx, val_idx = idx[:split], idx[split:]
        x_tr, y_tr = xtrain[train_idx], ytrain[train_idx]
        x_val, y_val = xtrain[val_idx], ytrain[val_idx]
    else:
        x_tr, y_tr = xtrain, ytrain
        x_val, y_val = xtest, y_test

    # 3. Preprocessing & reshape
    if args.nn_type == "mlp":
        x_tr_processed = x_tr.reshape(x_tr.shape[0], -1)
        x_val_processed = x_val.reshape(x_val.shape[0], -1)
        xtest_proc = xtest.reshape(xtest.shape[0], -1)
        # normalize using training set stats
        mean = x_tr_processed.mean(axis=0)
        std = x_tr_processed.std(axis=0) + 1e-8
        x_tr_processed = (x_tr_processed - mean) / std
        x_val_processed = (x_val_processed - mean) / std
        x_test_final = (xtest_proc - mean) / std
    else:  # cnn
        # from (N,H,W,C) to (N,C,H,W) and scale to [0,1]
        x_tr_processed = x_tr.transpose(0,3,1,2).astype(np.float32) / 255.0
        x_val_processed = x_val.transpose(0,3,1,2).astype(np.float32) / 255.0
        x_test_final = xtest.transpose(0,3,1,2).astype(np.float32) / 255.0

    # 4. Model init & device
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(x_tr_processed.shape[1], n_classes)
    else:
        model = CNN(x_tr_processed.shape[1], n_classes, dropout_rate=args.dropout_rate)
    device = torch.device(args.device if (args.device!="cpu" and torch.cuda.is_available()) or args.device=="mps" else "cpu")
    model.to(device)
    summary(model)

    # 5. Trainer
    trainer = Trainer(model=model,
                      lr=args.lr,
                      epochs=args.max_iters,
                      batch_size=args.nn_batch_size,
                      device=device,
                      weight_decay=args.weight_decay)

    # 6. Train & evaluate
    start_train = time.time()
    preds_train = trainer.fit(x_tr_processed, y_tr)
    end_train = time.time()
    print(f"Training took {end_train - start_train:.2f} seconds")
    
    # Determine which data to use for prediction based on --test flag
    # If --test, x_val and y_val are the test set. x_val_processed is then processed test data.
    # If not --test, x_val and y_val are the validation set. x_val_processed is then processed validation data.
    # The original code uses x_val for trainer.predict if not test, which is x_val_processed.
    # And if test, it also implies x_val_processed (which would be derived from xtest) for prediction.
    # The variable x_test_final is what should be used if predicting on the full test set after training.
    # For evaluating on the validation set (if not args.test):
    predict_data = x_val_processed 
    true_labels_for_eval = y_val
    
    if args.test: # If --test is set, we evaluate on the actual test set
        predict_data = x_test_final 
        true_labels_for_eval = y_test

    start_pred = time.time()
    preds_eval   = trainer.predict(predict_data)
    end_pred = time.time()
    print(f"Prediction took {end_pred - start_pred:.2f} seconds")

    # 7. Report
    acc_train = accuracy_fn(preds_train, y_tr)
    f1_train  = macrof1_fn(preds_train, y_tr)
    print(f"\nTrain set:      accuracy = {acc_train:.3f}% - F1 = {f1_train:.6f}")
    
    set_name = "Test" if args.test else "Validation"
    acc_eval   = accuracy_fn(preds_eval, true_labels_for_eval)
    f1_eval    = macrof1_fn(preds_eval, true_labels_for_eval)
    print(f"{set_name} set:   accuracy = {acc_eval:.3f}% - F1 = {f1_eval:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str)
    parser.add_argument('--nn_type', default="mlp", choices=["mlp","cnn"])
    parser.add_argument('--nn_batch_size', type=int, default=64)
    parser.add_argument('--device', default="cpu", choices=["cpu","cuda","mps"])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate for CNN. MLP uses a fixed 0.5 dropout.")
    
    # --- ADDED: Arguments for MLP Hyperparameter Optimization ---
    parser.add_argument('--tune_mlp_hyperparams', action='store_true', help='Enable hyperparameter tuning for MLP.')
    parser.add_argument('--hpo_lr_range', nargs='+', type=float, help='List of learning rates for MLP HPO (e.g., 0.001 0.0001).')
    parser.add_argument('--hpo_wd_range', nargs='+', type=float, help='List of weight decay values for MLP HPO (e.g., 0.001 0.0).')
    parser.add_argument('--hpo_epochs_range', nargs='+', type=int, help='List of epoch counts for MLP HPO (e.g., 30 50).')

    args = parser.parse_args()
    main(args)