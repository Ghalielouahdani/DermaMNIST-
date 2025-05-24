import argparse
import torch
import numpy as np
from torchinfo import summary
from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes
import time

def main(args):
    # 1. Load data
    xtrain, xtest, ytrain, y_test = load_data()

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
        x_tr = x_tr.reshape(x_tr.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        xtest_proc = xtest.reshape(xtest.shape[0], -1)
        # normalize using training set stats
        mean = x_tr.mean(axis=0)
        std = x_tr.std(axis=0) + 1e-8
        x_tr = (x_tr - mean) / std
        x_val = (x_val - mean) / std
        x_test_final = (xtest_proc - mean) / std
    else:  # cnn
        # from (N,H,W,C) to (N,C,H,W) and scale to [0,1]
        x_tr = x_tr.transpose(0,3,1,2).astype(np.float32) / 255.0
        x_val = x_val.transpose(0,3,1,2).astype(np.float32) / 255.0
        x_test_final = xtest.transpose(0,3,1,2).astype(np.float32) / 255.0

    # 4. Model init & device
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(x_tr.shape[1], n_classes)
    else:
        model = CNN(x_tr.shape[1], n_classes)
    device = torch.device(args.device if (args.device!="cpu" and torch.cuda.is_available()) or args.device=="mps" else "cpu")
    model.to(device)
    summary(model)

    # 5. Trainer
    trainer = Trainer(model=model,
                      lr=args.lr,
                      epochs=args.max_iters,
                      batch_size=args.nn_batch_size,
                      device=device)

    # 6. Train & evaluate
    start_train = time.time()
    preds_train = trainer.fit(x_tr, y_tr)
    end_train = time.time()
    print(f"Training took {end_train - start_train:.2f} seconds")
    start_pred = time.time()
    preds_val   = trainer.predict(x_val)
    end_pred = time.time()
    print(f"Prediction took {end_pred - start_pred:.2f} seconds")

    # 7. Report
    acc_train = accuracy_fn(preds_train, y_tr)
    f1_train  = macrof1_fn(preds_train, y_tr)
    print(f"\nTrain set:      accuracy = {acc_train:.3f}% - F1 = {f1_train:.6f}")
    set_name = "Test" if args.test else "Validation"
    acc_val   = accuracy_fn(preds_val, y_val)
    f1_val    = macrof1_fn(preds_val, y_val)
    print(f"{set_name} set:   accuracy = {acc_val:.3f}% - F1 = {f1_val:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str)
    parser.add_argument('--nn_type', default="mlp", choices=["mlp","cnn"])
    parser.add_argument('--nn_batch_size', type=int, default=64)
    parser.add_argument('--device', default="cpu", choices=["cpu","cuda","mps"])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()
    main(args)