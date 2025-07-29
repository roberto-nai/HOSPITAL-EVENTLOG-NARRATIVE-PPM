"""
02_embedding_multi.py
This script is designed to handle multi-class classification tasks using embeddings.
It includes functionality for loading narratives, generating embeddings, training classifiers, and evaluating performance.
"""
import json
import time
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import os
import sys
import logging
import warnings
from dotenv import load_dotenv
import numpy as np
from local_functions import ensure_dir_with_gitkeep

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# HYPERPARAMETER TUNING SWITCH
HT_DO = 1  # 0 = no hyperparameter tuning, 1 = perform hyperparameter tuning

if HT_DO:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Setup logging
LOG_FILE = Path(sys.argv[0]).with_suffix(".log")
# LOG_FILE.write_text("")  # Reset log file
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__} in {filename}:{lineno} - {message}")

warnings.showwarning = custom_warning_handler

# Config
# input
NARRATIVE_DIR = Path("event_log_narratives")
INPUT_FILE = f"orbassano_narratives.csv"
INPUT_NARRATIVES = NARRATIVE_DIR / INPUT_FILE
N_NARRATIVES = -1
CSV_SEP = ";"
OUTCOME_COL = "true_outcome"
VALID_OUTCOMES = ["A domicilio", "Ricoverato"]
TEST_SPLIT = 0.2
SPLIT_SEED = 42
# output
OUTPUT_DIR = Path("output_classifiers")  # directory for classifier outputs
OUTPUT_HT_DIR = Path("output_classifiers_ht") # directory for hyperparameter tuning outputs
METRICS_CSV = OUTPUT_DIR / "embedding_classifier_metrics.csv"

def load_narratives_csv(path: Path, sep: str = CSV_SEP) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)

def balance_dataset(df: pd.DataFrame, label_col: str, n: int, seed: int = 42) -> pd.DataFrame:
    outcome_counts = df[label_col].value_counts()
    per_outcome = min(n // 2, outcome_counts.min()) if n > 0 else outcome_counts.min()
    balanced = [
        df[df[label_col] == outcome].sample(n=per_outcome, random_state=seed)
        for outcome in outcome_counts.index
    ]
    return pd.concat(balanced).sample(frac=1, random_state=seed).reset_index(drop=True)

def save_metrics_to_csv(classifier_name: str, model_name: str, y_true: List[int], y_pred: List[int], output_path: Path, n: int, seconds: float):
    file_exists = output_path.exists()
    df = pd.DataFrame([{
        "classifier": classifier_name,
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "ht": HT_DO,
        "dataset_len": n,
        "execution_time": round(seconds, 2)
    }])
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden1=256, hidden2=64, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperopt tuning functions (only used if HT_DO == 1)
if HT_DO:
    def optimise_rf(X_train, y_train, X_val, y_val, rstate=None):
        def objective(params):
            clf = RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                random_state=SPLIT_SEED,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            return {'loss': -f1, 'status': STATUS_OK}

        space = {
            'n_estimators': hp.quniform('n_estimators', 20, 100, 10),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
        }
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=15, trials=trials)
        best = {k: int(v) for k, v in best.items()}
        return best

    def optimise_xgb(X_train, y_train, X_val, y_val, rstate=None):
        def objective(params):
            clf = XGBClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                learning_rate=float(params['learning_rate']),
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=SPLIT_SEED,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            return {'loss': -f1, 'status': STATUS_OK}

        space = {
            'n_estimators': hp.quniform('n_estimators', 20, 100, 10),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)
        }
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=15, trials=trials)
        best['n_estimators'] = int(best['n_estimators'])
        best['max_depth'] = int(best['max_depth'])
        return best

    def optimise_nn(X_train, y_train, X_val, y_val, input_dim, rstate = None):
        def objective(params):
            hidden1 = int(params['hidden1'])
            hidden2 = int(params['hidden2'])
            dropout1 = float(params['dropout1'])
            dropout2 = float(params['dropout2'])
            lr = float(params['lr'])
            batch_size = int(params['batch_size'])

            model_nn = FeedForwardNN(input_dim, hidden1, hidden2, dropout1, dropout2)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model_nn.parameters(), lr=lr)

            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model_nn.train()
            for epoch in range(3):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    output = model_nn(xb)
                    loss = criterion(output, yb)
                    loss.backward()
                    optimizer.step()

            model_nn.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_pred_probs = model_nn(X_val_tensor).squeeze()
                y_pred_nn = (y_pred_probs >= 0.5).int().tolist()
            f1 = f1_score(y_val, y_pred_nn, zero_division=0)
            return {'loss': -f1, 'status': STATUS_OK}

        space = {
            'hidden1': hp.quniform('hidden1', 64, 256, 32),
            'hidden2': hp.quniform('hidden2', 16, 128, 16),
            'dropout1': hp.uniform('dropout1', 0.1, 0.5),
            'dropout2': hp.uniform('dropout2', 0.1, 0.5),
            'lr': hp.loguniform('lr', -5, -2),
            'batch_size': hp.quniform('batch_size', 16, 64, 16)
        }
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
        best['hidden1'] = int(best['hidden1'])
        best['hidden2'] = int(best['hidden2'])
        best['batch_size'] = int(best['batch_size'])
        best['dropout1'] = float(best['dropout1'])
        best['dropout2'] = float(best['dropout2'])
        best['lr'] = float(best['lr'])
        return best

if __name__ == "__main__":
    try:
        start_time = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="all-mpnet-base-v2", help="SentenceTransformer model name")
        args = parser.parse_args()

        rstate = np.random.RandomState(SPLIT_SEED)

        print(f"Script started at {time.ctime(start_time)}")

        # Output directories
        ensure_dir_with_gitkeep(OUTPUT_HT_DIR)
        ensure_dir_with_gitkeep(OUTPUT_DIR)

        MODEL_NAME = args.model
        print(f"Loading SentenceTransformer model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)

        print("\nLoading narratives from CSV...")
        df = load_narratives_csv(INPUT_NARRATIVES)
        print(f"Initial dataset size: {len(df)}")

        print("\nBalancing dataset...")
        df_balanced = balance_dataset(df, OUTCOME_COL, N_NARRATIVES, SPLIT_SEED)
        print(df_balanced[OUTCOME_COL].value_counts())

        print("\nSplitting dataset...")
        train_df, test_df = train_test_split(df_balanced, test_size=TEST_SPLIT, random_state=SPLIT_SEED, stratify=df_balanced[OUTCOME_COL])
        # For hyperopt, split train into train/val
        if HT_DO:
            train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SPLIT_SEED, stratify=train_df[OUTCOME_COL])

        print("\nGenerating embeddings...")
        X_train = model.encode(train_df["narrative"].tolist(), convert_to_numpy=True)
        y_train = [1 if o == "Ricoverato" else 0 for o in train_df[OUTCOME_COL]]

        X_test = model.encode(test_df["narrative"].tolist(), convert_to_numpy=True)
        y_test = [1 if o == "Ricoverato" else 0 for o in test_df[OUTCOME_COL]]

        if HT_DO:
            X_val = model.encode(val_df["narrative"].tolist(), convert_to_numpy=True)
            y_val = [1 if o == "Ricoverato" else 0 for o in val_df[OUTCOME_COL]]

        classifiers = {
            "RF": RandomForestClassifier(random_state=SPLIT_SEED),
            "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SPLIT_SEED)
        }

        for clf_name, clf in classifiers.items():
            clf_start = time.time()
            print(f"\nTraining classifier: {clf_name}...")

            if HT_DO:
                if clf_name == "RF":
                    best_rf = optimise_rf(X_train, y_train, X_val, y_val, rstate)
                    print(f"Best RF params: {best_rf}")
                    clf = RandomForestClassifier(**best_rf, random_state=SPLIT_SEED, n_jobs=-1)
                    path_ht = OUTPUT_HT_DIR / f"best_params_RF_{MODEL_NAME}.json"
                    print(f"Saving best RF params to {path_ht}")
                    with open(path_ht, "w") as f:
                        json.dump(best_rf, f, indent=2)
                elif clf_name == "XGB":
                    best_xgb = optimise_xgb(X_train, y_train, X_val, y_val, rstate)
                    print(f"Best XGB params: {best_xgb}")
                    clf = XGBClassifier(**best_xgb, use_label_encoder=False, eval_metric='logloss', random_state=SPLIT_SEED, n_jobs=-1)
                    path_ht = OUTPUT_HT_DIR / f"best_params_XGB_{MODEL_NAME}.json"
                    print(f"Saving best XGB params to {path_ht}")
                    with open(path_ht, "w") as f:
                        json.dump(best_xgb, f, indent=2)

            clf.fit(X_train, y_train)

            print(f"\nEvaluating {clf_name}...")
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred, target_names=["A domicilio", "Ricoverato"]))

            save_metrics_to_csv(clf_name, MODEL_NAME, y_test, y_pred, METRICS_CSV, len(df_balanced), time.time() - clf_start)

        print("\nTraining classifier: FeedForwardNN...")
        input_dim = X_train.shape[1]

        if HT_DO:
            best_nn = optimise_nn(X_train, y_train, X_val, y_val, input_dim, rstate)
            print(f"Best FFNN params: {best_nn}")
            model_nn = FeedForwardNN(input_dim, best_nn['hidden1'], best_nn['hidden2'], best_nn['dropout1'], best_nn['dropout2'])
            lr = best_nn['lr']
            batch_size = best_nn['batch_size']
            path_ht = OUTPUT_HT_DIR / f"best_params_FFNN_{MODEL_NAME}.json"
            print(f"Saving best FFNN params to {path_ht}")
            with open(path_ht, "w") as f:
                json.dump(best_nn, f, indent=2)
        else:
            model_nn = FeedForwardNN(input_dim)
            lr = 0.001
            batch_size = 32

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model_nn.parameters(), lr=lr)

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        nn_start = time.time()
        model_nn.train()
        for epoch in range(5):
            for xb, yb in loader:
                optimizer.zero_grad()
                output = model_nn(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

        model_nn.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred_probs = model_nn(X_test_tensor).squeeze()
            y_pred_nn = (y_pred_probs >= 0.5).int().tolist()

        print(classification_report(y_test, y_pred_nn, target_names=["A domicilio", "Ricoverato"]))
        save_metrics_to_csv("FFNN", MODEL_NAME, y_test, y_pred_nn, METRICS_CSV, len(df_balanced), time.time() - nn_start)

        print("\nSorting metrics by F1-score...")
        metrics_df = pd.read_csv(METRICS_CSV)
        metrics_df = metrics_df.sort_values(by="f1_score", ascending=False)
        metrics_df.to_csv(METRICS_CSV, index=False)

        print("\nAll done.")

    except Exception:
        logging.exception("Unhandled exception occurred during execution.")
        print(f"An error occurred. Check the log file: {LOG_FILE}")