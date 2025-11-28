# ml_random10k_eval.py
# -*- coding: utf-8 -*-
"""
Random undersampling (training) + Random-10k evaluation (testing)

This script keeps the original random undersampling strategy for *training*,
but changes the *evaluation* to: sample exactly N=10,000 items at random
(without class balancing) from the un-undersampled test pool (pos_test ∪ neg_test).
If the pool has fewer than N items, it uses the entire pool.

Exports:
  - <out_dir>/fold_results_random10k.csv          (per-fold, per-model metrics on Random-10k)
  - <out_dir>/agg_performance_random10k.csv       (mean±sd across folds)
  - <out_dir>/best_params_random10k.csv           (per-fold best hyperparameters)
"""

import argparse
import json
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve, auc, f1_score, matthews_corrcoef, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# ----------------------- Logging -----------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------- Models & Search Grids -----------------------
def make_models_params(seed: int):
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        "SVM": (
            SVC(probability=True, random_state=seed),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=seed),
            {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=seed,
                tree_method="hist",
                n_jobs=-1,
                use_label_encoder=False,
            ),
            {"n_estimators": [200, 500], "max_depth": [3, 6], "learning_rate": [0.03, 0.1]},
        ),
    }


# ----------------------- Fingerprints (MorganGenerator) -----------------------
def get_morgan_generator(radius: int, fp_size: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


def smiles_to_fp(smiles: str, gen, n_bits: int):
    if not isinstance(smiles, str):
        arr = np.zeros(n_bits, dtype=np.uint8)
        return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        arr = np.zeros(n_bits, dtype=np.uint8)
        return arr
    fp_bitvect = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp_bitvect, arr)
    return arr


# ----------------------- Random undersampling (for training) -----------------------
def random_select(neg_df: pd.DataFrame, n_target: int, random_state: int):
    n_target = int(min(max(n_target, 0), len(neg_df)))
    if n_target == 0:
        return neg_df.iloc[[]]
    if n_target >= len(neg_df):
        return neg_df.copy()
    return neg_df.sample(n=n_target, replace=False, random_state=random_state).copy()


# ----------------------- Metrics & scoring -----------------------
def evaluate_once(y_true, scores_or_proba, y_pred_binary):
    metrics = {}
    # Protections for degenerate scores
    if np.all(scores_or_proba == scores_or_proba[0]):
        metrics["ROC-AUC"] = 0.5
        prevalence = float(np.mean(y_true))
        metrics["PRC-AUC"] = prevalence
    else:
        metrics["ROC-AUC"] = roc_auc_score(y_true, scores_or_proba)
        prec, rec, _ = precision_recall_curve(y_true, scores_or_proba)
        metrics["PRC-AUC"] = auc(rec, prec)

    metrics["Accuracy"]  = accuracy_score(y_true, y_pred_binary)
    metrics["F1 Score"]  = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics["MCC"]       = matthews_corrcoef(y_true, y_pred_binary)
    metrics["Recall"]    = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics["Precision"] = precision_score(y_true, y_pred_binary, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    metrics["False Positives"]     = int(fp)
    metrics["False Positive Rate"] = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return metrics


def get_scores(best_model, X_test):
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X_test)[:, 1]
    else:
        df = best_model.decision_function(X_test)
        df = df.ravel()
        if df.max() == df.min():
            proba = np.full_like(df, 0.5, dtype=float)
        else:
            proba = (df - df.min()) / (df.max() - df.min())
    y_pred = (proba >= 0.5).astype(int)
    return proba, y_pred


# ----------------------- Core: Random-10k evaluation -----------------------
def run_random10k_eval(
    df: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
    fp_bits: int = 2048,
    fp_radius: int = 2,
    n_splits_outer: int = 5,
    inner_cv_splits: int = 5,
    neg_pos_ratio: float = 2.0,
    random_eval_n: int = 10_000,
):
    """
    Train with random undersampling (both positives & selected negatives in training),
    then evaluate on a random subset of size N from the *un-undersampled* test pool.

    The random subset ignores class labels entirely.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # Compute fingerprints once if not present
    if "fingerprint" not in df.columns:
        logging.info("Computing Morgan fingerprints ...")
        gen = get_morgan_generator(radius=fp_radius, fp_size=fp_bits)
        df = df.copy()
        df["fingerprint"] = df["smiles"].apply(lambda s: smiles_to_fp(s, gen, fp_bits))
        logging.info("Fingerprints computed.")

    # Split positives and negatives
    pos_df = df[df["antibiotic_activity"] == 1].reset_index(drop=True)
    neg_df = df[df["antibiotic_activity"] == 0].reset_index(drop=True)
    logging.info(f"Random-10k eval: pos={len(pos_df)}  neg={len(neg_df)}")

    models_params = make_models_params(seed)
    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)

    fold_rows, param_rows = [], []

    for fold_id, (pos_tr_idx, pos_te_idx) in enumerate(kf_outer.split(pos_df), start=1):
        logging.info(f"[random10k] Fold {fold_id}/{n_splits_outer}")

        pos_train = pos_df.iloc[pos_tr_idx].reset_index(drop=True)
        pos_test  = pos_df.iloc[pos_te_idx].reset_index(drop=True)

        # Negative split (un-undersampled pools)
        neg_train, neg_test = train_test_split(
            neg_df, test_size=0.2, random_state=seed + fold_id, shuffle=True
        )
        neg_train = neg_train.reset_index(drop=True)
        neg_test  = neg_test.reset_index(drop=True)

        # TRAIN: random undersampling for negatives
        n_neg_tr = int(round(neg_pos_ratio * len(pos_train)))
        neg_train_sel = random_select(neg_train, n_neg_tr, random_state=seed + fold_id)

        train_df = pd.concat([pos_train, neg_train_sel], ignore_index=True)

        # TEST POOL (un-undersampled): pos_test ∪ neg_test
        test_pool_df = pd.concat([pos_test, neg_test], ignore_index=True)

        # Build arrays
        X_train = np.vstack(train_df["fingerprint"].values)
        y_train = train_df["antibiotic_activity"].astype(int).values

        X_pool  = np.vstack(test_pool_df["fingerprint"].values)
        y_pool  = test_pool_df["antibiotic_activity"].astype(int).values
        n_pool  = len(test_pool_df)

        # Random-N subset from the pool (ignore class)
        rng = np.random.default_rng(seed + fold_id)
        n_eval = min(random_eval_n, n_pool)
        idx_eval = rng.choice(np.arange(n_pool), size=n_eval, replace=False)
        X_eval = X_pool[idx_eval]
        y_eval = y_pool[idx_eval]

        # Scale (fit on train only)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_eval_sc  = scaler.transform(X_eval)

        inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=seed)

        for model_name, (model, grid) in models_params.items():
            clf = GridSearchCV(
                estimator=model,
                param_grid=grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            clf.fit(X_train_sc, y_train)
            best_model = clf.best_estimator_

            proba, y_pred = get_scores(best_model, X_eval_sc)
            met = evaluate_once(y_eval, proba, y_pred)

            row = {
                "Strategy": "random_10k",
                "Fold": fold_id,
                "Model": model_name,
                "EvalSetSize": n_eval,
                **met,
            }
            fold_rows.append(row)

            param_rows.append({
                "Strategy": "random_10k",
                "Fold": fold_id,
                "Model": model_name,
                "best_params": json.dumps(clf.best_params_),
            })

    # Save per-fold details
    fold_df = pd.DataFrame(fold_rows)
    params_df = pd.DataFrame(param_rows)

    # Aggregate mean±sd
    metric_cols = [c for c in fold_df.columns if c not in {"Strategy", "Fold", "Model", "EvalSetSize"}]
    agg = fold_df.groupby(["Strategy", "Model"]).agg({mc: ["mean", "std"] for mc in metric_cols})
    agg.columns = [" ".join(col).strip() for col in agg.columns.values]

    def mean_sd_str(m, s):
        if pd.isna(m) or pd.isna(s):
            return "nan"
        return f"{m:.4f}±{s:.4f}"

    perf_table = pd.DataFrame(index=agg.index)
    for mc in metric_cols:
        mcol, scol = f"{mc} mean", f"{mc} std"
        if mcol in agg.columns and scol in agg.columns:
            perf_table[mc] = [mean_sd_str(m, s) for m, s in zip(agg[mcol], agg[scol])]

    # Write files
    fold_csv   = out_dir / "fold_results_random10k.csv"
    agg_csv    = out_dir / "agg_performance_random10k.csv"
    params_csv = out_dir / "best_params_random10k.csv"

    fold_df.to_csv(fold_csv, index=False)
    perf_table.to_csv(agg_csv)
    params_df.to_csv(params_csv, index=False)

    logging.info("Saved:")
    logging.info(f" - {fold_csv}")
    logging.info(f" - {agg_csv}")
    logging.info(f" - {params_csv}")


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Random undersampling + Random-10k evaluation")
    parser.add_argument("--data_csv", type=str, required=True, help="CSV with 'smiles' and 'antibiotic_activity'.")
    parser.add_argument("--out_dir", type=str, default="./ml_outputs2", help="Output directory.")
    parser.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint size.")
    parser.add_argument("--fp_radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--splits", type=int, default=5, help="Outer KFold splits (on positives).")
    parser.add_argument("--inner_cv", type=int, default=5, help="Inner CV splits for GridSearchCV.")
    parser.add_argument("--neg_pos_ratio", type=float, default=2.0, help="Target negatives = ratio * #positives (training).")
    parser.add_argument("--random_eval_n", type=int, default=10000, help="Random evaluation subset size from test pool.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_csv)
    required = {"smiles", "antibiotic_activity"}
    missing = required - set(df.columns)
    assert not missing, f"CSV missing required columns: {missing}"

    run_random10k_eval(
        df=df,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius,
        n_splits_outer=args.splits,
        inner_cv_splits=args.inner_cv,
        neg_pos_ratio=args.neg_pos_ratio,
        random_eval_n=args.random_eval_n,
    )


if __name__ == "__main__":
    main()
