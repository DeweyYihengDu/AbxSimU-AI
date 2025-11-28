# ml_random10k_repeats.py
# -*- coding: utf-8 -*-
"""
Random undersampling (training) + Random-10k evaluation with repeats (testing)

Key behavior:
- By default we DO NOT fix any random seed. All splits/sampling use system entropy,
  so each run will produce different random subsets. If reproducibility is wanted,
  pass an explicit --seed integer.

Training:
  - Random undersampling on TRAIN negatives (n_neg = ratio * #pos_in_train).
  - Inner CV (GridSearchCV + StratifiedKFold) selects best hyperparams by ROC-AUC.
  - StandardScaler fitted on training only.

Evaluation ("Random-10k"):
  - From the UN-undersampled test pool, uniformly sample N=10,000 molecules
    WITHOUT considering labels (use entire pool if less than 10k).
  - Repeat this sampling R times per outer fold (each repeat uses fresh entropy).
  - Report metrics per repeat and aggregated mean±sd and 95% CI across folds×repeats.
  - Record subset prevalence and AP-Lift = AP / prevalence.

Exports:
  - <out_dir>/fold_results_random10k_repeats.csv
  - <out_dir>/agg_performance_random10k_repeats.csv
  - <out_dir>/agg_performance_random10k_repeats_ci.csv
  - <out_dir>/best_params_random10k_repeats.csv
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

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Sklearn + XGBoost
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

from typing import Optional


# ---------------------------- Logging ----------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------- Models & grids ----------------------------
def _rand_int32_from_entropy() -> int:
    """Generate a 32-bit-ish random integer from OS entropy (no fixed seed)."""
    return int(np.random.SeedSequence().entropy % (2**31 - 1))

def make_models_params(seed: Optional[int]):
    """
    If seed is None => use OS entropy to create a run-specific random_state
    (esp. for XGBoost which defaults to 0).
    """
    run_state = seed if seed is not None else _rand_int32_from_entropy()

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=run_state, solver="lbfgs"),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        "SVM": (
            SVC(probability=True, random_state=run_state),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=run_state, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=run_state),
            {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=run_state,   # avoid XGB default 0
                tree_method="hist",
                n_jobs=-1,
                use_label_encoder=False,
            ),
            {"n_estimators": [200, 500], "max_depth": [3, 6], "learning_rate": [0.03, 0.1]},
        ),
    }


# ---------------------------- Fingerprints ----------------------------
def get_morgan_generator(radius: int, fp_size: int):
    """RDKit MorganGenerator wrapper (avoids deprecation warnings)."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

def smiles_to_fp(smiles: str, gen, n_bits: int):
    if not isinstance(smiles, str):
        arr = np.zeros(n_bits, dtype=np.uint8); return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        arr = np.zeros(n_bits, dtype=np.uint8); return arr
    fp_bitvect = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp_bitvect, arr)
    return arr


# ---------------------------- Undersampling (training) ----------------------------
def random_select(neg_df: pd.DataFrame, n_target: int):
    """Randomly select negatives WITHOUT a fixed seed; each call uses fresh OS entropy."""
    n_target = int(min(max(n_target, 0), len(neg_df)))
    if n_target == 0:
        return neg_df.iloc[[]]
    if n_target >= len(neg_df):
        return neg_df.copy()
    rs = _rand_int32_from_entropy()
    return neg_df.sample(n=n_target, replace=False, random_state=rs).copy()


# ---------------------------- Metrics ----------------------------
def evaluate_once(y_true, scores_or_proba, y_pred_binary):
    """Return a dict of all required metrics, robust to degenerate scores."""
    metrics = {}
    if scores_or_proba.max() == scores_or_proba.min():
        metrics["ROC-AUC"] = 0.5
        prevalence = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
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
    """Get probability-like scores and 0/1 predictions with threshold 0.5."""
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

def ap_lift(ap_value: float, prevalence: float):
    return np.nan if prevalence is None or prevalence <= 0 else ap_value / prevalence


# ---------------------------- Core routine ----------------------------
def run_random10k_repeats(
    df: pd.DataFrame,
    out_dir: Path,
    seed: Optional[int] = None,
    fp_bits: int = 2048,
    fp_radius: int = 2,
    n_splits_outer: int = 5,
    inner_cv_splits: int = 5,
    neg_pos_ratio: float = 2.0,
    random_eval_n: int = 10_000,
    random_eval_repeats: int = 30,
):
    """
    Train with random undersampling (training negatives),
    then evaluate on R random-10k subsets drawn from the un-undersampled test pool.
    By default, NO fixed seeds are set (seed=None) so results vary across runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprints
    if "fingerprint" not in df.columns:
        logging.info("Computing Morgan fingerprints ...")
        # FIX: pass fp_size (not fpSize) to our wrapper
        gen = get_morgan_generator(radius=fp_radius, fp_size=fp_bits)
        df = df.copy()
        df["fingerprint"] = df["smiles"].apply(lambda s: smiles_to_fp(s, gen, fp_bits))
        logging.info("Fingerprints computed.")

    # Split by label
    pos_df = df[df["antibiotic_activity"] == 1].reset_index(drop=True)
    neg_df = df[df["antibiotic_activity"] == 0].reset_index(drop=True)
    logging.info(f"Dataset: pos={len(pos_df)}  neg={len(neg_df)}")

    models_params = make_models_params(seed)

    # Outer KFold on positives; shuffle=True; random_state = seed (or None)
    kf_outer = KFold(
        n_splits=n_splits_outer,
        shuffle=True,
        random_state=seed  # None => new randomness each run
    )

    fold_rows = []
    param_rows = []

    for fold_id, (pos_tr_idx, pos_te_idx) in enumerate(kf_outer.split(pos_df), start=1):
        logging.info(f"[Random10k-Repeats] Fold {fold_id}/{n_splits_outer}")

        pos_train = pos_df.iloc[pos_tr_idx].reset_index(drop=True)
        pos_test  = pos_df.iloc[pos_te_idx].reset_index(drop=True)

        # Independent split on negatives (no fixed random_state unless seed is provided)
        neg_train, neg_test = train_test_split(
            neg_df, test_size=0.2, random_state=(seed + fold_id) if seed is not None else None, shuffle=True
        )
        neg_train = neg_train.reset_index(drop=True)
        neg_test  = neg_test.reset_index(drop=True)

        # TRAIN: random undersampling on negatives (no fixed seed)
        n_neg_tr = int(round(neg_pos_ratio * len(pos_train)))
        neg_train_sel = random_select(neg_train, n_neg_tr)

        train_df = pd.concat([pos_train, neg_train_sel], ignore_index=True)

        # TEST POOL (un-undersampled): pos_test ∪ neg_test
        test_pool_df = pd.concat([pos_test, neg_test], ignore_index=True)
        X_pool = np.vstack(test_pool_df["fingerprint"].values)
        y_pool = test_pool_df["antibiotic_activity"].astype(int).values
        n_pool = len(test_pool_df)

        # Prepare training arrays
        X_train = np.vstack(train_df["fingerprint"].values)
        y_train = train_df["antibiotic_activity"].astype(int).values

        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_pool_sc  = scaler.transform(X_pool)

        # Inner CV (no fixed seed unless provided)
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
            param_rows.append({
                "Strategy": "random_10k_repeats",
                "Fold": fold_id,
                "Model": model_name,
                "best_params": json.dumps(clf.best_params_),
            })

            # Repeated Random-10k evaluation: each repeat uses fresh entropy
            for rep in range(1, random_eval_repeats + 1):
                n_eval = min(random_eval_n, n_pool)
                rng_rep = np.random.default_rng()   # fresh entropy each repeat
                idx_eval = rng_rep.choice(n_pool, size=n_eval, replace=False)

                X_eval = X_pool_sc[idx_eval]
                y_eval = y_pool[idx_eval]

                prev = float(np.mean(y_eval)) if len(y_eval) > 0 else 0.0
                proba, y_pred = get_scores(best_model, X_eval)
                met = evaluate_once(y_eval, proba, y_pred)
                met["AP-Lift"] = ap_lift(met["PRC-AUC"], prev)

                row = {
                    "Strategy": "random_10k_repeats",
                    "Fold": fold_id,
                    "Repeat": rep,
                    "Model": model_name,
                    "EvalSetSize": int(n_eval),
                    "Subset #Pos": int(y_eval.sum()),
                    "Subset #Neg": int((1 - y_eval).sum()),
                    "Subset Prevalence": prev,
                    **met,
                }
                fold_rows.append(row)

    # Save per-fold×repeat details
    fold_df = pd.DataFrame(fold_rows)
    params_df = pd.DataFrame(param_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_csv   = out_dir / "fold_results_random10k_repeats.csv"
    params_csv = out_dir / "best_params_random10k_repeats.csv"
    fold_df.to_csv(fold_csv, index=False)
    params_df.to_csv(params_csv, index=False)
    logging.info(f"Saved per-repeat details: {fold_csv}")
    logging.info(f"Saved best params:       {params_csv}")

    # Aggregate: mean±sd across (fold, repeat)
    metric_cols = [
        "ROC-AUC", "PRC-AUC", "AP-Lift", "Accuracy", "F1 Score", "MCC",
        "Recall", "Precision", "False Positives", "False Positive Rate",
        "Subset Prevalence"
    ]
    agg = fold_df.groupby(["Strategy", "Model"]).agg({mc: ["mean", "std", "count"] for mc in metric_cols})
    agg.columns = [" ".join(col).strip() for col in agg.columns.values]

    def mean_sd_str(m, s):
        return "nan" if (pd.isna(m) or pd.isna(s)) else f"{m:.4f}±{s:.4f}"

    perf_table = pd.DataFrame(index=agg.index)
    for mc in metric_cols:
        mcol, scol = f"{mc} mean", f"{mc} std"
        if mcol in agg.columns and scol in agg.columns:
            perf_table[mc] = [mean_sd_str(m, s) for m, s in zip(agg[mcol], agg[scol])]

    agg_csv = out_dir / "agg_performance_random10k_repeats.csv"
    perf_table.to_csv(agg_csv)
    logging.info(f"Saved mean±sd table:     {agg_csv}")

    # CI table
    ci_rows = []
    for (strategy, model), sub in fold_df.groupby(["Strategy", "Model"]):
        for mc in metric_cols:
            vals = sub[mc].dropna().values
            N = len(vals)
            if N == 0:
                continue
            mean = float(np.mean(vals))
            sd   = float(np.std(vals, ddof=1)) if N > 1 else 0.0
            sem  = sd / np.sqrt(N) if N > 1 else 0.0
            ci_low = mean - 1.96 * sem
            ci_up  = mean + 1.96 * sem
            ci_rows.append({
                "Strategy": strategy,
                "Model": model,
                "Metric": mc,
                "Mean": mean,
                "SD": sd,
                "N": N,
                "95% CI Lower": ci_low,
                "95% CI Upper": ci_up
            })
    ci_df = pd.DataFrame(ci_rows)
    ci_csv = out_dir / "agg_performance_random10k_repeats_ci.csv"
    ci_df.to_csv(ci_csv, index=False)
    logging.info(f"Saved CI table:          {ci_csv}")


# ---------------------------- CLI ----------------------------
def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Random undersampling (train) + Random-10k repeated evaluation (test)")
    parser.add_argument("--data_csv", type=str, required=True, help="CSV with 'smiles' and 'antibiotic_activity'.")
    parser.add_argument("--out_dir", type=str, default="./ml_outputs2", help="Output directory.")

    parser.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint size.")
    parser.add_argument("--fp_radius", type=int, default=2, help="Morgan fingerprint radius.")

    parser.add_argument("--splits", type=int, default=5, help="Outer KFold splits (on positives).")
    parser.add_argument("--inner_cv", type=int, default=5, help="Inner CV splits for GridSearchCV.")
    parser.add_argument("--neg_pos_ratio", type=float, default=2.0, help="Target negatives = ratio * #positives (training).")

    parser.add_argument("--random_eval_n", type=int, default=10000, help="Random evaluation subset size from test pool.")
    parser.add_argument("--random_eval_repeats", type=int, default=30, help="Number of random draws per fold.")
    parser.add_argument("--seed", type=int, default=None, help="If provided, results become reproducible. Default: None (fully random).")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_csv)
    required = {"smiles", "antibiotic_activity"}
    missing = required - set(df.columns)
    assert not missing, f"CSV missing required columns: {missing}"

    run_random10k_repeats(
        df=df,
        out_dir=Path(args.out_dir),
        seed=args.seed,                      # None => no fixed seed
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius,
        n_splits_outer=args.splits,
        inner_cv_splits=args.inner_cv,
        neg_pos_ratio=args.neg_pos_ratio,
        random_eval_n=args.random_eval_n,
        random_eval_repeats=args.random_eval_repeats,
    )


if __name__ == "__main__":
    main()
