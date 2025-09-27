"""
Antibiotic Activity ML benchmark

This script trains and evaluates classical ML models on antibiotic-activity
classification from SMILES using Morgan fingerprints. It implements:
  1) KMeans-based undersampling on negatives (train & test)  -> strategy "kmeans"
  2) No undersampling (full negatives)                       -> strategy "none"
  3) Random undersampling (train & test)                     -> strategy "random"
  4) Random undersampling + balanced test repeats            -> strategy "random_balanced"
     (Same training as #3; additionally evaluates on repeatedly balanced
      test draws from the *un-undersampled* test pool to reduce randomness.)

For each strategy the script exports:
  - ./ml_outputs2/fold_results_<strategy>.csv      (per-fold, per-model detailed metrics)
  - ./ml_outputs2/agg_performance_<strategy>.csv   (mean±sd table across folds)
  - ./ml_outputs2/best_params_<strategy>.csv       (per-fold best hyperparameters)

Additionally, it writes:
  - ./ml_outputs2/combined_agg_performance.csv     (all strategies concatenated)

Data requirements (CSV):
  - columns: 'smiles' (str), 'antibiotic_activity' (0/1)

Recommended environment (conda):
  conda install -c conda-forge rdkit xgboost scikit-learn pandas numpy

Example:
  python ml_benchmark.py \
      --data_csv ./data/raw_data.csv \
      --out_dir ./ml_outputs2 \
      --splits 5 \
      --inner_cv 5 \
      --neg_pos_ratio 2.0 \
      --balanced_repeats 10 \
      --seed 42

Notes:
  - Uses ROC-AUC as inner-CV selection metric (GridSearchCV + StratifiedKFold).
  - Protects against degenerate scores (constant decision_function/probability).
  - KMeans selection picks closest-to-centroid negatives; if not enough samples,
    randomly fills to the target count for stability.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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
    """
    Define model zoo and hyperparameter grids for GridSearchCV.
    Uses ROC-AUC as the selection metric in inner cross validation.
    """
    models_params = {
        "Logistic Regression": (
            # Avoid passing n_jobs to prevent version-compat issues
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
    return models_params


# ----------------------- Fingerprints (MorganGenerator) -----------------------
def get_morgan_generator(radius: int, fp_size: int):
    """
    Use RDKit's MorganGenerator to avoid deprecation warnings.
    """
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


def smiles_to_fp(smiles: str, gen, n_bits: int):
    """
    Convert a SMILES string to a numpy uint8 binary fingerprint vector.
    Returns all-zero vector on invalid SMILES.
    """
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


# ----------------------- Negative sampling strategies -----------------------
def kmeans_select(neg_df: pd.DataFrame, n_target: int, fp_col="fingerprint", random_state=42):
    """
    KMeans-based selection: cluster negatives into n_clusters and take the point
    closest to each centroid. If not enough unique samples, randomly fill to n_target.
    """
    n_target = int(min(max(n_target, 0), len(neg_df)))
    if n_target == 0:
        return neg_df.iloc[[]]
    if n_target >= len(neg_df):
        return neg_df.copy()

    X = np.vstack(neg_df[fp_col].values)
    n_clusters = max(1, min(n_target, len(neg_df)))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    km.fit(X)
    closest_idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    chosen = set(closest_idx.tolist())

    # Random fill if unique representatives < target
    if len(chosen) < n_target:
        remain = [i for i in range(len(neg_df)) if i not in chosen]
        need = n_target - len(chosen)
        rng = np.random.default_rng(random_state)
        extra = rng.choice(remain, size=need, replace=False)
        chosen.update(extra.tolist())

    chosen = list(chosen)[:n_target]
    return neg_df.iloc[chosen].copy()


def random_select(neg_df: pd.DataFrame, n_target: int, random_state=42):
    """
    Uniform random selection of negatives.
    """
    n_target = int(min(max(n_target, 0), len(neg_df)))
    if n_target == 0:
        return neg_df.iloc[[]]
    if n_target >= len(neg_df):
        return neg_df.copy()
    return neg_df.sample(n=n_target, replace=False, random_state=random_state).copy()


# ----------------------- Metrics & Scoring -----------------------
def evaluate_once(y_true, scores_or_proba, y_pred_binary):
    """
    Compute a set of robust metrics with protections for degenerate cases.
    Returns a dict keyed by metric name.
    """
    metrics = {}

    # ROC/PR protections for constant scores
    if np.all(scores_or_proba == scores_or_proba[0]):
        metrics["ROC-AUC"] = 0.5
        # For PR-AUC, fall back to a trivial baseline equal to positive prevalence
        prevalence = float(np.mean(y_true))
        metrics["PRC-AUC"] = prevalence
    else:
        metrics["ROC-AUC"] = roc_auc_score(y_true, scores_or_proba)
        prec, rec, _ = precision_recall_curve(y_true, scores_or_proba)
        metrics["PRC-AUC"] = auc(rec, prec)

    metrics["Accuracy"] = accuracy_score(y_true, y_pred_binary)
    metrics["F1 Score"] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred_binary)
    metrics["Recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics["Precision"] = precision_score(y_true, y_pred_binary, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    metrics["False Positives"] = int(fp)
    metrics["False Positive Rate"] = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return metrics


def get_scores(best_model, X_test):
    """
    Return (probability-like scores in [0,1], hard predictions with threshold=0.5).
    Uses predict_proba if available; otherwise normalizes decision_function.
    """
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


# ----------------------- Strategy Runner -----------------------
def run_strategy(
    data: pd.DataFrame,
    strategy_name: str,
    neg_sampling_train: str,   # "kmeans" | "random" | "none"
    neg_sampling_test: str,    # "kmeans" | "random" | "none"
    out_dir: Path,
    seed: int,
    n_splits_outer: int,
    inner_cv_splits: int,
    neg_pos_ratio: float,
    balanced_repeats: int = 10,
    do_balanced_eval: bool = False,
):
    """
    Run one strategy (outer KFold on positives), export per-fold results, aggregated tables,
    and per-fold best hyperparameters. Optionally perform repeated balanced test evaluation.
    """
    assert neg_sampling_train in {"kmeans", "random", "none"}
    assert neg_sampling_test in {"kmeans", "random", "none"}

    models_params = make_models_params(seed)

    # Positive/negative split
    pos_df = data[data["antibiotic_activity"] == 1].reset_index(drop=True)
    neg_df = data[data["antibiotic_activity"] == 0].reset_index(drop=True)
    logging.info(f"Strategy={strategy_name}  pos={len(pos_df)}  neg={len(neg_df)}")

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)

    fold_rows = []
    param_rows = []

    for fold_id, (pos_tr_idx, pos_te_idx) in enumerate(kf_outer.split(pos_df), start=1):
        logging.info(f"[{strategy_name}] Fold {fold_id}/{n_splits_outer}")

        pos_train = pos_df.iloc[pos_tr_idx].reset_index(drop=True)
        pos_test = pos_df.iloc[pos_te_idx].reset_index(drop=True)

        neg_train, neg_test = train_test_split(
            neg_df, test_size=0.2, random_state=seed + fold_id, shuffle=True
        )
        neg_train = neg_train.reset_index(drop=True)
        neg_test = neg_test.reset_index(drop=True)

        # Target negative counts relative to positive counts
        n_neg_tr_target = int(round(neg_pos_ratio * len(pos_train)))
        n_neg_te_target = int(round(neg_pos_ratio * len(pos_test)))

        # Select negatives for training
        if neg_sampling_train == "kmeans":
            neg_train_sel = kmeans_select(
                neg_train, n_neg_tr_target, fp_col="fingerprint", random_state=seed + fold_id
            )
        elif neg_sampling_train == "random":
            neg_train_sel = random_select(
                neg_train, n_neg_tr_target, random_state=seed + fold_id
            )
        else:  # "none"
            neg_train_sel = neg_train

        # Select negatives for testing (strategy-defined test set)
        if neg_sampling_test == "kmeans":
            neg_test_sel = kmeans_select(
                neg_test, n_neg_te_target, fp_col="fingerprint", random_state=seed + fold_id
            )
        elif neg_sampling_test == "random":
            neg_test_sel = random_select(
                neg_test, n_neg_te_target, random_state=seed + fold_id
            )
        else:  # "none"
            neg_test_sel = neg_test

        train_df = pd.concat([pos_train, neg_train_sel], ignore_index=True)
        test_df = pd.concat([pos_test, neg_test_sel], ignore_index=True)

        X_train = np.vstack(train_df["fingerprint"].values)
        y_train = train_df["antibiotic_activity"].astype(int).values
        X_test = np.vstack(test_df["fingerprint"].values)
        y_test = test_df["antibiotic_activity"].astype(int).values

        # A "test pool" without any test undersampling, for balanced evaluation
        test_pool_df = pd.concat([pos_test, neg_test], ignore_index=True)
        X_test_pool = np.vstack(test_pool_df["fingerprint"].values)
        y_test_pool = test_pool_df["antibiotic_activity"].astype(int).values

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_test_pool_scaled = scaler.transform(X_test_pool)

        # Inner CV (Stratified)
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
            clf.fit(X_train_scaled, y_train)
            best_model = clf.best_estimator_

            # Strategy-defined test set (regular evaluation)
            proba, y_pred = get_scores(best_model, X_test_scaled)
            met = evaluate_once(y_test, proba, y_pred)
            row = {"Strategy": strategy_name, "Fold": fold_id, "Model": model_name}
            row.update(met)
            fold_rows.append(row)

            # Save best params
            param_rows.append(
                {
                    "Strategy": strategy_name,
                    "Fold": fold_id,
                    "Model": model_name,
                    "best_params": json.dumps(clf.best_params_),
                }
            )

            # Optional: Balanced repeated test evaluation (only for "random_balanced")
            if do_balanced_eval:
                pos_idx_pool = np.where(y_test_pool == 1)[0]
                neg_idx_pool = np.where(y_test_pool == 0)[0]
                n_bal = int(min(len(pos_idx_pool), len(neg_idx_pool)))

                if n_bal == 0:
                    # No way to balance; attach NaNs
                    for k in met.keys():
                        fold_rows[-1][f"{k} (balanced)"] = np.nan
                else:
                    rng = np.random.default_rng(seed + fold_id)
                    rep_metrics = []
                    for _ in range(balanced_repeats):
                        pos_sel = rng.choice(pos_idx_pool, size=n_bal, replace=False)
                        neg_sel = rng.choice(neg_idx_pool, size=n_bal, replace=False)
                        idx = np.concatenate([pos_sel, neg_sel])
                        Xb = X_test_pool_scaled[idx]
                        yb = y_test_pool[idx]
                        proba_b, y_pred_b = get_scores(best_model, Xb)
                        rep_metrics.append(evaluate_once(yb, proba_b, y_pred_b))
                    # Mean across repeats
                    for k in rep_metrics[0].keys():
                        fold_rows[-1][f"{k} (balanced)"] = float(np.mean([m[k] for m in rep_metrics]))

    # Export per-fold details
    fold_df = pd.DataFrame(fold_rows)
    params_df = pd.DataFrame(param_rows)

    metric_cols = [c for c in fold_df.columns if c not in {"Strategy", "Fold", "Model"}]

    # Aggregation (mean±sd across folds)
    agg_spec = {mc: ["mean", "std"] for mc in metric_cols}
    g = fold_df.groupby(["Strategy", "Model"]).agg(agg_spec)
    g.columns = [" ".join(col).strip() for col in g.columns.values]

    def mean_sd_str(m, s):
        if pd.isna(m) or pd.isna(s):
            return "nan"
        return f"{m:.4f}±{s:.4f}"

    perf_table = pd.DataFrame(index=g.index)
    for mc in metric_cols:
        mcol, scol = f"{mc} mean", f"{mc} std"
        if mcol in g.columns and scol in g.columns:
            perf_table[mc] = [mean_sd_str(m, s) for m, s in zip(g[mcol], g[scol])]

    # Write strategy files
    fold_csv = out_dir / f"fold_results_{strategy_name}.csv"
    agg_csv = out_dir / f"agg_performance_{strategy_name}.csv"
    params_csv = out_dir / f"best_params_{strategy_name}.csv"
    fold_df.to_csv(fold_csv, index=False)
    perf_table.to_csv(agg_csv)
    params_df.to_csv(params_csv, index=False)

    logging.info(f"[{strategy_name}] saved:")
    logging.info(f" - {fold_csv}")
    logging.info(f" - {agg_csv}")
    logging.info(f" - {params_csv}")

    return fold_df, perf_table, params_df


# ----------------------- Main -----------------------
def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Antibiotic Activity ML benchmark")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to CSV with 'smiles' and 'antibiotic_activity'.")
    parser.add_argument("--out_dir", type=str, default="./ml_outputs2", help="Output directory.")
    parser.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint size.")
    parser.add_argument("--fp_radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--splits", type=int, default=5, help="Outer KFold splits on positives.")
    parser.add_argument("--inner_cv", type=int, default=5, help="Inner CV splits for GridSearchCV.")
    parser.add_argument("--neg_pos_ratio", type=float, default=2.0, help="Target negatives = ratio * #positives.")
    parser.add_argument("--balanced_repeats", type=int, default=10, help="Balanced test repeats for random_balanced.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read data
    df = pd.read_csv(args.data_csv)
    required = {"smiles", "antibiotic_activity"}
    missing = required - set(df.columns)
    assert not missing, f"CSV missing required columns: {missing}"

    # Compute fingerprints once (if not already provided)
    if "fingerprint" not in df.columns:
        logging.info("Computing Morgan fingerprints ...")
        gen = get_morgan_generator(radius=args.fp_radius, fp_size=args.fp_bits)
        df = df.copy()
        df["fingerprint"] = df["smiles"].apply(lambda s: smiles_to_fp(s, gen, args.fp_bits))
        logging.info("Fingerprints computed.")

    # 1) KMeans undersampling (train & test)
    run_strategy(
        data=df,
        strategy_name="kmeans",
        neg_sampling_train="kmeans",
        neg_sampling_test="kmeans",
        out_dir=out_dir,
        seed=args.seed,
        n_splits_outer=args.splits,
        inner_cv_splits=args.inner_cv,
        neg_pos_ratio=args.neg_pos_ratio,
        balanced_repeats=args.balanced_repeats,
        do_balanced_eval=False,
    )

    # 2) No undersampling (full negatives)
    run_strategy(
        data=df,
        strategy_name="none",
        neg_sampling_train="none",
        neg_sampling_test="none",
        out_dir=out_dir,
        seed=args.seed,
        n_splits_outer=args.splits,
        inner_cv_splits=args.inner_cv,
        neg_pos_ratio=args.neg_pos_ratio,
        balanced_repeats=args.balanced_repeats,
        do_balanced_eval=False,
    )

    # 3) Random undersampling (regular evaluation)
    run_strategy(
        data=df,
        strategy_name="random",
        neg_sampling_train="random",
        neg_sampling_test="random",
        out_dir=out_dir,
        seed=args.seed,
        n_splits_outer=args.splits,
        inner_cv_splits=args.inner_cv,
        neg_pos_ratio=args.neg_pos_ratio,
        balanced_repeats=args.balanced_repeats,
        do_balanced_eval=False,
    )

    # 4) Random undersampling + balanced test repeats (extra evaluation)
    _, perf_balanced, _ = run_strategy(
        data=df,
        strategy_name="random_balanced",
        neg_sampling_train="random",
        neg_sampling_test="random",
        out_dir=out_dir,
        seed=args.seed,
        n_splits_outer=args.splits,
        inner_cv_splits=args.inner_cv,
        neg_pos_ratio=args.neg_pos_ratio,
        balanced_repeats=args.balanced_repeats,
        do_balanced_eval=True,
    )

    # Combine all aggregated tables for convenience
    combined = []
    for name in ["kmeans", "none", "random", "random_balanced"]:
        p = out_dir / f"agg_performance_{name}.csv"
        if p.exists():
            t = pd.read_csv(p, index_col=0)
            t.reset_index(inplace=True)
            t.rename(columns={"index": "Strategy_Model"}, inplace=True)
            t.insert(0, "Strategy", name)
            combined.append(t)

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        combined_csv = out_dir / "combined_agg_performance.csv"
        combined_df.to_csv(combined_csv, index=False)
        logging.info(f"[combined] saved: {combined_csv}")


if __name__ == "__main__":
    main()
