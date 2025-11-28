# -*- coding: utf-8 -*-
"""
Learning Curve for ECFP bits (1024/2048/4096) under your evaluation protocol:
- Input CSV: ./data/raw_data.csv with columns: smiles, antibiotic_activity (0/1)
- Fingerprints: Morgan (ECFP4, radius=2), bits in [1024, 2048, 4096]
- Outer split: positives only; negatives sampled once into train/test by random split
- Negative selection: KMeans prototypes (train/test) with n_clusters = 2 * #positives  (bounded)
- Inner CV: GridSearchCV (5-fold) with scoring='roc_auc', refit=True
- Learning curve: grow training positives by fractions (e.g., 5%→100%),
  each fraction repeated R times, evaluate on a fixed external test set
- Outputs: per-run CSV, aggregated CSV (mean±std), and plots (AUC vs training size)
"""

import os, json, time, math, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef, recall_score, precision_score,
    confusion_matrix, pairwise_distances, pairwise_distances_argmin_min
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------
# Config
# ---------------------
SEED = 42
OUTDIR = "./ml_lc_outputs"
DATA_CSV = "./data/raw_data.csv"  # must have 'smiles' + 'antibiotic_activity'
BITS_LIST = [1024, 2048, 4096]
TRAIN_FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]   # learning curve
REPEATS_PER_FRACTION = 3                                 # repeat sampling to get mean±std
INNER_CV_SPLITS = 5
PLOT = True

# Models to evaluate (可自行增减；树类不敏感缩放，但统一缩放也无妨)
MODELS = {
    "XGBoost": (
        XGBClassifier(
            use_label_encoder=False, eval_metric='logloss',
            tree_method='hist', random_state=SEED, n_jobs=-1
        ),
        {'n_estimators': [300, 600], 'max_depth': [3, 6], 'learning_rate': [0.03, 0.1]}
    ),
    "LogReg": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {'C': [0.01, 0.1, 1, 10, 100]}
    ),
    "SVM": (
        SVC(probability=True, random_state=SEED),
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {'n_estimators': [200, 500], 'max_depth': [None, 10, 20]}
    ),
}

os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(SEED)

# ---------------------
# Helpers
# ---------------------
def ecfp_bits(smiles: str, n_bits: int, radius: int = 2):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        arr = np.zeros((n_bits,), dtype=np.int8)
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def gen_fps_matrix(smiles_list, n_bits: int) -> np.ndarray:
    t0 = time.time()
    fps = np.vstack([ecfp_bits(s, n_bits=n_bits, radius=2) for s in smiles_list]).astype(np.float32)
    secs = time.time() - t0
    density = float(fps.mean())
    return fps, secs, density

def kmeans_prototypes(X: np.ndarray, n_clusters: int, random_state: int = SEED) -> np.ndarray:
    """Return indices in X of points nearest to KMeans centers (memory-safe)."""
    n_clusters = max(1, min(n_clusters, X.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.astype(np.float32)
    # memory-friendly nearest indices
    try:
        idx, _ = pairwise_distances_argmin_min(centers, X, metric='euclidean')
    except MemoryError:
        idx = np.empty(n_clusters, dtype=np.int64)
        best = np.full(n_clusters, np.inf, dtype=np.float64)
        chunk = 4096
        for s in range(0, X.shape[0], chunk):
            block = X[s:s+chunk]
            D = pairwise_distances(centers, block, metric='euclidean', n_jobs=1)
            j = D.argmin(axis=1)
            v = D[np.arange(n_clusters), j]
            mask = v < best
            best[mask] = v[mask]
            idx[mask] = s + j[mask]
    return idx

def evaluate_probs(y_true, y_proba):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    return {
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'PRC-AUC': auc(rec, prec),
        'Average Precision': average_precision_score(y_true, y_proba)
    }

def evaluate_labels(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'False Positive Rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0
    }

def fit_predict_with_inner_cv(model, grid, X_train, y_train, X_test):
    clf = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring='roc_auc',
        refit=True,
        cv=INNER_CV_SPLITS,
        n_jobs=-1,
        return_train_score=False
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_secs = time.time() - t0

    best = clf.best_estimator_
    best_params = clf.best_params_
    best_cv_auc = clf.best_score_

    t1 = time.time()
    if hasattr(best, "predict_proba"):
        proba = best.predict_proba(X_test)[:, 1]
    else:
        s = best.decision_function(X_test)
        proba = (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else np.zeros_like(s)
    y_pred = (proba >= 0.5).astype(int)
    pred_secs = time.time() - t1
    return proba, y_pred, fit_secs, pred_secs, best_params, best_cv_auc

# ---------------------
# Main
# ---------------------
def main():
    # 1) Load data
    df = pd.read_csv(DATA_CSV)
    assert 'smiles' in df.columns and 'antibiotic_activity' in df.columns, \
        "CSV must contain 'smiles' and 'antibiotic_activity' columns."
    smiles = df['smiles'].astype(str).tolist()
    labels = df['antibiotic_activity'].astype(int).values

    # 2) Split positives/negatives once; build a fixed external test set
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_df = df[pos_mask].reset_index(drop=True)
    neg_df = df[neg_mask].reset_index(drop=True)

    # 外部测试：正样本 20%；负样本 20% -> 再KMeans选代表点
    pos_train, pos_test = train_test_split(pos_df, test_size=0.2, random_state=SEED, stratify=None)
    neg_train, neg_test = train_test_split(neg_df, test_size=0.2, random_state=SEED, stratify=None)

    # 3) For each bits: build fingerprints for all samples (一次生成，后续索引切片)
    all_results = []
    agg_rows = []

    for bits in BITS_LIST:
        print(f"\n=== ECFP bits = {bits} ===")
        fps, fp_secs, fp_density = gen_fps_matrix(smiles, n_bits=bits)

        # Index helpers
        df_reset = df.reset_index(drop=True)
        pos_idx_all = df_reset.index[df_reset['antibiotic_activity'] == 1].values
        neg_idx_all = df_reset.index[df_reset['antibiotic_activity'] == 0].values

        pos_train_idx = pos_train.index.values
        pos_test_idx  = pos_test.index.values
        neg_train_idx = neg_train.index.values
        neg_test_idx  = neg_test.index.values

        X_pos_train_all = fps[pos_train_idx]
        X_pos_test_all  = fps[pos_test_idx]
        X_neg_train_all = fps[neg_train_idx]
        X_neg_test_all  = fps[neg_test_idx]

        # 固定外部测试负样本代表点（KMeans）
        n_clusters_test = max(1, min(2 * len(pos_test_idx), len(neg_test_idx)))
        idx_neg_te = kmeans_prototypes(X_neg_test_all, n_clusters_test, random_state=SEED)
        X_test = np.vstack([X_pos_test_all, X_neg_test_all[idx_neg_te]])
        y_test = np.concatenate([np.ones(len(X_pos_test_all), dtype=np.int32),
                                 np.zeros(len(idx_neg_te), dtype=np.int32)])

        # 4) Learning Curve over fractions
        for frac in TRAIN_FRACTIONS:
            n_pos_use = max(1, int(round(len(pos_train_idx) * frac)))
            for rep in range(REPEATS_PER_FRACTION):
                rep_seed = SEED + 1000*rep + int(bits) + int(100*frac)
                rng = np.random.default_rng(rep_seed)
                # 采样正样本
                sel_pos_idx = rng.choice(pos_train_idx, size=n_pos_use, replace=False)
                X_pos_tr = fps[sel_pos_idx]

                # 训练集负样本代表点（KMeans）
                n_clusters_tr = max(1, min(2 * len(sel_pos_idx), len(neg_train_idx)))
                idx_neg_tr = kmeans_prototypes(X_neg_train_all, n_clusters_tr, random_state=rep_seed)
                X_neg_tr = X_neg_train_all[idx_neg_tr]

                # 构造训练集
                X_train = np.vstack([X_pos_tr, X_neg_tr])
                y_train = np.concatenate([np.ones(len(X_pos_tr), dtype=np.int32),
                                          np.zeros(len(X_neg_tr), dtype=np.int32)])

                # 标准化（树模型无所谓；统一流程便于并行）
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc  = scaler.transform(X_test)

                # 评估每个模型
                for model_name, (est, grid) in MODELS.items():
                    proba, y_pred, fit_secs, pred_secs, best_params, best_cv_auc = \
                        fit_predict_with_inner_cv(est, grid, X_train_sc, y_train, X_test_sc)

                    rank_metrics = evaluate_probs(y_test, proba)
                    cls_metrics  = evaluate_labels(y_test, y_pred)

                    row = {
                        'fp_bits': bits,
                        'fp_density': fp_density,
                        'fp_gen_secs': fp_secs,
                        'train_frac': frac,
                        'repeat': rep,
                        'n_pos_train': int(len(X_pos_tr)),
                        'n_neg_train': int(len(X_neg_tr)),
                        'n_pos_test':  int(len(X_pos_test_all)),
                        'n_neg_test':  int(len(idx_neg_te)),
                        'model': model_name,
                        'inner_cv_splits': INNER_CV_SPLITS,
                        'inner_cv_metric': 'roc_auc',
                        'inner_cv_mean': best_cv_auc,
                        'best_params': json.dumps(best_params),
                        'fit_secs': fit_secs,
                        'pred_secs': pred_secs,
                    }
                    row.update(rank_metrics)
                    row.update(cls_metrics)
                    all_results.append(row)

        # 每个 bits 循环结束后可以即时落盘一次
        pd.DataFrame(all_results).to_csv(os.path.join(OUTDIR, "lc_per_run_results.csv"), index=False)

    # 5) 聚合：按 (model, fp_bits, train_frac) 求 mean±std
    per_run = pd.DataFrame(all_results)
    agg = (per_run
           .groupby(['model','fp_bits','train_frac'])
           .agg({
                'ROC-AUC':['mean','std'],
                'PRC-AUC':['mean','std'],
                'Average Precision':['mean','std'],
                'Accuracy':['mean','std'],
                'F1 Score':['mean','std'],
                'MCC':['mean','std'],
                'Recall':['mean','std'],
                'Precision':['mean','std'],
                'False Positive Rate':['mean','std'],
                'fit_secs':['mean','std']
            }).reset_index())

    # 拍扁列名
    agg.columns = ['model','fp_bits','train_frac'] + [f"{c[0]} ({c[1]})" for c in agg.columns[3:]]
    agg_path = os.path.join(OUTDIR, "lc_aggregated_mean_std.csv")
    agg.to_csv(agg_path, index=False)
    print(f"\nSaved: {agg_path}")
    print(f"Per-run CSV: {os.path.join(OUTDIR, 'lc_per_run_results.csv')}")

    # 6) 画学习曲线（每个模型一张图；y=ROC-AUC(mean)）
    if PLOT:
        for model in agg['model'].unique():
            sub = agg[agg['model']==model].copy()
            plt.figure(figsize=(6.5,5.0), dpi=150)
            for bits in sorted(sub['fp_bits'].unique()):
                sb = sub[sub['fp_bits']==bits].sort_values('train_frac')
                x = sb['train_frac'].values
                y = sb['ROC-AUC (mean)'].values
                e = sb['ROC-AUC (std)'].values / math.sqrt(max(REPEATS_PER_FRACTION,1))  # 标准误差
                plt.plot(x, y, marker='o', label=f"bits={bits}")
                plt.fill_between(x, y-1.96*e, y+1.96*e, alpha=0.15)
            plt.xlabel("Training fraction of positive set (negatives via KMeans prototypes)")
            plt.ylabel("ROC-AUC (mean across repeats)")
            plt.ylim(0, 1)
            plt.title(f"Learning Curve — {model}")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            fpath = os.path.join(OUTDIR, f"lc_rocauc_{model}.png")
            plt.tight_layout()
            plt.savefig(fpath, dpi=220)
            plt.close()
            print(f"Plot saved: {fpath}")

if __name__ == "__main__":
    main()
