# -*- coding: utf-8 -*-
"""
Cross-strategy benchmark (single file)

Route A: Classic ML after graph-similarity undersampling
    1) Build molecular graphs from SMILES (RDKit) and train a Graph Autoencoder (GAE; GINE encoder)
    2) Get graph embeddings for all compounds
    3) Per outer fold, undersample negatives by similarity to the positive centroid (train/test separated)
       - 'nearest' or 'farthest' w.r.t. the centroid; cosine (default) or euclidean
       - target #neg = ratio * #pos (for both train and test)
    4) Train classical ML models on Morgan fingerprints with inner GridSearchCV (ROC-AUC), evaluate on strategy-defined test set

Route B: GCN after KMeans undersampling
    1) Compute Morgan fingerprints
    2) Per outer fold, KMeans-based undersampling of negatives in fingerprint space (train/test separated)
    3) Train a GINE classifier (5-fold Stratified CV inside the balanced set) and report metrics

Outputs (under --out_dir):
  - fold_results_ml_graphsim.csv
  - agg_performance_ml_graphsim.csv
  - best_params_ml_graphsim.csv
  - fold_results_gcn_kmeans.csv
  - agg_performance_gcn_kmeans.csv
  - best_gine_kmeans.pth (best ROC-AUC fold for Route B)

Data requirements:
  CSV with 'smiles' (str) and 'antibiotic_activity' (0/1)

Example:
  python cross_benchmark.py \
      --csv ./data/raw_data.csv \
      --out_dir ./ml_outputs_swap \
      --splits 5 --inner_cv 5 \
      --neg_pos_ratio 1.0 \
      --pick nearest --metric cosine \
      --seed 42
"""

import argparse
import json
import logging
import os
from pathlib import Path
import warnings
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

# sklearn
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve, auc, f1_score, matthews_corrcoef, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# PyTorch / PyG for GAE + GINE
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.loader import DataLoader
except Exception:
    from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, GINEConv, BatchNorm


# =========================================================
# Logging
# =========================================================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


# =========================================================
# Global config (defaults)
# =========================================================
SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
NEG_POS_RATIO = 1.0
OUT_DIR = Path("./ml_outputs_swap")
ROUTE_A_NAME = "ml_graphsim"
ROUTE_B_NAME = "gcn_kmeans"

# GAE / GINE defaults
BATCH_SIZE_AE = 64
BATCH_SIZE_CLS = 64
EPOCHS_AE = 60
EPOCHS_GCN = 120
PATIENCE = 15
LR_AE = 1e-3
LR_GCN = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN = 128
NUM_GINE = 3
VAL_SPLIT = 0.10

COMMON_Z = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 19, 11, 12, 20, 26, 29, 30, 35, 53]  # common atomic numbers


# =========================================================
# Fingerprints (MorganGenerator) + graphs
# =========================================================
def get_morgan_generator(radius: int, fp_size: int):
    """RDKit Morgan generator wrapper (avoids deprecation warnings)."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

def smiles_to_fp(smiles: str, gen, n_bits: int):
    if not isinstance(smiles, str):
        arr = np.zeros(n_bits, dtype=np.uint8); return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        arr = np.zeros(n_bits, dtype=np.uint8); return arr
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def one_hot(val, choices):
    vec = [0] * len(choices)
    if val in choices:
        vec[choices.index(val)] = 1
    return vec

def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    z = atom.GetAtomicNum()
    z_onehot = one_hot(z if z in COMMON_Z else -1, COMMON_Z + [-1])
    degree = atom.GetTotalDegree()
    degree_onehot = one_hot(min(degree, 5), list(range(6)))
    hyb = atom.GetHybridization()
    hyb_choices = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hyb_onehot = one_hot(hyb if hyb in hyb_choices else None, hyb_choices + [None])
    charge = int(atom.GetFormalCharge()); charge = max(-2, min(2, charge))
    charge_onehot = one_hot(charge, [-2, -1, 0, 1, 2])
    num_h = min(atom.GetTotalNumHs(), 4)
    num_h_onehot = one_hot(num_h, [0, 1, 2, 3, 4])
    aromatic = [int(atom.GetIsAromatic())]
    ring = [int(atom.IsInRing())]
    chiral_tag = atom.GetChiralTag()
    chiral_choices = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    chiral_onehot = one_hot(
        chiral_tag if chiral_tag in chiral_choices else Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        chiral_choices
    )
    return z_onehot + degree_onehot + hyb_onehot + charge_onehot + num_h_onehot + aromatic + ring + chiral_onehot

def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    bt = bond.GetBondType()
    bt_choices = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    bt_onehot = one_hot(bt if bt in bt_choices else None, bt_choices + [None])
    conj = [int(bond.GetIsConjugated())]
    ring = [int(bond.IsInRing())]
    stereo = bond.GetStereo()
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    stereo_onehot = one_hot(stereo if stereo in stereo_choices else Chem.rdchem.BondStereo.STEREONONE, stereo_choices)
    return bt_onehot + conj + ring + stereo_onehot

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=False)
    AllChem.Compute2DCoords(mol)
    x = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        edge_index.append([i, j]); edge_attr.append(f)
        edge_index.append([j, i]); edge_attr.append(f)
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        dummy = Chem.MolFromSmiles("CC").GetBonds()[0]
        feat_w = len(bond_features(dummy))
        edge_attr = torch.empty((0, feat_w), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    g.smiles = smiles
    return g


# =========================================================
# GINE/GAE models
# =========================================================
def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

class GINEEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = HIDDEN, num_layers: int = NUM_GINE, dropout: float = DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.convs.append(GINEConv(mlp(in_dim, hidden), edge_dim=edge_dim))
        self.bns.append(BatchNorm(hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(mlp(hidden, hidden), edge_dim=edge_dim))
            self.bns.append(BatchNorm(hidden))

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h_res.shape == h.shape:
                h = h + 0.1 * h_res
        return h

class GraphAE(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = HIDDEN, num_layers: int = NUM_GINE, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = GINEEncoder(in_dim, edge_dim, hidden, num_layers, dropout)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr)
        x_hat = self.decoder(h)
        g = global_mean_pool(h, batch)
        return x_hat, g

    def encode_nodes(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)

class GINEClassifier(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = HIDDEN, num_layers: int = NUM_GINE, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = GINEEncoder(in_dim, edge_dim, hidden, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr)
        g = global_mean_pool(h, batch)
        logit = self.head(g).view(-1)
        return logit

    def load_from_ae(self, ae: GraphAE):
        self.encoder.load_state_dict(ae.encoder.state_dict(), strict=False)


# =========================================================
# Training utils (GAE / GCN)
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_ae(ae: GraphAE, loader: DataLoader, device, epochs: int = EPOCHS_AE, lr: float = LR_AE,
             wd: float = WEIGHT_DECAY, patience: int = PATIENCE) -> GraphAE:
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=wd)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=False)
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best_loss, bad = float("inf"), 0
    for ep in range(1, epochs + 1):
        ae.train()
        total = 0.0
        for data in loader:
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, _ = ae(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.mse_loss(x_hat, data.x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 2.0)
            opt.step()
            total += loss.item()
        mean_loss = total / max(len(loader), 1)
        scheduler.step(mean_loss)
        logging.info(f"[AE] Epoch {ep:03d} | recon MSE {mean_loss:.5f} | lr {opt.param_groups[0]['lr']:.2e}")
        if mean_loss < best_loss - 1e-5:
            best_loss, bad = mean_loss, 0
            torch.save(ae.state_dict(), "best_graph_ae_tmp.pth")
        else:
            bad += 1
            if bad >= patience and ep >= 20:
                logging.info("[AE] Early stop.")
                break
    ae.load_state_dict(torch.load("best_graph_ae_tmp.pth", map_location=device))
    ae.eval()
    return ae

@torch.no_grad()
def get_graph_embeddings(ae: GraphAE, loader: DataLoader, device) -> np.ndarray:
    ae.eval()
    embs = []
    for data in loader:
        data = data.to(device)
        h = ae.encode_nodes(data.x, data.edge_index, data.edge_attr)
        g = global_mean_pool(h, data.batch)
        embs.append(g.cpu().numpy())
    return np.concatenate(embs, axis=0)

def train_one_epoch_gcn(model: nn.Module, loader: DataLoader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        y = data.y.view(-1).to(device)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)

@torch.no_grad()
def infer_proba_gcn(model: nn.Module, loader: DataLoader, device):
    model.eval()
    ys, ps = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        prob = torch.sigmoid(logits)
        ys.append(data.y.view(-1).cpu().numpy())
        ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    return y_true, y_prob


# =========================================================
# Negative selection: graph similarity + KMeans
# =========================================================
def select_negatives_by_centroid(
    neg_embs: np.ndarray,
    pos_embs: np.ndarray,
    k: int,
    metric: str = "cosine",
    pick: str = "nearest"
) -> np.ndarray:
    """
    Select 'k' negatives by similarity to the centroid of positive embeddings.
    Returns indices (relative to neg_embs).
    """
    if k <= 0 or len(neg_embs) == 0:
        return np.array([], dtype=int)
    k = int(min(k, len(neg_embs)))
    pos_centroid = pos_embs.mean(axis=0, keepdims=True)
    if metric == "cosine":
        a = neg_embs / (np.linalg.norm(neg_embs, axis=1, keepdims=True) + 1e-9)
        b = pos_centroid / (np.linalg.norm(pos_centroid, axis=1, keepdims=True) + 1e-9)
        sim = (a @ b.T).reshape(-1)  # larger => more similar
        order = np.argsort(-sim) if pick == "nearest" else np.argsort(sim)
    elif metric == "euclidean":
        dist = np.linalg.norm(neg_embs - pos_centroid, axis=1)  # smaller => more similar
        order = np.argsort(dist) if pick == "nearest" else np.argsort(-dist)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    return order[:k]

def kmeans_select_in_fp(neg_fp_mat: np.ndarray, n_target: int, random_state: int) -> np.ndarray:
    """
    KMeans-based selection on fingerprint space: return indices (relative to neg_fp_mat).
    Strategy: pick closest sample to each centroid; if fewer unique than n_target, fill with random.
    """
    n = len(neg_fp_mat)
    if n_target <= 0 or n == 0:
        return np.array([], dtype=int)
    n_target = int(min(n_target, n))
    n_clusters = max(1, min(n_target, n))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    km.fit(neg_fp_mat)
    closest_idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, neg_fp_mat)
    chosen = set(closest_idx.tolist())
    if len(chosen) < n_target:
        remain = [i for i in range(n) if i not in chosen]
        need = n_target - len(chosen)
        rng = np.random.default_rng(random_state)
        extra = rng.choice(remain, size=need, replace=False)
        chosen.update(extra.tolist())
    chosen = list(chosen)[:n_target]
    return np.array(chosen, dtype=int)


# =========================================================
# Metrics
# =========================================================
def evaluate_once(y_true, scores_or_proba, y_pred_binary):
    m = {}
    if scores_or_proba.max() == scores_or_proba.min():
        m["ROC-AUC"] = 0.5
        prev = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
        m["PRC-AUC"] = prev
    else:
        m["ROC-AUC"] = roc_auc_score(y_true, scores_or_proba)
        prec, rec, _ = precision_recall_curve(y_true, scores_or_proba)
        m["PRC-AUC"] = auc(rec, prec)
    m["Accuracy"] = accuracy_score(y_true, y_pred_binary)
    m["F1 Score"] = f1_score(y_true, y_pred_binary, zero_division=0)
    m["MCC"] = matthews_corrcoef(y_true, y_pred_binary)
    m["Recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
    m["Precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    m["False Positives"] = int(fp)
    m["False Positive Rate"] = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return m

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


# =========================================================
# Model zoo (Route A)
# =========================================================
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
                eval_metric="logloss", random_state=seed, tree_method="hist",
                n_jobs=-1, use_label_encoder=False
            ),
            {"n_estimators": [200, 500], "max_depth": [3, 6], "learning_rate": [0.03, 0.1]},
        ),
    }


# =========================================================
# Route A: Classic ML after graph-similarity undersampling
# =========================================================
def run_route_A_ml_graphsim(
    df: pd.DataFrame,
    out_dir: Path,
    seed: int,
    n_splits_outer: int,
    inner_cv_splits: int,
    neg_pos_ratio: float,
    pick: str = "nearest",
    metric: str = "cosine",
    fp_bits: int = FP_BITS,
    fp_radius: int = FP_RADIUS
):
    logging.info("=== Route A: Classic-ML after Graph-Similarity Undersampling ===")

    # Build graphs and Morgan fingerprints
    smiles = df["smiles"].astype(str).tolist()
    labels = df["antibiotic_activity"].astype(int).to_numpy()

    graphs, drop_idx = [], []
    for i, smi in enumerate(smiles):
        g = smiles_to_graph(smi)
        if g is None or g.x.numel() == 0:
            drop_idx.append(i); continue
        g.y = torch.tensor([labels[i]], dtype=torch.float32)
        graphs.append(g)
    if drop_idx:
        logging.warning(f"Dropped {len(drop_idx)} invalid SMILES during graph construction.")
        labels = np.delete(labels, drop_idx, axis=0)
        df = df.drop(index=drop_idx).reset_index(drop=True)

    # Morgan fingerprints
    if "fingerprint" not in df.columns:
        gen = get_morgan_generator(radius=fp_radius, fp_size=fp_bits)
        df = df.copy()
        df["fingerprint"] = df["smiles"].apply(lambda s: smiles_to_fp(s, gen, fp_bits))

    # Train GAE once (unsupervised) and get embeddings for all graphs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    loader_ae = DataLoader(graphs, batch_size=BATCH_SIZE_AE, shuffle=True, num_workers=0)
    g0 = graphs[0]
    in_dim = g0.x.size(1)
    edge_dim = g0.edge_attr.size(1) if g0.edge_attr is not None else 0
    ae = GraphAE(in_dim, edge_dim, hidden=HIDDEN, num_layers=NUM_GINE, dropout=DROPOUT)
    ae = train_ae(ae, loader_ae, device=device)
    loader_eval = DataLoader(graphs, batch_size=BATCH_SIZE_AE, shuffle=False, num_workers=0)
    embs = get_graph_embeddings(ae, loader_eval, device)   # [N, hidden]

    # Split pos/neg indices on the filtered df
    pos_idx_all = np.where(labels == 1)[0]
    neg_idx_all = np.where(labels == 0)[0]
    pos_df = df.iloc[pos_idx_all].reset_index(drop=True)
    neg_df = df.iloc[neg_idx_all].reset_index(drop=True)
    pos_embs_all = embs[pos_idx_all]
    neg_embs_all = embs[neg_idx_all]

    models_params = make_models_params(seed)
    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)

    fold_rows = []
    param_rows = []

    for fold_id, (pos_tr_idx, pos_te_idx) in enumerate(kf_outer.split(pos_df), start=1):
        logging.info(f"[{ROUTE_A_NAME}] Fold {fold_id}/{n_splits_outer}")

        # Fold-wise positives
        pos_tr = pos_df.iloc[pos_tr_idx].reset_index(drop=True)
        pos_te = pos_df.iloc[pos_te_idx].reset_index(drop=True)
        pos_tr_emb = pos_embs_all[pos_tr_idx]
        pos_te_emb = pos_embs_all[pos_te_idx]

        # Independent split of negatives (like your earlier design)
        neg_tr_raw, neg_te_raw = train_test_split(
            neg_df, test_size=0.2, random_state=seed + fold_id, shuffle=True
        )
        # Align embeddings for those negatives
        # Map back to original neg indices:
        neg_tr_mask = neg_tr_raw.index.values
        neg_te_mask = neg_te_raw.index.values
        neg_tr_emb = neg_embs_all[neg_tr_mask]
        neg_te_emb = neg_embs_all[neg_te_mask]

        # Targets
        n_neg_tr = int(round(neg_pos_ratio * len(pos_tr)))
        n_neg_te = int(round(neg_pos_ratio * len(pos_te)))

        # Select by centroid similarity
        idx_sel_tr_local = select_negatives_by_centroid(
            neg_tr_emb, pos_tr_emb, k=n_neg_tr, metric=metric, pick=pick
        )
        idx_sel_te_local = select_negatives_by_centroid(
            neg_te_emb, pos_te_emb, k=n_neg_te, metric=metric, pick=pick
        )

        neg_tr_sel = neg_tr_raw.iloc[idx_sel_tr_local].reset_index(drop=True)
        neg_te_sel = neg_te_raw.iloc[idx_sel_te_local].reset_index(drop=True)

        train_df = pd.concat([pos_tr, neg_tr_sel], ignore_index=True)
        test_df  = pd.concat([pos_te, neg_te_sel], ignore_index=True)

        # Prepare ML features (Morgan FP) and labels
        X_train = np.vstack(train_df["fingerprint"].values)
        y_train = train_df["antibiotic_activity"].astype(int).values
        X_test  = np.vstack(test_df["fingerprint"].values)
        y_test  = test_df["antibiotic_activity"].astype(int).values

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=seed)

        for model_name, (model, grid) in models_params.items():
            clf = GridSearchCV(
                estimator=model, param_grid=grid, scoring="roc_auc",
                cv=inner_cv, n_jobs=-1, refit=True
            )
            clf.fit(X_train_sc, y_train)
            best_model = clf.best_estimator_

            proba, y_pred = get_scores(best_model, X_test_sc)
            met = evaluate_once(y_test, proba, y_pred)
            row = {"Route": ROUTE_A_NAME, "Fold": fold_id, "Model": model_name}
            row.update(met)
            fold_rows.append(row)

            param_rows.append({
                "Route": ROUTE_A_NAME,
                "Fold": fold_id,
                "Model": model_name,
                "best_params": json.dumps(clf.best_params_),
            })

    # Save results
    fold_df = pd.DataFrame(fold_rows)
    params_df = pd.DataFrame(param_rows)

    metric_cols = [c for c in fold_df.columns if c not in {"Route", "Fold", "Model"}]
    agg_spec = {mc: ["mean", "std"] for mc in metric_cols}
    g = fold_df.groupby(["Route", "Model"]).agg(agg_spec)
    g.columns = [" ".join(col).strip() for col in g.columns.values]

    def mean_sd_str(m, s):
        if pd.isna(m) or pd.isna(s): return "nan"
        return f"{m:.4f}±{s:.4f}"

    perf_table = pd.DataFrame(index=g.index)
    for mc in metric_cols:
        mcol, scol = f"{mc} mean", f"{mc} std"
        if mcol in g.columns and scol in g.columns:
            perf_table[mc] = [mean_sd_str(m, s) for m, s in zip(g[mcol], g[scol])]

    fold_csv   = out_dir / f"fold_results_{ROUTE_A_NAME}.csv"
    agg_csv    = out_dir / f"agg_performance_{ROUTE_A_NAME}.csv"
    params_csv = out_dir / f"best_params_{ROUTE_A_NAME}.csv"
    fold_df.to_csv(fold_csv, index=False)
    perf_table.to_csv(agg_csv)
    params_df.to_csv(params_csv, index=False)
    logging.info(f"[{ROUTE_A_NAME}] saved:\n - {fold_csv}\n - {agg_csv}\n - {params_csv}")


# =========================================================
# Route B: GCN after KMeans undersampling
# =========================================================
def calc_metrics_prob(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):
    """
    Compute binary-classification metrics from probability scores.
    - Uses ROC-AUC and PRC-AUC from continuous scores (not labels).
    - Includes protections for degenerate/constant scores.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    if y_true.size == 0:
        keys = [
            "ROC-AUC", "PRC-AUC", "Accuracy", "F1", "MCC",
            "Recall", "Precision", "False Positives", "False Positive Rate",
        ]
        return {k: float("nan") for k in keys}

    # hard labels for threshold-based metrics
    y_hat = (y_prob >= thr).astype(int)

    # --- ROC-AUC with protection for constant scores ---
    if np.all(y_prob == y_prob[0]):
        roc = 0.5  # no discrimination
        pr_auc = float(np.mean(y_true))  # fallback to prevalence baseline
    else:
        roc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # auc expects x=recall, y=precision
        pr_auc = auc(recall, precision)

    acc = accuracy_score(y_true, y_hat)
    f1  = f1_score(y_true, y_hat, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_hat)
    rec = recall_score(y_true, y_hat, zero_division=0)
    pre = precision_score(y_true, y_hat, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "ROC-AUC": roc,
        "PRC-AUC": pr_auc,
        "Accuracy": acc,
        "F1": f1,
        "MCC": mcc,
        "Recall": rec,
        "Precision": pre,
        "False Positives": int(fp),
        "False Positive Rate": fpr,
    }


def run_route_B_gcn_kmeans(
    df: pd.DataFrame,
    out_dir: Path,
    seed: int,
    n_splits_outer: int,
    neg_pos_ratio: float,
    fp_bits: int = FP_BITS,
    fp_radius: int = FP_RADIUS
):
    logging.info("=== Route B: GCN after KMeans Undersampling ===")

    # Build graphs
    smiles = df["smiles"].astype(str).tolist()
    labels = df["antibiotic_activity"].astype(int).to_numpy()
    graphs, drop_idx = [], []
    for i, smi in enumerate(smiles):
        g = smiles_to_graph(smi)
        if g is None or g.x.numel() == 0:
            drop_idx.append(i); continue
        g.y = torch.tensor([labels[i]], dtype=torch.float32)
        graphs.append(g)
    if drop_idx:
        logging.warning(f"Dropped {len(drop_idx)} invalid SMILES during graph construction.")
        labels = np.delete(labels, drop_idx, axis=0)
        df = df.drop(index=drop_idx).reset_index(drop=True)

    # Fingerprints for KMeans space
    if "fingerprint" not in df.columns:
        gen = get_morgan_generator(radius=fp_radius, fp_size=fp_bits)
        df = df.copy()
        df["fingerprint"] = df["smiles"].apply(lambda s: smiles_to_fp(s, gen, fp_bits))

    # Indices after drop
    pos_idx_all = np.where(labels == 1)[0]
    neg_idx_all = np.where(labels == 0)[0]

    pos_df = df.iloc[pos_idx_all].reset_index(drop=True)
    neg_df = df.iloc[neg_idx_all].reset_index(drop=True)

    # GCN dims
    g0 = graphs[0]
    in_dim = g0.x.size(1)
    edge_dim = g0.edge_attr.size(1) if g0.edge_attr is not None else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
    fold_rows = []
    best_score = -1.0
    best_model_path = out_dir / "best_gine_kmeans.pth"

    for fold_id, (pos_tr_idx, pos_te_idx) in enumerate(kf_outer.split(pos_df), start=1):
        logging.info(f"[{ROUTE_B_NAME}] Fold {fold_id}/{n_splits_outer}")

        # Build fold sets in Data objects space
        pos_tr_idx_glob = pos_idx_all[pos_tr_idx]
        pos_te_idx_glob = pos_idx_all[pos_te_idx]

        # Negative split (independent)
        neg_tr_all, neg_te_all = train_test_split(
            neg_df, test_size=0.2, random_state=seed + fold_id, shuffle=True
        )
        # Map to global indices for graphs / labels
        neg_tr_glob = neg_idx_all[neg_tr_all.index.values]
        neg_te_glob = neg_idx_all[neg_te_all.index.values]

        # KMeans undersampling in FP space (train/test separately)
        n_neg_tr = int(round(neg_pos_ratio * len(pos_tr_idx_glob)))
        n_neg_te = int(round(neg_pos_ratio * len(pos_te_idx_glob)))

        neg_tr_fp = np.vstack(neg_tr_all["fingerprint"].values)
        neg_te_fp = np.vstack(neg_te_all["fingerprint"].values)

        idx_sel_tr = kmeans_select_in_fp(neg_tr_fp, n_target=n_neg_tr, random_state=seed + fold_id)
        idx_sel_te = kmeans_select_in_fp(neg_te_fp, n_target=n_neg_te, random_state=seed + fold_id)

        neg_tr_sel_glob = neg_tr_glob[idx_sel_tr]
        neg_te_sel_glob = neg_te_glob[idx_sel_te]

        # Build balanced Data lists for this fold
        train_ids = np.concatenate([pos_tr_idx_glob, neg_tr_sel_glob])
        test_ids  = np.concatenate([pos_te_idx_glob, neg_te_sel_glob])

        train_data = [graphs[i] for i in train_ids]
        test_data  = [graphs[i] for i in test_ids]

        # Stratified split train->train/val
        y_train_all = labels[train_ids]
        tr_part_idx, val_part_idx = train_test_split(
            np.arange(len(train_data)), test_size=VAL_SPLIT, stratify=y_train_all, random_state=seed
        )
        train_part = [train_data[i] for i in tr_part_idx]
        val_part   = [train_data[i] for i in val_part_idx]

        # DataLoaders
        train_loader = DataLoader(train_part, batch_size=BATCH_SIZE_CLS, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_part,   batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=0)

        # Model, loss, optim
        model = GINEClassifier(in_dim, edge_dim, hidden=HIDDEN, num_layers=NUM_GINE, dropout=DROPOUT).to(device)
        pos_count = float((labels[train_ids] == 1).sum()); neg_count = float((labels[train_ids] == 0).sum())
        pos_w = torch.tensor([(neg_count / max(pos_count, 1.0))], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_GCN, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        best_val, bad = -1.0, 0
        best_state = None

        for ep in range(1, EPOCHS_GCN + 1):
            loss = train_one_epoch_gcn(model, train_loader, optimizer, criterion, device)
            yv, pv = infer_proba_gcn(model, val_loader, device)
            if yv.size > 0:
                prec, rec, _ = precision_recall_curve(yv, pv)
                val_ap = auc(rec, prec)
            else:
                val_ap = 0.0
            scheduler.step(val_ap)
            logging.info(f"[{ROUTE_B_NAME}|Fold {fold_id}] Ep {ep:03d} | train loss {loss:.4f} | val AP {val_ap:.4f}")

            if val_ap > best_val + 1e-5:
                best_val, bad = val_ap, 0
                best_state = model.state_dict()
            else:
                bad += 1
                if bad >= PATIENCE and ep >= 30:
                    logging.info(f"[{ROUTE_B_NAME}|Fold {fold_id}] Early stop.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        yt, pt = infer_proba_gcn(model, test_loader, device)
        m = calc_metrics_prob(yt, pt, thr=0.5)
        fold_rows.append({"Route": ROUTE_B_NAME, "Fold": fold_id, **m})

        # Track best by ROC-AUC
        if m["ROC-AUC"] > best_score:
            best_score = m["ROC-AUC"]
            torch.save(model.state_dict(), best_model_path)

    # Save outputs
    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in {"Route", "Fold"}]
    agg_spec = {mc: ["mean", "std"] for mc in metric_cols}
    g = fold_df.groupby(["Route"]).agg(agg_spec)
    g.columns = [" ".join(col).strip() for col in g.columns.values]

    def mean_sd_str(m, s):
        if pd.isna(m) or pd.isna(s): return "nan"
        return f"{m:.4f}±{s:.4f}"

    perf_table = pd.DataFrame(index=g.index)
    for mc in metric_cols:
        mcol, scol = f"{mc} mean", f"{mc} std"
        if mcol in g.columns and scol in g.columns:
            perf_table[mc] = [mean_sd_str(m, s) for m, s in zip(g[mcol], g[scol])]

    fold_csv = out_dir / f"fold_results_{ROUTE_B_NAME}.csv"
    agg_csv  = out_dir / f"agg_performance_{ROUTE_B_NAME}.csv"
    fold_df.to_csv(fold_csv, index=False)
    perf_table.to_csv(agg_csv)
    logging.info(f"[{ROUTE_B_NAME}] saved:\n - {fold_csv}\n - {agg_csv}\n - {best_model_path}")


# =========================================================
# Main
# =========================================================
def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Cross-strategy benchmark: ML(GraphSim) vs GCN(KMeans)")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with 'smiles' and 'antibiotic_activity'.")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR), help="Output directory.")
    parser.add_argument("--splits", type=int, default=5, help="Outer KFold splits on positives.")
    parser.add_argument("--inner_cv", type=int, default=5, help="Inner CV splits for ML GridSearchCV.")
    parser.add_argument("--neg_pos_ratio", type=float, default=NEG_POS_RATIO, help="Target #neg = ratio * #pos.")
    parser.add_argument("--fp_bits", type=int, default=FP_BITS, help="Morgan FP size.")
    parser.add_argument("--fp_radius", type=int, default=FP_RADIUS, help="Morgan FP radius.")
    parser.add_argument("--pick", type=str, choices=["nearest", "farthest"], default="nearest",
                        help="For Route A: pick negatives nearest/farthest to the positive centroid.")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default="cosine",
                        help="For Route A: similarity metric for centroid selection.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)
    required = {"smiles", "antibiotic_activity"}
    missing = required - set(df.columns)
    assert not missing, f"CSV missing required columns: {missing}"

    # Route A: Classic ML after Graph-Similarity undersampling
    run_route_A_ml_graphsim(
        df=df.copy(),
        out_dir=out_dir,
        seed=args.seed,
        n_splits_outer=args.splits,
        inner_cv_splits=args.inner_cv,
        neg_pos_ratio=args.neg_pos_ratio,
        pick=args.pick,
        metric=args.metric,
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius
    )

    # Route B: GCN after KMeans undersampling
    run_route_B_gcn_kmeans(
        df=df.copy(),
        out_dir=out_dir,
        seed=args.seed,
        n_splits_outer=args.splits,
        neg_pos_ratio=args.neg_pos_ratio,
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius
    )


if __name__ == "__main__":
    main()
