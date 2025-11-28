# -*- coding: utf-8 -*-
"""
GINE + Graph Autoencoder (GAE) undersampling pipeline with explainability
for molecular activity prediction.

Pipeline
--------
1) RDKit: convert SMILES -> molecular graphs with rich node/edge features
2) Train a Graph Autoencoder (reconstruct node features) to obtain graph embeddings
3) Undersample negatives by similarity to the positive centroid in embedding space:
   - 'nearest'  : pick the most similar negatives (smallest difference)  [DEFAULT]
   - 'farthest' : pick the most dissimilar negatives (largest difference)
   Supports cosine (default) or euclidean distance; negative:positive ~ 1:1 (configurable)
4) Train a GINE-based graph classifier with Stratified 5-Fold cross-validation
5) Report and save metrics per fold and overall:
   ROC-AUC, PRC-AUC, Accuracy, F1, MCC, Recall, Precision, False Positives, False Positive Rate
6) Save best-fold model weights and optional embeddings CSV
7) Explainability (optional via CLI):
   - Integrated Gradients for node features and edge features
   - Global feature importance (semantic names)
   - Highlight most important atoms/bonds on molecule images

Requirements
------------
- rdkit
- torch, torch_geometric (>= 2.2 recommended)
- scikit-learn, pandas, numpy, tqdm
- rdkit.Chem.Draw (for visualization)

Input
-----
CSV with:
- 'smiles' (string)
- 'antibiotic_activity' (0/1)

Usage
-----
Train + explain top 5 molecules (save PNGs to ./explain_out):
    python train_gcn_gae_pipeline.py --csv ./data/raw_data.csv \
        --pick nearest --metric cosine --ratio 1.0 \
        --explain 5 --explain-steps 64 --explain-save-dir ./explain_out
"""

import os
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn

# PyG
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except Exception:  # backward compatibility
    from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, GINEConv, BatchNorm

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
DEFAULT_CSV = "./data/raw_data.csv"
RESULT_CSV = "cv_results.csv"
FOLD_DETAIL_CSV = "cv_per_fold.csv"
BEST_MODEL_PATH = "best_gine_model.pth"
EMBED_CSV = "graph_embeddings.csv"

BATCH_SIZE_AE = 64
BATCH_SIZE_CLS = 64
EPOCHS_AE = 60
EPOCHS_CLS = 120
PATIENCE = 15
LR_AE = 1e-3
LR_CLS = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN = 128
NUM_GINE = 3
VAL_SPLIT = 0.10
DIST_METRIC = "cosine"        # 'cosine' or 'euclidean'
NEG_POS_RATIO = 1.0
DEFAULT_PICK = "nearest"      # 'nearest' or 'farthest'

# Common atomic numbers; everything else goes to "other"
COMMON_Z = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 19, 11, 12, 20, 26, 29, 30, 35, 53]
# H,B,C,N,O,F,Si,P,S,Cl,K,Na,Mg,Ca,Fe,Cu,Zn,Br,I

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(val, choices):
    vec = [0] * len(choices)
    if val in choices:
        vec[choices.index(val)] = 1
    return vec


def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """
    Node features:
      - atomic number: one-hot (COMMON_Z + 'other')
      - degree: one-hot [0..5]
      - hybridization: one-hot {sp, sp2, sp3, sp3d, sp3d2, other}
      - formal charge: one-hot [-2..2]
      - total hydrogens: one-hot [0..4]
      - aromatic (bool)
      - in ring (bool)
      - chirality tag: one-hot {unspecified, CW, CCW}
    """
    z = atom.GetAtomicNum()
    z_onehot = one_hot(z if z in COMMON_Z else -1, COMMON_Z + [-1])

    degree = atom.GetTotalDegree()
    degree_onehot = one_hot(min(degree, 5), list(range(6)))

    hyb = atom.GetHybridization()
    hyb_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb_onehot = one_hot(hyb if hyb in hyb_choices else None, hyb_choices + [None])

    charge = int(atom.GetFormalCharge())
    charge = max(-2, min(2, charge))
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
        chiral_choices,
    )

    return (
        z_onehot
        + degree_onehot
        + hyb_onehot
        + charge_onehot
        + num_h_onehot
        + aromatic
        + ring
        + chiral_onehot
    )


def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    """
    Edge features:
      - bond type: one-hot {single,double,triple,aromatic,other}
      - conjugated (bool)
      - in ring (bool)
      - stereo: one-hot {none, Z, E}
    """
    bt = bond.GetBondType()
    bt_choices = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    bt_onehot = one_hot(bt if bt in bt_choices else None, bt_choices + [None])

    conj = [int(bond.GetIsConjugated())]
    ring = [int(bond.IsInRing())]

    stereo = bond.GetStereo()
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    stereo_onehot = one_hot(
        stereo if stereo in stereo_choices else Chem.rdchem.BondStereo.STEREONONE, stereo_choices
    )

    return bt_onehot + conj + ring + stereo_onehot


def smiles_to_graph(smiles: str):
    """Build a PyG `Data` object from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Keep aromatic flags; 2D coords are sufficient for this pipeline
    Chem.Kekulize(mol, clearAromaticFlags=False)
    AllChem.Compute2DCoords(mol)

    # Node features
    x = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    # Edges + edge features (bidirectional)
    edge_index = []
    edge_attr = []
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
    g.smiles = smiles  # keep for visualization/explainability
    return g

# -----------------------------
# Models
# -----------------------------
def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


class GINEEncoder(nn.Module):
    """GINE encoder with BatchNorm, dropout and a light residual connection."""

    def __init__(self, in_dim: int, edge_dim: int, hidden: int = HIDDEN, num_layers: int = NUM_GINE, dropout: float = DROPOUT):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GINEConv(mlp(in_dim, hidden), edge_dim=edge_dim))
        self.bns.append(BatchNorm(hidden))

        # Subsequent layers
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
                h = h + 0.1 * h_res  # tiny residual
        return h  # node embeddings


class GraphAE(nn.Module):
    """Graph Autoencoder: node encoder -> reconstruct node features; returns graph embedding."""

    def __init__(self, in_dim: int, edge_dim: int, hidden: int = HIDDEN, num_layers: int = NUM_GINE, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = GINEEncoder(in_dim, edge_dim, hidden, num_layers, dropout)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr)  # [N, hidden]
        x_hat = self.decoder(h)                     # [N, in_dim]
        g = global_mean_pool(h, batch)              # [B, hidden]
        return x_hat, g

    def encode_nodes(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)


class GINEClassifier(nn.Module):
    """Graph-level classifier; encoder can be initialized from a trained AE."""

    def __init__(self, in_dim: int, edge_dim: int, hidden: int = HIDDEN, num_layers: int = NUM_GINE, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = GINEEncoder(in_dim, edge_dim, hidden, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr)
        g = global_mean_pool(h, batch)
        logit = self.head(g).view(-1)
        return logit

    def load_from_ae(self, ae: GraphAE):
        self.encoder.load_state_dict(ae.encoder.state_dict(), strict=False)

# -----------------------------
# Training & evaluation helpers
# -----------------------------
def train_ae(ae: GraphAE, loader: DataLoader, device, epochs: int = EPOCHS_AE, lr: float = LR_AE,
             wd: float = WEIGHT_DECAY, patience: int = PATIENCE) -> GraphAE:
    """Train GraphAE with node feature reconstruction loss (MSE)."""
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
        print(f"[AE] Epoch {ep:03d} | recon MSE {mean_loss:.5f} | lr {opt.param_groups[0]['lr']:.2e}")

        if mean_loss < best_loss - 1e-5:
            best_loss, bad = mean_loss, 0
            torch.save(ae.state_dict(), "best_graph_ae.pth")
        else:
            bad += 1
            if bad >= patience and ep >= 20:
                print("[AE] Early stop.")
                break

    ae.load_state_dict(torch.load("best_graph_ae.pth", map_location=device))
    ae.eval()
    return ae


@torch.no_grad()
def get_graph_embeddings(ae: GraphAE, loader: DataLoader, device) -> np.ndarray:
    """Return graph embeddings [N_graphs, hidden] from a trained AE encoder."""
    ae.eval()
    embs = []
    for data in loader:
        data = data.to(device)
        h = ae.encode_nodes(data.x, data.edge_index, data.edge_attr)
        g = global_mean_pool(h, data.batch)  # [B, hidden]
        embs.append(g.cpu().numpy())
    return np.concatenate(embs, axis=0)


def select_negatives_by_similarity(
    embs: np.ndarray,
    labels: np.ndarray,
    metric: str = DIST_METRIC,
    ratio: float = NEG_POS_RATIO,
    pick: str = DEFAULT_PICK
) -> np.ndarray:
    """Select negatives relative to the positive centroid in the embedding space."""
    assert pick in ("nearest", "farthest"), "pick must be 'nearest' or 'farthest'"

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.ones_like(labels, dtype=bool)

    pos_centroid = embs[pos_idx].mean(axis=0, keepdims=True)
    neg_embs = embs[neg_idx]

    if metric == "cosine":
        a = neg_embs / (np.linalg.norm(neg_embs, axis=1, keepdims=True) + 1e-9)
        b = pos_centroid / (np.linalg.norm(pos_centroid, axis=1, keepdims=True) + 1e-9)
        sim = (a @ b.T).reshape(-1)  # larger = more similar
        order = np.argsort(-sim) if pick == "nearest" else np.argsort(sim)
    elif metric == "euclidean":
        dist = np.linalg.norm(neg_embs - pos_centroid, axis=1)  # smaller = more similar
        order = np.argsort(dist) if pick == "nearest" else np.argsort(-dist)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    k = int(round(len(pos_idx) * ratio))
    chosen_neg = neg_idx[order[:k]]

    mask = np.zeros_like(labels, dtype=bool)
    mask[pos_idx] = True
    mask[chosen_neg] = True
    return mask


def train_one_epoch_cls(model: nn.Module, loader: DataLoader, optimizer, criterion, device) -> float:
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
def infer(model: nn.Module, loader: DataLoader, device):
    model.eval()
    ys, ps = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        prob = torch.sigmoid(logits)
        ys.append(data.y.view(-1).cpu().numpy())
        ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    return y_true, y_pred


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Return required metrics, including FP and FPR."""
    if y_true.size == 0:
        keys = [
            "ROC-AUC", "PRC-AUC", "Accuracy", "F1", "MCC",
            "Recall", "Precision", "False Positives", "False Positive Rate",
        ]
        return {k: float("nan") for k in keys}

    y_hat = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    fpr = fp / max((fp + tn), 1)

    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PRC-AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_hat),
        "F1": f1_score(y_true, y_hat, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_hat),
        "Recall": recall_score(y_true, y_hat, zero_division=0),
        "Precision": precision_score(y_true, y_hat, zero_division=0),
        "False Positives": int(fp),
        "False Positive Rate": fpr,
    }

# -----------------------------
# Explainability helpers
# -----------------------------
def build_node_feature_names() -> List[str]:
    """Return semantic names for each dimension in node features (must match atom_features order)."""
    z_names = [f"Z={Chem.GetPeriodicTable().GetElementSymbol(z)}" for z in COMMON_Z] + ["Z=other"]
    deg_names = [f"deg={d}" for d in range(6)]
    hyb_names = ["hyb=sp", "hyb=sp2", "hyb=sp3", "hyb=sp3d", "hyb=sp3d2", "hyb=other"]
    charge_names = [f"charge={c}" for c in [-2, -1, 0, 1, 2]]
    numh_names = [f"numH={h}" for h in range(5)]
    flags = ["aromatic", "in_ring"]
    chiral = ["chiral=unspecified", "chiral=CW", "chiral=CCW"]
    return z_names + deg_names + hyb_names + charge_names + numh_names + flags + chiral


def build_edge_feature_names() -> List[str]:
    """Return semantic names for each dimension in edge features (must match bond_features order)."""
    bond_names = ["bond=single", "bond=double", "bond=triple", "bond=aromatic", "bond=other"]
    flags = ["conjugated", "bond_in_ring"]
    stereo = ["stereo=none", "stereo=Z", "stereo=E"]
    return bond_names + flags + stereo


def integrated_gradients_graph(
    model: nn.Module,
    data: Data,
    steps: int = 64,
    device: str = "cpu",
    baseline_x: torch.Tensor = None,
    baseline_e: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrated Gradients for a single-graph Data on node features (x) and edge features (edge_attr).
    Returns:
        attr_x: [N_nodes, x_dim]
        attr_e: [N_edges, e_dim] or None if edge_attr is None
    """
    model.eval()
    device = torch.device(device)

    # Prepare inputs
    x = data.x.clone().detach().to(device)
    e = data.edge_attr.clone().detach().to(device) if data.edge_attr is not None else None
    edge_index = data.edge_index.to(device)
    batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=device))

    if baseline_x is None:
        baseline_x = torch.zeros_like(x, device=device)
    else:
        baseline_x = baseline_x.to(device)

    if e is not None:
        if baseline_e is None:
            baseline_e = torch.zeros_like(e, device=device)
        else:
            baseline_e = baseline_e.to(device)

    # Accumulators
    sum_grad_x = torch.zeros_like(x, device=device)
    sum_grad_e = torch.zeros_like(e, device=device) if e is not None else None

    for i in range(1, steps + 1):
        alpha = float(i) / steps

        # Make LEAF tensors for interpolation points
        x_int = (baseline_x + alpha * (x - baseline_x)).detach().requires_grad_(True)
        if e is not None:
            e_int = (baseline_e + alpha * (e - baseline_e)).detach().requires_grad_(True)
        else:
            e_int = None

        # Forward & scalar output for the single-graph batch
        logit = model(x_int, edge_index, e_int, batch).view(-1)[0]

        # Use autograd.grad to obtain gradients wrt x_int / e_int
        grads = torch.autograd.grad(
            outputs=logit,
            inputs=[x_int] + ([e_int] if e_int is not None else []),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,   # in case some inputs do not influence the output
        )

        grad_x = grads[0] if grads[0] is not None else torch.zeros_like(x_int)
        sum_grad_x = sum_grad_x + grad_x

        if e_int is not None:
            grad_e = grads[1] if len(grads) > 1 and grads[1] is not None else torch.zeros_like(e_int)
            sum_grad_e = sum_grad_e + grad_e

    # Average gradients across steps, then multiply by (input - baseline)
    avg_grad_x = sum_grad_x / steps
    attr_x = (x - baseline_x) * avg_grad_x

    if e is not None:
        avg_grad_e = sum_grad_e / steps
        attr_e = (e - baseline_e) * avg_grad_e
    else:
        attr_e = None

    return attr_x.detach(), (attr_e.detach() if attr_e is not None else None)



def aggregate_feature_importance_over_dataset(
    model: nn.Module,
    dataset: List[Data],
    device: str,
    steps: int = 64,
    max_samples: int = 256
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute global (dataset-level) feature importance by averaging |IG| over samples.
    Returns:
        node_feat_df: DataFrame with columns ['feature', 'abs_importance']
        edge_feat_df: DataFrame with columns ['feature', 'abs_importance']  (empty if no edges)
    """
    node_names = build_node_feature_names()
    edge_names = build_edge_feature_names()

    node_accum = np.zeros(len(node_names), dtype=float)
    edge_accum = np.zeros(len(edge_names), dtype=float)

    device = torch.device(device)
    model.eval()

    count_graphs = 0
    for g in tqdm(dataset[:max_samples], desc="IG(dataset)"):
        gg = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
        gg.batch = torch.zeros(gg.x.size(0), dtype=torch.long)
        gg = gg.to(device)
        attr_x, attr_e = integrated_gradients_graph(model, gg, steps=steps, device=device)

        # node feature importance: mean over nodes of |attr|
        node_imp = attr_x.abs().mean(dim=0).cpu().numpy()
        node_accum += node_imp

        # edge feature importance: mean over edges of |attr|
        if attr_e is not None and attr_e.numel() > 0:
            e_imp = attr_e.abs().mean(dim=0).cpu().numpy()
            # pad if current model edge_dim < edge_names (should not happen if feature def consistent)
            if len(e_imp) < len(edge_accum):
                tmp = np.zeros_like(edge_accum); tmp[:len(e_imp)] = e_imp; e_imp = tmp
            edge_accum += e_imp
        count_graphs += 1

    if count_graphs == 0:
        count_graphs = 1

    node_feat_df = pd.DataFrame({
        "feature": node_names[:len(node_accum)],
        "abs_importance": node_accum / count_graphs
    }).sort_values("abs_importance", ascending=False)

    # if no edge features, return empty df
    if edge_accum.sum() == 0:
        edge_feat_df = pd.DataFrame(columns=["feature", "abs_importance"])
    else:
        edge_feat_df = pd.DataFrame({
            "feature": edge_names[:len(edge_accum)],
            "abs_importance": edge_accum / count_graphs
        }).sort_values("abs_importance", ascending=False)

    return node_feat_df, edge_feat_df


def rank_nodes_and_edges_from_ig(attr_x: torch.Tensor, attr_e: torch.Tensor):
    """
    Collapse per-dimension attributions to per-node/per-edge scores (L1 norm).
    Returns:
        node_scores: [N_nodes] numpy
        edge_scores: [N_edges] numpy (None if attr_e is None)
    """
    node_scores = attr_x.abs().sum(dim=1).detach().cpu().numpy()
    edge_scores = attr_e.abs().sum(dim=1).detach().cpu().numpy() if attr_e is not None else None
    return node_scores, edge_scores


def draw_highlight_smiles(smiles: str, node_scores: np.ndarray, edge_index: torch.Tensor,
                          edge_scores: np.ndarray = None, topk_nodes: int = 10, topk_edges: int = 10,
                          out_png: str = "explain.png"):
    """
    Draw molecule with highlighted top-k atoms/bonds by importance scores.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    # top-k atoms
    node_order = np.argsort(-node_scores)
    hl_atoms = set(node_order[:min(topk_nodes, len(node_scores))].tolist())

    # prepare bond mapping (PyG edges are directed; RDKit bonds are undirected)
    bonds = []
    if edge_scores is not None and edge_index is not None and edge_index.size(1) > 0:
        # Merge the two directions by taking max importance among (i->j) and (j->i)
        m = {}
        for k in range(edge_index.size(1)):
            i, j = int(edge_index[0, k].item()), int(edge_index[1, k].item())
            if i == j:
                continue
            key = tuple(sorted((i, j)))
            score = edge_scores[k]
            m[key] = max(m.get(key, 0.0), score)
        # rank bonds
        uniq_pairs = list(m.items())
        uniq_pairs.sort(key=lambda x: -x[1])
        bonds = [pair for pair, _ in uniq_pairs[:min(topk_edges, len(uniq_pairs))]]

    # convert bonds to RDKit bond indices
    hl_bonds_idx = set()
    for i, j in bonds:
        b = mol.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            hl_bonds_idx.add(b.GetIdx())

    # RDKit draw
    atom_colors = {idx: (1.0, 0.2, 0.2) for idx in hl_atoms}  # red-ish for atoms
    bond_colors = {idx: (0.2, 0.2, 1.0) for idx in hl_bonds_idx}  # blue-ish for bonds
    drawer = Draw.MolDraw2DCairo(800, 600)
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(hl_atoms),
        highlightBonds=list(hl_bonds_idx),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    drawer.WriteDrawingText(out_png)

# -----------------------------
# Main
# -----------------------------
def main(args) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load data
    df = pd.read_csv(args.csv)
    assert "smiles" in df.columns and "antibiotic_activity" in df.columns, \
        "CSV must contain columns 'smiles' and 'antibiotic_activity'."
    smiles = df["smiles"].astype(str).tolist()
    labels = df["antibiotic_activity"].astype(int).to_numpy()

    # 2) Build graphs
    data_list: List[Data] = []
    drop_idx = []
    for i, smi in enumerate(tqdm(smiles, desc="SMILES->Graph")):
        g = smiles_to_graph(smi)
        if g is None or g.x.numel() == 0:
            drop_idx.append(i); continue
        g.y = torch.tensor([labels[i]], dtype=torch.float32)
        data_list.append(g)

    if len(data_list) == 0:
        raise RuntimeError("No valid molecules after SMILES->graph conversion.")

    if drop_idx:
        print(f"Warning: {len(drop_idx)} SMILES failed to convert and were skipped.")
        labels = np.delete(labels, drop_idx, axis=0)

    in_dim = data_list[0].x.size(1)
    edge_dim = data_list[0].edge_attr.size(1) if data_list[0].edge_attr is not None else 0
    print(f"Node feat dim = {in_dim} | Edge feat dim = {edge_dim} | N graphs = {len(data_list)}")

    # 3) Train Graph AE
    ae_loader = DataLoader(data_list, batch_size=BATCH_SIZE_AE, shuffle=True, num_workers=0)
    ae = GraphAE(in_dim, edge_dim, hidden=HIDDEN, num_layers=NUM_GINE, dropout=DROPOUT)
    ae = train_ae(ae, ae_loader, device)

    # 4) Embeddings & undersampling
    eval_loader = DataLoader(data_list, batch_size=BATCH_SIZE_AE, shuffle=False, num_workers=0)
    embs = get_graph_embeddings(ae, eval_loader, device)  # [N, hidden]
    pd.DataFrame(embs).to_csv(EMBED_CSV, index=False)

    pick_mode = getattr(args, "pick", DEFAULT_PICK)
    dist_metric = getattr(args, "metric", DIST_METRIC)
    ratio = float(getattr(args, "ratio", NEG_POS_RATIO))

    mask = select_negatives_by_similarity(
        embs, labels, metric=dist_metric, ratio=ratio, pick=pick_mode
    )
    data_balanced = [d for d, m in zip(data_list, mask) if m]
    labels_balanced = labels[mask]
    print(
        f"After undersampling by '{pick_mode}' ({dist_metric}) similarity: "
        f"pos={labels_balanced.sum()} neg={(1 - labels_balanced).sum()} total={len(data_balanced)}"
    )

    # 5) 5-fold CV training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []
    all_true, all_prob = [], []
    best_score = -1.0

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.arange(len(data_balanced)), labels_balanced), 1):
        train_subset = [data_balanced[i] for i in tr_idx]
        test_subset  = [data_balanced[i] for i in te_idx]
        y_train = labels_balanced[tr_idx]
        y_test  = labels_balanced[te_idx]

        tr_part, val_part, _, _ = train_test_split(
            train_subset, y_train, test_size=VAL_SPLIT, stratify=y_train, random_state=SEED
        )

        train_loader = DataLoader(tr_part, batch_size=BATCH_SIZE_CLS, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_part, batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_subset, batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=0)

        model = GINEClassifier(in_dim, edge_dim, hidden=HIDDEN, num_layers=NUM_GINE, dropout=DROPOUT).to(device)
        model.load_from_ae(ae)

        pos_count = float((y_train == 1).sum()); neg_count = float((y_train == 0).sum())
        pos_weight = torch.tensor([(neg_count / max(pos_count, 1.0))], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR_CLS, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        best_val, bad = -1.0, 0
        best_state = None

        for ep in range(1, EPOCHS_CLS + 1):
            loss = train_one_epoch_cls(model, train_loader, optimizer, criterion, device)
            yv, pv = infer(model, val_loader, device)
            val_ap = average_precision_score(yv, pv) if yv.size > 0 else 0.0
            scheduler.step(val_ap)
            print(
                f"[Fold {fold}] Ep {ep:03d} | train loss {loss:.4f} | val AP {val_ap:.4f} "
                f"| lr {optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_ap > best_val + 1e-5:
                best_val, bad = val_ap, 0
                best_state = model.state_dict()
            else:
                bad += 1
                if bad >= PATIENCE and ep >= 30:
                    print(f"[Fold {fold}] Early stop.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        yt, pt = infer(model, test_loader, device)
        all_true.append(yt); all_prob.append(pt)
        m = calc_metrics(yt, pt, threshold=0.5)
        fold_metrics.append({"Fold": fold, **m})
        print(f"[Fold {fold}] Test metrics: {m}")

        if m["ROC-AUC"] > best_score:
            best_score = m["ROC-AUC"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    pd.DataFrame(fold_metrics).to_csv(FOLD_DETAIL_CSV, index=False)

    all_true = np.concatenate(all_true); all_prob = np.concatenate(all_prob)
    overall_metrics = calc_metrics(all_true, all_prob, threshold=0.5)

    print("\n========== Overall (5-fold aggregated) ==========")
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"Best fold model saved to: {BEST_MODEL_PATH}")

    out_row = {
        "Model": f"GINE(AE init) + {pick_mode} undersampling ({dist_metric}), ratio={ratio}",
        **overall_metrics,
    }
    pd.DataFrame([out_row]).to_csv(RESULT_CSV, index=False)
    print(f"\nPer-fold metrics -> {FOLD_DETAIL_CSV}")
    print(f"Overall metrics -> {RESULT_CSV}")

    # -----------------------------
    # 6) Optional explainability
    # -----------------------------
    if getattr(args, "explain", 0) and getattr(args, "explain", 0) > 0:
        explain_n = int(getattr(args, "explain", 0))
        explain_steps = int(getattr(args, "explain_steps", 64))
        save_dir = getattr(args, "explain_save_dir", "./explain_out")
        os.makedirs(save_dir, exist_ok=True)

        # Load the best model picked above
        expl_model = GINEClassifier(in_dim, edge_dim, hidden=HIDDEN, num_layers=NUM_GINE, dropout=DROPOUT).to(device)
        expl_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        expl_model.eval()

        # Pick top-N confident positives and negatives from the balanced set
        loader_all = DataLoader(data_balanced, batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=0)
        _, prob_all = infer(expl_model, loader_all, device)
        prob_all = prob_all.reshape(-1)
        idx_sorted_pos = np.argsort(-prob_all)  # most positive
        idx_sorted_neg = np.argsort(prob_all)   # most negative

        picks = []
        for i in idx_sorted_pos[:explain_n]:
            picks.append(("pos", i))
        for i in idx_sorted_neg[:explain_n]:
            picks.append(("neg", i))

        # Global feature importance (on a sample up to 256 graphs)
        node_df, edge_df = aggregate_feature_importance_over_dataset(
            expl_model, data_balanced, device=device, steps=explain_steps, max_samples=256
        )
        node_df.to_csv(os.path.join(save_dir, "global_node_feature_importance.csv"), index=False)
        edge_df.to_csv(os.path.join(save_dir, "global_edge_feature_importance.csv"), index=False)

        print(f"[Explain] Saved global feature importance CSVs to: {save_dir}")

        # Per-molecule IG + visualization
        for tag, idx in picks:
            g = data_balanced[int(idx)]
            gg = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
            gg.batch = torch.zeros(gg.x.size(0), dtype=torch.long)
            gg = gg.to(device)

            attr_x, attr_e = integrated_gradients_graph(expl_model, gg, steps=explain_steps, device=device)
            node_scores, edge_scores = rank_nodes_and_edges_from_ig(attr_x, attr_e)

            # Save raw per-feature attributions (mean over nodes/edges) for this molecule
            node_names = build_node_feature_names()
            per_node_feat = attr_x.abs().mean(dim=0).detach().cpu().numpy()
            pd.DataFrame({"feature": node_names[:len(per_node_feat)],
                          "abs_IG": per_node_feat}).sort_values("abs_IG", ascending=False)\
                .to_csv(os.path.join(save_dir, f"{tag}_mol{idx}_node_feat_IG.csv"), index=False)

            if attr_e is not None and attr_e.numel() > 0:
                edge_names = build_edge_feature_names()
                per_edge_feat = attr_e.abs().mean(dim=0).detach().cpu().numpy()
                pd.DataFrame({"feature": edge_names[:len(per_edge_feat)],
                              "abs_IG": per_edge_feat}).sort_values("abs_IG", ascending=False)\
                    .to_csv(os.path.join(save_dir, f"{tag}_mol{idx}_edge_feat_IG.csv"), index=False)

            # Draw highlight PNG
            out_png = os.path.join(save_dir, f"{tag}_mol{idx}.png")
            draw_highlight_smiles(
                g.smiles, node_scores, g.edge_index.cpu(),
                edge_scores=edge_scores, topk_nodes=10, topk_edges=10, out_png=out_png
            )
            print(f"[Explain] Saved {out_png}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help="Path to CSV with columns 'smiles' and 'antibiotic_activity'.")
    parser.add_argument("--pick", type=str, choices=["nearest", "farthest"], default=DEFAULT_PICK,
                        help="Negative selection mode relative to positive centroid (default: nearest).")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default=DIST_METRIC,
                        help="Similarity metric in embedding space (default: cosine).")
    parser.add_argument("--ratio", type=float, default=NEG_POS_RATIO,
                        help="Negative:positive ratio for undersampling (default: 1.0).")

    # Explainability options
    parser.add_argument("--explain", type=int, default=0,
                        help="If >0, generate IG explanations for top-N positives and negatives.")
    parser.add_argument("--explain-steps", type=int, default=64,
                        help="Number of IG Riemann steps (default: 64).")
    parser.add_argument("--explain-save-dir", type=str, default="./explain_out",
                        help="Directory to save explanation outputs (PNGs and CSVs).")

    # Parse-known to be notebook-friendly
    args, _ = parser.parse_known_args()
    main(args)
