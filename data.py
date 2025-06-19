import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem

ATOM_FEATURES = [
    lambda a: a.GetAtomicNum(),
    lambda a: a.GetDegree(),
    lambda a: a.GetFormalCharge(),
    lambda a: int(a.GetHybridization()),
    lambda a: int(a.GetIsAromatic()),
]


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # node features
    node_feats = [[f(atom) for f in ATOM_FEATURES] for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    # edges
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        attrs += [[b.GetBondTypeAsDouble()]] * 2
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float) if attrs else torch.empty((0, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GraphDataset(Dataset):
    """Wraps a list of PyG `Data` objects."""

    def __init__(self, df: pd.DataFrame):
        self.graphs = []
        for _, row in df.iterrows():
            g = smiles_to_graph(row["smiles"])
            if g is not None:
                g.y = torch.tensor([row["antibiotic_activity"]], dtype=torch.float)
                self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
        