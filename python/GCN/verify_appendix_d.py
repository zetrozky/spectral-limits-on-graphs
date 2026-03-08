import argparse
import io
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops

EPOCHS = 200
TRIALS = 10
LR = 0.01
WD = 5e-4
SPLIT_SEED_BASE = 1000

SPECTRAL_COUPLINGS = [1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -50.0, -100.0, -500.0, -1000.0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpectralModulationSAGE(MessagePassing):
    """
    'Lobotomized' SAGE: P(A) = A + cI
    Uses a single shared linear layer to enforce strict spectral isolation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_shared = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, c_val):
        # 1. Aggregation (A * X)
        # Note: edge_index must have NO self-loops for this to be pure A
        agg_neighbors = self.propagate(edge_index, x=x)

        # 2. Spectral Modulation (A*X + c*X)
        # This implements P(A) = A + cI
        out = self.lin_shared(agg_neighbors + (c_val * x))
        return out


def load_platonov_squirrel(device):
    print("\n[Protocol] Fetching Filtered Squirrel (Platonov et al., 2023)...")
    url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/squirrel_filtered.npz"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error downloading: {e}")
        exit(1)

    with np.load(io.BytesIO(response.content)) as f:
        x = torch.from_numpy(f['node_features']).float()
        y = torch.from_numpy(f['node_labels']).long()
        edge_index = torch.from_numpy(f['edges']).t().contiguous().long()

        # Select only Split 0 ([:, 0]) to get a 1D mask [Nodes]
        # otherwise you get [Nodes, 10] and the training loop crashes.
        train_masks = torch.from_numpy(f['train_masks']).t().contiguous()[:, 0].bool()
        val_masks = torch.from_numpy(f['val_masks']).t().contiguous()[:, 0].bool()
        test_masks = torch.from_numpy(f['test_masks']).t().contiguous()[:, 0].bool()

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks

    print(f"[Status] Loaded Filtered Squirrel: {data.num_nodes} nodes, {data.num_edges} edges.")
    return data.to(device)


def load_data(dataset_name):
    """
    Loads data and performs 'Sanitization' (Removal of Self-Loops)
    to ensure precise control over the spectral operator.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))

    if dataset_name.lower() in ['cora', 'citeseer']:
        # PLANETOID (Standard Benchmarks)
        # transform=None (Raw features as per Shi et al.)
        path = os.path.join(script_dir, 'data', 'Planetoid')
        dataset = Planetoid(root=path, name=dataset_name, transform=None)
        data = dataset[0]

    elif dataset_name.lower() == 'platonov-squirrel':
        data = load_platonov_squirrel(device)

    else:
        path = os.path.join(script_dir, 'new_data', dataset_name.lower())
        # CUSTOM TEXT FILES (Squirrel/Chameleon/etc.)
        print(f"Loading {dataset_name} from Custom Text Files (Raw)...")
        # Adjust paths to match your folder structure
        feature_file = os.path.join(path, "out1_node_feature_label.txt")
        edge_file = os.path.join(path, "out1_graph_edges.txt")

        # Load Features & Labels
        # Skip header lines if necessary (adjust skiprows)
        feat_data = np.genfromtxt(feature_file, dtype=str, skip_header=1)
        # Column 0 is ID, 1 is Feature String, 2 is Label

        # Parse Features
        features = []
        labels = []
        node_map = {}

        for idx, row in enumerate(feat_data):
            node_id = row[0]
            node_map[node_id] = idx
            feat_vec = [float(x) for x in row[1].split(',')]
            features.append(feat_vec)
            labels.append(int(row[2]))

        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # Load Edges
        edge_data = np.genfromtxt(edge_file, dtype=str, skip_header=1)
        src = [node_map[row[0]] for row in edge_data]
        dst = [node_map[row[1]] for row in edge_data]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Create Data Object
        data = Data(x=x, edge_index=edge_index, y=y)

        # Create standard masks (60/20/20 split)
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        train_len = int(num_nodes * 0.6)
        val_len = int(num_nodes * 0.2)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[indices[:train_len]] = True

        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[indices[train_len:train_len + val_len]] = True

        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[indices[train_len + val_len:]] = True

    # Remove self-loops from data so model's 'c' is the ONLY loop
    # If we don't do this, c=0 is actually c=1 (if data has loops)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    return data


def run_trial(data, num_features, num_classes, c_val, trial_idx):
    # Set Seed
    seed = SPLIT_SEED_BASE + trial_idx
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = SpectralModulationSAGE(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    data = data.to(device)

    # Train
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, c_val)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Test
    model.eval()
    out = model(data.x, data.edge_index, c_val)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    return acc


def calculate_mlp_limit(data, num_features, num_classes):
    """
    Calculates the MLP Limit (c -> -infinity).
    This simulates the model ignoring the graph entirely (A=0).
    """
    print("\nCalculating MLP Limit (Edges Removed)...")

    data = data.to(device)

    # Create an edge_index with NO edges to simulate c -> -inf
    # The limit of (A + cI) as c -> inf is effectively just I
    empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)

    accs = []
    for t in range(TRIALS):
        # We manually run a trial with c=1.0 but *no edges*.
        # This is equivalent to no graph at all.
        seed = SPLIT_SEED_BASE + t
        torch.manual_seed(seed)

        model = SpectralModulationSAGE(num_features, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            # Pass empty edges
            out = model(data.x, empty_edges, c_val=1.0)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        out = model(data.x, empty_edges, c_val=1.0)
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        accs.append(acc)

    return np.mean(accs), np.std(accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='citeseer, cora, squirrel, chameleon, platonov-squirrel')
    args = parser.parse_args()

    TARGET_DATASET = args.dataset
    print(f"--- Running Audit on {TARGET_DATASET.upper()} ---")

    data = load_data(TARGET_DATASET)
    num_features = data.x.shape[1]
    num_classes = int(data.y.max()) + 1

    # 1. MLP limit
    mlp_mean, mlp_std = calculate_mlp_limit(data, num_features, num_classes)
    print(f"MLP Limit | Mean: {mlp_mean:.4f} | Std: {mlp_std:.4f}")

    # 2. RUN THE SPECTRAL SWEEP
    means = []
    stds = []

    print(f"\nCoupling   | Mean      | Std      ")
    print("-" * 30)

    for c in SPECTRAL_COUPLINGS:
        accs = [run_trial(data, num_features, num_classes, c, t) for t in range(TRIALS)]
        mu, sigma = np.mean(accs), np.std(accs)
        means.append(mu)
        stds.append(sigma)
        print(f"{c:<10.1f} | {mu:<8.4f} | {sigma:<8.4f}")

    # 3. PLOT
    plt.figure(figsize=(12, 7), dpi=200)
    x_indices = range(len(SPECTRAL_COUPLINGS) + 1)

    plot_means = means + [mlp_mean]
    plot_stds = stds + [mlp_std]

    plt.errorbar(x_indices, plot_means, yerr=plot_stds, fmt='-o',
                 color='black', ecolor='gray', capsize=4, elinewidth=1.5,
                 label=f'Spectral SAGE ({TARGET_DATASET.title()})')

    plt.axhline(y=mlp_mean, color='red', linestyle='--', alpha=0.5, label='MLP Limit')
    plt.fill_between(x_indices, mlp_mean - mlp_std, mlp_mean + mlp_std, color='red', alpha=0.1)

    tick_labels = [str(c) for c in SPECTRAL_COUPLINGS] + [r'$-\infty$']
    plt.xticks(x_indices, tick_labels, rotation=45)
    plt.title(f"Audit: {TARGET_DATASET.upper()}")
    plt.xlabel("Spectral Coupling (c)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"verify_appendix_d_{TARGET_DATASET}.png")
    print(f"Plot saved to verify_appendix_d_{TARGET_DATASET}.png")
